//
//  tiled_shaders.metal
//  GuassianSplatting
//
//  Tiled rasterizer for training with proper gradient computation
//
//  IMPORTANT: Data format expectations:
//  - Scale: LOG space (we apply exp() here)
//  - Opacity: RAW pre-sigmoid (we apply sigmoid() here)
//  - Rotation: quaternion (w,x,y,z) stored as float4(.x=w, .y=x, .z=y, .w=z)
//
#include <metal_stdlib>
using namespace metal;

// CRITICAL: Must match C++ simd struct layout EXACTLY
// C++ simd_float3 is 12 bytes with 16-byte alignment
// Use packed_float3 in Metal which is exactly 12 bytes
// Layout: position(0-12), scale(16-28), rotation(32-48), opacity(48-52), sh(52-100), total 112
struct Gaussian {
    packed_float3 position; // offset 0, 12 bytes
    float _pad0;            // offset 12, 4 bytes padding (to align scale to 16)
    packed_float3 scale;    // offset 16, LOG scale, 12 bytes
    float _pad1;            // offset 28, 4 bytes padding (to align rotation to 32)
    float4 rotation;        // offset 32, (w,x,y,z) as (.x=w, .y=x, .z=y, .w=z)
    float opacity;          // offset 48, RAW pre-sigmoid
    float sh[12];           // offset 52, 48 bytes
};  // Total: 100 bytes, padded to 112 for struct alignment

// Projected Gaussian data for tiled rendering
// CRITICAL: Use packed_float3 to match C++ memory layout (12 bytes, not 16)
// float2 has 8-byte alignment, so we need explicit padding after tileMaxY
struct ProjectedGaussian {
    float2 screenPos;       // 8 bytes, offset 0
    packed_float3 conic;    // 12 bytes, offset 8 - Inverse 2D covariance
    float depth;            // 4 bytes, offset 20
    float opacity;          // 4 bytes, offset 24 - AFTER sigmoid (for rendering efficiency)
    packed_float3 color;    // 12 bytes, offset 28
    float radius;           // 4 bytes, offset 40
    uint tileMinX;          // 4 bytes, offset 44
    uint tileMinY;          // 4 bytes, offset 48
    uint tileMaxX;          // 4 bytes, offset 52
    uint tileMaxY;          // 4 bytes, offset 56
    float _pad1;            // 4 bytes, offset 60 - explicit padding for float2 alignment
    float2 viewPos_xy;      // 8 bytes, offset 64
    // Store cov2D for backward pass (needed for correct gradient)
    packed_float3 cov2D;    // 12 bytes, offset 72 - (a, b, c) - the 2D covariance BEFORE inversion
    float _pad2;            // 4 bytes, offset 84 - padding to make struct 88 bytes (multiple of 8)
};  // Total: 88 bytes

struct TileRange {
    uint start;
    uint count;
};

struct TiledUniforms {
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4x4 viewProjectionMatrix;
    float2 screenSize;
    float2 focalLength;
    float3 cameraPos;        // 16 bytes (includes implicit 4-byte padding)
    uint numTilesX;
    uint numTilesY;
    uint numGaussians;
    uint _pad2;
};

struct GaussianGradients {
    float position_x;
    float position_y;
    float position_z;
    float opacity;
    float scale_x;
    float scale_y;
    float scale_z;
    float _pad1;
    float4 rotation;
    float sh[12];
    
    // Viewspace (screen-space) gradients for density control
    // Official 3DGS uses these for densification decisions
    float viewspace_grad_x;
    float viewspace_grad_y;
    float _pad2;
    float _pad3;
};

constant float SH_C0 = 0.28209479177387814f;
constant uint TILE_SIZE = 16;
constant float MAX_RADIUS = 512.0f;
constant float MAX_SCALE = 5.0f;  // exp(5) ≈ 148 - more reasonable max scale

// Quaternion to rotation matrix
// q.x=w, q.y=x, q.z=y, q.w=z
float3x3 quatToMat(float4 q) {
    float w = q.x, x = q.y, y = q.z, z = q.w;
    // Metal float3x3 constructor takes COLUMNS
    return float3x3(
        float3(1.0 - 2.0*(y*y + z*z), 2.0*(x*y + w*z), 2.0*(x*z - w*y)), // Column 0
        float3(2.0*(x*y - w*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z + w*x)), // Column 1
        float3(2.0*(x*z + w*y), 2.0*(y*z - w*x), 1.0 - 2.0*(x*x + y*y))  // Column 2
    );
}

kernel void projectGaussians(
    device const Gaussian* gaussians [[buffer(0)]],
    device ProjectedGaussian* projected [[buffer(1)]],
    constant TiledUniforms& uniforms [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= uniforms.numGaussians) return;
    
    Gaussian g = gaussians[tid];
    ProjectedGaussian proj = {};
    proj.radius = 0;
    proj.tileMinX = UINT_MAX;
    proj.tileMaxX = 0;
    proj.tileMinY = UINT_MAX;
    proj.tileMaxY = 0;
    
    // Skip invalid Gaussians
    if (isnan(g.position.x) || isnan(g.position.y) || isnan(g.position.z) ||
        isnan(g.scale.x) || isnan(g.scale.y) || isnan(g.scale.z) ||
        abs(g.position.x) > 1e6 || abs(g.position.y) > 1e6 || abs(g.position.z) > 1e6) {
        projected[tid] = proj;
        return;
    }
    
    float4 worldPos = float4(g.position, 1.0);
    float4 viewPos = uniforms.viewMatrix * worldPos;
    float4 clipPos = uniforms.viewProjectionMatrix * worldPos;
    
    // COLMAP uses OpenCV convention: camera looks down +Z axis
    // Objects in FRONT of camera have POSITIVE viewZ
    // clipPos.w = viewZ (with our projection), so should be positive for visible objects
    if (clipPos.w <= 0.1 || viewPos.z <= 0.1) {
        projected[tid] = proj;
        return;
    }
    
    float3 ndc = clipPos.xyz / clipPos.w;
    
    // Outside frustum
    if (abs(ndc.x) > 1.2 || abs(ndc.y) > 1.2) {
        projected[tid] = proj;
        return;
    }
    
    // Screen position from NDC
    // COLMAP Y-down + our projection Y-flip means: 
    //   Top of COLMAP image (small pixel y) -> NDC y negative
    //   Bottom of COLMAP image (large pixel y) -> NDC y positive
    // Metal texture origin is top-left (row 0 = top)
    // Standard conversion: screenY = (ndc.y + 1) / 2 * height  
    // But we need to invert because of Y-flip: screenY = (1 - ndc.y) / 2 * height
    proj.screenPos = float2(
        (ndc.x * 0.5 + 0.5) * uniforms.screenSize.x,
        (ndc.y * 0.5 + 0.5) * uniforms.screenSize.y  // Standard NDC to screen (projection already flipped Y)
    );
    proj.depth = viewPos.z;  // Store positive depth (COLMAP: +Z is forward)
    proj.viewPos_xy = viewPos.xy;
    
    // ===== SCALE: Apply exp() to log scale =====
    float3 logScale = clamp(g.scale, -MAX_SCALE, MAX_SCALE);
    float3 scale = exp(logScale);
    
    // Prevent extremely elongated Gaussians (max 20:1 aspect ratio)
    float maxScale = max(max(scale.x, scale.y), scale.z);
    float minScale = min(min(scale.x, scale.y), scale.z);
    if (maxScale > 20.0f * minScale) {
        // Clamp the max scale to prevent extreme elongation
        float targetMax = 20.0f * minScale;
        scale = scale * (targetMax / maxScale);
    }
    
    // Normalize quaternion
    float4 q = g.rotation;
    float qLen = length(q);
    q = (qLen > 0.001) ? (q / qLen) : float4(1, 0, 0, 0);
    
    // Build 3D covariance: Sigma = R * S * S^T * R^T = M * M^T where M = R * S
    float3x3 R = quatToMat(q);
    // Diagonal scale matrix - Metal column constructor
    float3x3 S = float3x3(
        float3(scale.x, 0, 0),  // Column 0
        float3(0, scale.y, 0),  // Column 1
        float3(0, 0, scale.z)   // Column 2
    );
    float3x3 M = R * S;
    float3x3 Sigma3D = M * transpose(M);
    
    // View space projection using same approach as official 3DGS
    // COLMAP convention: objects in front have positive viewZ
    float z_cam = viewPos.z;
    float fx = uniforms.focalLength.x;
    float fy = uniforms.focalLength.y;
    
    // Clamp to avoid numerical issues at edges
    float limx = 1.3 * fx / z_cam;
    float limy = 1.3 * fy / z_cam;
    float txtz = clamp(viewPos.x / z_cam, -limx, limx);
    float tytz = clamp(viewPos.y / z_cam, -limy, limy);
    
    // Jacobian of projection (perspective projection derivative)
    // Maps 3D view space -> 2D screen space
    // Official 3DGS does NOT include Y flip in Jacobian for covariance
    // (covariance is a shape - ellipse looks same either way)
    float J00 = fx / z_cam;
    float J02 = -fx * txtz / z_cam;
    float J11 = fy / z_cam;
    float J12 = -fy * tytz / z_cam;
    
    // Metal float3x3 constructor takes COLUMNS
    // J = | J00  0    J02 |
    //     | 0    J11  J12 |
    //     | 0    0    0   |
    float3x3 J = float3x3(
        float3(J00, 0, 0),    // Column 0
        float3(0, J11, 0),    // Column 1
        float3(J02, J12, 0)   // Column 2
    );
    
    // View matrix rotation (world-to-view)
    // matrix[i] in Metal gives COLUMN i
    float3x3 W = float3x3(uniforms.viewMatrix[0].xyz,
                          uniforms.viewMatrix[1].xyz,
                          uniforms.viewMatrix[2].xyz);
    
    // Combined transform: T = J * W projects world covariance to screen
    float3x3 T = J * W;
    
    // Project 3D covariance to 2D: cov2D = T * Sigma3D * T^T
    float3x3 cov2D_mat = T * Sigma3D * transpose(T);
    
    // Metal matrix indexing: matrix[col][row]
    // For symmetric matrix: [0][0]=top-left, [1][1]=bottom-right, [0][1]=[1][0]=off-diagonal
    float a = cov2D_mat[0][0];  // Column 0, Row 0 = (0,0)
    float b = cov2D_mat[1][0];  // Column 1, Row 0 = (0,1) - the off-diagonal term
    float c = cov2D_mat[1][1];  // Column 1, Row 1 = (1,1)
    
    // Low-pass filter (add before storing for backward pass)
    a += 0.3;
    c += 0.3;
    
    // Store cov2D for backward pass AFTER low-pass filter
    proj.cov2D = float3(a, b, c);
    
    float det = a * c - b * b;
    if (det < 0.0001) {
        projected[tid] = proj;
        return;
    }
    
    // Conic (inverse covariance)
    float inv_det = 1.0 / det;
    proj.conic = float3(c * inv_det, -b * inv_det, a * inv_det);
    
    // Compute radius from eigenvalues
    float mid = 0.5 * (a + c);
    float disc = mid * mid - det;
    float l1 = mid + sqrt(max(0.1f, disc));
    float rawRadius = 3.0 * sqrt(l1);
    proj.radius = min(ceil(rawRadius), MAX_RADIUS);  // 2x for safety margin
    
    if (proj.radius <= 0) {
        projected[tid] = proj;
        return;
    }
    
    // Tile bounds
    float r = proj.radius;
    int minX = max(0, int(proj.screenPos.x - r));
    int minY = max(0, int(proj.screenPos.y - r));
    int maxX = min(int(uniforms.screenSize.x) - 1, int(proj.screenPos.x + r));
    int maxY = min(int(uniforms.screenSize.y) - 1, int(proj.screenPos.y + r));
    
    if (minX > maxX || minY > maxY) {
        proj.radius = 0;
        projected[tid] = proj;
        return;
    }
    
    proj.tileMinX = uint(minX) / TILE_SIZE;
    proj.tileMinY = uint(minY) / TILE_SIZE;
    proj.tileMaxX = min(uint(maxX) / TILE_SIZE, uniforms.numTilesX - 1);
    proj.tileMaxY = min(uint(maxY) / TILE_SIZE, uniforms.numTilesY - 1);
    
    // Limit tile coverage (increased from 64 to allow larger Gaussians)
    uint tilesX = proj.tileMaxX - proj.tileMinX + 1;
    uint tilesY = proj.tileMaxY - proj.tileMinY + 1;
    if (tilesX * tilesY > 256) {
        proj.radius = 0;
        projected[tid] = proj;
        return;
    }
    
    // ===== OPACITY: Apply sigmoid to raw opacity =====
    float rawOpacity = clamp(g.opacity, -8.0f, 8.0f);
    proj.opacity = 1.0 / (1.0 + exp(-rawOpacity));
    
    // Color from SH DC terms
    proj.color = clamp(float3(
        SH_C0 * g.sh[0] + 0.5f,
        SH_C0 * g.sh[4] + 0.5f,
        SH_C0 * g.sh[8] + 0.5f
    ), 0.0f, 1.0f);
    
    projected[tid] = proj;
}

kernel void tiledForward(
    device const Gaussian* gaussians [[buffer(0)]],
    device const ProjectedGaussian* projected [[buffer(1)]],
    device const uint* sortedIndices [[buffer(2)]],
    device const TileRange* tileRanges [[buffer(3)]],
    constant TiledUniforms& uniforms [[buffer(4)]],
    device uint* lastContribIdx [[buffer(5)]],
    texture2d<float, access::write> output [[texture(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= uint(uniforms.screenSize.x) || gid.y >= uint(uniforms.screenSize.y)) return;
    
    uint tileX = gid.x / TILE_SIZE;
    uint tileY = gid.y / TILE_SIZE;
    uint tileIdx = tileY * uniforms.numTilesX + tileX;
    TileRange range = tileRanges[tileIdx];
    
    float3 color = float3(0);
    float T = 1.0;
    float2 pixelPos = float2(gid) + 0.5;
    
    uint lastIdx = 0;
    bool hasContrib = false;
    
    // Rely purely on T termination instead of artificial cap
    for (uint i = 0; i < range.count && T > 0.0001; i++) {
        uint sortIdx = range.start + i;
        uint gIdx = sortedIndices[sortIdx];
        
        if (gIdx >= uniforms.numGaussians) continue;
        
        ProjectedGaussian p = projected[gIdx];
        
        if (p.radius <= 0) continue;
        
        float2 d = pixelPos - p.screenPos;
        
        // Debug: check if conic is valid
        float conicMag = abs(p.conic.x) + abs(p.conic.y) + abs(p.conic.z);
        if (conicMag < 0.0001) continue;  // Skip if conic is essentially zero
        
        float power = -0.5 * (p.conic.x * d.x * d.x +
                              2.0 * p.conic.y * d.x * d.y +
                              p.conic.z * d.y * d.y);
        
        if (power > 0.0 || power < -4.5) continue;
        
        float G = exp(power);
        float alpha = min(p.opacity * G, 0.99f);
        
        if (alpha < 1.0 / 255.0) continue;
        
        color += p.color * alpha * T;
        T *= (1.0 - alpha);
        
        lastIdx = sortIdx;
        hasContrib = true;
    }
    
    uint pixelIdx = gid.y * uint(uniforms.screenSize.x) + gid.x;
    lastContribIdx[pixelIdx] = hasContrib ? lastIdx : UINT_MAX;
    
    output.write(float4(color, 1.0 - T), gid);
}

kernel void tiledBackward(
    device const Gaussian* gaussians [[buffer(0)]],
    device GaussianGradients* gradients [[buffer(1)]],
    device const ProjectedGaussian* projected [[buffer(2)]],
    device const uint* sortedIndices [[buffer(3)]],
    device const TileRange* tileRanges [[buffer(4)]],
    constant TiledUniforms& uniforms [[buffer(5)]],
    device const uint* lastContribIdx [[buffer(6)]],
    texture2d<float, access::read> rendered [[texture(0)]],
    texture2d<float, access::read> groundTruth [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= uint(uniforms.screenSize.x) || gid.y >= uint(uniforms.screenSize.y)) return;
    
    uint pixelIdx = gid.y * uint(uniforms.screenSize.x) + gid.x;
    uint lastIdx = lastContribIdx[pixelIdx];
    
    if (lastIdx == UINT_MAX) return;
    
    uint tileX = gid.x / TILE_SIZE;
    uint tileY = gid.y / TILE_SIZE;
    uint tileIdx = tileY * uniforms.numTilesX + tileX;
    TileRange range = tileRanges[tileIdx];
    
    float2 pixelPos = float2(gid) + 0.5;
    
    float4 r_pix = rendered.read(gid);
    float4 gt_pix = groundTruth.read(gid);
    
    // L1 gradient
    float3 diff = r_pix.rgb - gt_pix.rgb;
    float3 dL_dPixel = sign(diff) / 3.0;
    
    // We need to traverse BACK-TO-FRONT for correct gradients
    // First, find how many Gaussians contribute and compute T_final
    // Use lastIdx from forward pass (which relied on T termination)
    uint endIdx = min(lastIdx + 1, range.start + range.count);
    
    // Pre-compute T (transmittance) for the forward pass to get T_final
    float T_final = 1.0;
    for (uint sortIdx = range.start; sortIdx < endIdx; sortIdx++) {
        uint gIdx = sortedIndices[sortIdx];
        if (gIdx >= uniforms.numGaussians) continue;
        
        ProjectedGaussian p = projected[gIdx];
        if (p.radius <= 0) continue;
        
        float2 d = pixelPos - p.screenPos;
        float power = -0.5 * (p.conic.x * d.x * d.x +
                              2.0 * p.conic.y * d.x * d.y +
                              p.conic.z * d.y * d.y);
        
        if (power > 0.0 || power < -4.5) continue;
        
        float G = exp(power);
        float alpha = min(p.opacity * G, 0.99f);
        
        if (alpha < 1.0 / 255.0) continue;
        
        float test_T = T_final * (1.0 - alpha);
        if (test_T < 0.0001) break;
        T_final = test_T;
    }
    
    // Now traverse BACK-TO-FRONT for gradients
    // T starts at T_final and we "undo" by dividing by (1-alpha)
    float T = T_final;
    float3 accum_rec = float3(0.0);  // Accumulated color from Gaussians BEHIND current (already processed)
    
    // Iterate in reverse order (back to front)
    for (int sortIdx = int(endIdx) - 1; sortIdx >= int(range.start); sortIdx--) {
        uint gIdx = sortedIndices[uint(sortIdx)];
        
        if (gIdx >= uniforms.numGaussians) continue;
        
        ProjectedGaussian p = projected[gIdx];
        
        if (p.radius <= 0) continue;
        
        float2 d = pixelPos - p.screenPos;
        float power = -0.5 * (p.conic.x * d.x * d.x +
                              2.0 * p.conic.y * d.x * d.y +
                              p.conic.z * d.y * d.y);
        
        if (power > 0.0 || power < -4.5) continue;
        
        float G = exp(power);
        float alpha = min(p.opacity * G, 0.99f);
        
        if (alpha < 1.0 / 255.0) continue;
        
        // Recover T at this position by "undoing" the alpha blend
        // In forward: T_next = T * (1 - alpha)
        // So: T = T_next / (1 - alpha)
        T = T / max(1.0 - alpha, 0.0001);
        
        float weight = alpha * T;
        
        // ===== Color/SH gradients =====
        // IMPORTANT: Account for clamping in forward pass
        // If color was clamped at 0 or 1, gradient should be 0
        float3 dL_dColor = dL_dPixel * weight;
        // If p.color is at 0 or 1 limit, zero the gradient to prevent SH explosion
        if (p.color.r <= 0.01 || p.color.r >= 0.99) dL_dColor.r = 0;
        if (p.color.g <= 0.01 || p.color.g >= 0.99) dL_dColor.g = 0;
        if (p.color.b <= 0.01 || p.color.b >= 0.99) dL_dColor.b = 0;
        
        // ===== Alpha gradient (CORRECT FORMULA from 3DGS paper) =====
        // The key insight: changing alpha affects:
        // 1. This Gaussian's own contribution: color * alpha * T
        // 2. All subsequent Gaussians' contributions via transmittance
        //
        // accum_rec tracks the accumulated color from Gaussians BEHIND this one
        // (already processed in back-to-front order)
        // dL/dalpha = T * dot(dL_dPixel, color - accum_rec)
        //
        // IMPORTANT: Use accum_rec BEFORE updating it (it represents what's behind current Gaussian)
        float dL_dAlpha = T * dot(dL_dPixel, p.color - accum_rec);
        
        // NOW update accum_rec to include current Gaussian for the next (closer) Gaussian
        // accum_rec_new = alpha * color + (1-alpha) * accum_rec_old
        // This represents: contribution from current Gaussian + attenuated contribution from behind
        accum_rec = alpha * p.color + (1.0 - alpha) * accum_rec;
        
        // ===== Opacity gradient =====
        // alpha = sigmoid(raw_opacity) * G
        // dalpha/d_raw_opacity = sigmoid * (1-sigmoid) * G
        float sig = p.opacity;  // This is ALREADY sigmoid (from projection)
        float dAlpha_dRawOp = sig * (1.0 - sig) * G;
        float dL_dRawOpacity = dL_dAlpha * dAlpha_dRawOp;
        
        // ===== Gaussian gradient =====
        // For position/scale gradients, we need dL/dG
        float dL_dG = dL_dAlpha * sig;
        
        // ===== Screen position gradient =====
        // d = pixel - screenPos (offset from Gaussian center to current pixel)
        // power = -0.5 * (conic.x * d.x^2 + 2 * conic.y * d.x * d.y + conic.z * d.y^2)
        // G = exp(power)
        //
        // dG/d(d) = G * dpower/d(d) = -G * [conic.x*dx + conic.y*dy, conic.y*dx + conic.z*dy]
        //
        // But we want dG/d(screenPos), and since d = pixel - screenPos:
        //   d(d)/d(screenPos) = -1
        //   dG/d(screenPos) = dG/d(d) * (-1) = G * [conic.x*dx + conic.y*dy, ...]
        //
        // This matches official 3DGS where they add dL_dG * dG_ddelx directly to dL_dmean2D
        // because dG_ddelx already accounts for the chain rule properly
        float gdx = G * d.x;
        float gdy = G * d.y;
        
        // Screen position gradient:
        // Your d = pixel - center, official uses d = center - pixel (opposite signs)
        //
        // At a pixel to the RIGHT of gaussian center:
        //   Your d.x = +positive, their d.x = -negative
        //   Your gdx = G * (+pos) = +positive, their gdx = G * (-neg) = -negative  
        //   Your dG_ddelx = -gdx * conic = -(+pos) = -negative
        //   Their dG_ddelx = -gdx * conic = -(-neg) = +positive
        //
        // Chain rule for dG/d(center):
        //   Your dd/d(center) = -1, their dd/d(center) = +1
        //   Your dG/d(center) = dG_ddelx * (-1) = (-neg) * (-1) = +positive
        //   Their dG/d(center) = dG_ddelx * (+1) = (+pos) * (+1) = +positive
        //
        // Both are positive (same physics), but computed values differ by sign!
        // Official code adds: dL_dG * dG_ddelx (their positive value)
        // Your code should add: dL_dG * (-dG_ddelx) = dL_dG * (positive value)
        float dG_ddelx = -gdx * p.conic.x - gdy * p.conic.y;
        float dG_ddely = -gdy * p.conic.z - gdx * p.conic.y;
        
        // Negate because your d convention is opposite of official
        float2 dL_dScreenPos = dL_dG * float2(-dG_ddelx, -dG_ddely);
        
        // ===== World position gradient =====
        // Chain rule: dL/dViewPos = dL/dScreenPos * dScreenPos/dViewPos
        // 
        // Forward pass (perspective projection):
        //   screenX ∝ fx * viewX / viewZ
        //   screenY ∝ fy * viewY / viewZ
        //
        // Derivatives (dScreenPos in pixels, dViewPos in world units):
        //   dScreenX/dViewX = fx / z
        //   dScreenY/dViewY = fy / z  
        //   dScreenX/dViewZ = -fx * (viewX/z) / z = -fx * txtz / z
        //   dScreenY/dViewZ = -fy * (viewY/z) / z = -fy * tytz / z
        float z = p.depth;
        float fx = uniforms.focalLength.x;
        float fy = uniforms.focalLength.y;
        
        // Compute normalized view ratios (same as forward pass Jacobian)
        float txtz = p.viewPos_xy.x / z;
        float tytz = p.viewPos_xy.y / z;
        
        float3 dL_dViewPos;
        dL_dViewPos.x = dL_dScreenPos.x * fx / z;
        dL_dViewPos.y = dL_dScreenPos.y * fy / z;
        dL_dViewPos.z = -dL_dScreenPos.x * fx * txtz / z
                        -dL_dScreenPos.y * fy * tytz / z;
        
        // Extract view rotation (columns = rotation matrix)
        // viewMatrix is world-to-view, so transpose gives view-to-world
        float3x3 viewRot = float3x3(
            uniforms.viewMatrix[0].xyz,
            uniforms.viewMatrix[1].xyz,
            uniforms.viewMatrix[2].xyz
        );
        // viewRot already contains columns of rotation, transpose gives view-to-world
        float3 dL_dWorldPos = transpose(viewRot) * dL_dViewPos;
        
        // ===== Conic gradient =====
        // power = -0.5 * (conic.x * dx^2 + 2 * conic.y * dx*dy + conic.z * dy^2)
        // dL/dconic = dL/dG * dG/dpower * dpower/dconic
        //           = dL_dG * G * (-0.5) * (dx^2, 2*dx*dy, dy^2)
        // Note: official 3DGS uses -0.5f factor for off-diagonal
        float3 dL_dConic;
        dL_dConic.x = -0.5f * dL_dG * G * d.x * d.x;
        dL_dConic.y = -0.5f * dL_dG * G * 2.0f * d.x * d.y;  // Factor of 2 because conic.y appears with 2* in power
        dL_dConic.z = -0.5f * dL_dG * G * d.y * d.y;
        
        // ===== Cov2D gradient (CORRECT formula from official backward.cu) =====
        // conic = inverse(cov2D), so dL/dCov2D requires derivative of matrix inverse
        // For 2x2 symmetric matrix [[a,b],[b,c]] with det = a*c - b*b:
        // inv = [[c, -b], [-b, a]] / det
        // The derivative is complex - use formula from backward.cu
        float cov_a = p.cov2D.x;
        float cov_b = p.cov2D.y;
        float cov_c = p.cov2D.z;
        
        float denom = cov_a * cov_c - cov_b * cov_b;
        float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);
        
        float3 dL_dCov2D;
        // From backward.cu: these formulas come from differentiating conic = inv(cov2D)
        dL_dCov2D.x = denom2inv * (-cov_c * cov_c * dL_dConic.x
                                   + 2.0f * cov_b * cov_c * dL_dConic.y
                                   + (denom - cov_a * cov_c) * dL_dConic.z);
        dL_dCov2D.z = denom2inv * (-cov_a * cov_a * dL_dConic.z
                                   + 2.0f * cov_a * cov_b * dL_dConic.y
                                   + (denom - cov_a * cov_c) * dL_dConic.x);
        dL_dCov2D.y = denom2inv * 2.0f * (cov_b * cov_c * dL_dConic.x
                                          - (denom + 2.0f * cov_b * cov_b) * dL_dConic.y
                                          + cov_a * cov_b * dL_dConic.z);
        
        // ===== Cov3D gradient =====
        // cov2D = T * Sigma3D * T^T where T = J * W (Jacobian * view rotation)
        // dL/dSigma3D = T^T * dL/dCov2D * T
        // But we need dL/dM where Sigma3D = M * M^T, so dL/dM = 2 * dL/dSigma3D * M
        float3 t_cam = float3(p.viewPos_xy, p.depth);
        txtz = t_cam.x / t_cam.z;
        tytz = t_cam.y / t_cam.z;
        
        // Jacobian (must match forward pass - NO Y flip for covariance)
        float J00 = fx / t_cam.z;
        float J02 = -fx * txtz / t_cam.z;
        float J11 = fy / t_cam.z;
        float J12 = -fy * tytz / t_cam.z;
        
        // T = J * W, compute T rows for gradient computation
        // T[0] = (J00, 0, J02) * W = J00 * W[0] + J02 * W[2]
        // T[1] = (0, J11, J12) * W = J11 * W[1] + J12 * W[2]
        float3 T0 = J00 * viewRot[0] + J02 * viewRot[2];
        float3 T1 = J11 * viewRot[1] + J12 * viewRot[2];
        
        // dL/dCov3D = T^T * dL/dCov2D_mat * T
        // For symmetric 2D covariance stored as (a, b, c):
        // dL_dCov2D_mat = [[dL_dCov2D.x, dL_dCov2D.y], [dL_dCov2D.y, dL_dCov2D.z]]
        // A = dL_dCov2D.x * T0 + dL_dCov2D.y * T1 (first row of dL_dCov2D_mat * T)
        // B = dL_dCov2D.y * T0 + dL_dCov2D.z * T1 (second row of dL_dCov2D_mat * T)
        float3 A = dL_dCov2D.x * T0 + dL_dCov2D.y * T1;
        float3 B = dL_dCov2D.y * T0 + dL_dCov2D.z * T1;
        
        // dL_dCov3D[i] = T0[i] * A + T1[i] * B (outer product sum)
        float3x3 dL_dCov3D;
        dL_dCov3D[0] = T0.x * A + T1.x * B;
        dL_dCov3D[1] = T0.y * A + T1.y * B;
        dL_dCov3D[2] = T0.z * A + T1.z * B;
        
        // ===== Scale and Rotation gradients =====
        // Sigma3D = M * M^T where M = R * S
        // dL/dM = 2 * dL/dSigma3D * M (for symmetric Sigma3D)
        Gaussian g_orig = gaussians[gIdx];
        float3 scale = exp(clamp(g_orig.scale, -MAX_SCALE, MAX_SCALE));
        float3x3 R = quatToMat(g_orig.rotation);
        float3x3 S = float3x3(
            float3(scale.x, 0, 0),
            float3(0, scale.y, 0),
            float3(0, 0, scale.z)
        );
        float3x3 M = R * S;
        
        // dL_dM = 2 * dL_dCov3D * M
        float3x3 dL_dM = 2.0f * dL_dCov3D * M;
        
        // Scale gradient: dL/dS = R^T * dL/dM
        // For diagonal S, we only need diagonal elements
        float3x3 Rt = transpose(R);
        float3x3 RtdLdM = Rt * dL_dM;
        float3 dL_dScale_val = float3(RtdLdM[0][0], RtdLdM[1][1], RtdLdM[2][2]);
        
        // Convert to log scale gradient: dL/d(log_s) = dL/ds * s
        float3 dL_dLogScale = dL_dScale_val * scale;
        
        // Rotation gradient: dL/dR = dL/dM * S^T = dL/dM * S (S is diagonal)
        float3x3 dL_dR = float3x3(
            dL_dM[0] * scale.x,
            dL_dM[1] * scale.y,
            dL_dM[2] * scale.z
        );
        
        // Quaternion gradient from rotation matrix gradient
        // Using the standard formula for dR/dq
        float4 dL_dq = float4(0);
        float w = g_orig.rotation.x;
        float x = g_orig.rotation.y;
        float y = g_orig.rotation.z;
        float z_rot = g_orig.rotation.w;
        
        // Derivatives of R w.r.t. quaternion components (from standard formulas)
        dL_dq.x = 2.0f * (
            dot(dL_dR[0], float3(0, z_rot, -y)) +
            dot(dL_dR[1], float3(-z_rot, 0, x)) +
            dot(dL_dR[2], float3(y, -x, 0))
        );
        dL_dq.y = 2.0f * (
            dot(dL_dR[0], float3(0, y, z_rot)) +
            dot(dL_dR[1], float3(y, -2*x, w)) +
            dot(dL_dR[2], float3(z_rot, -w, -2*x))
        );
        dL_dq.z = 2.0f * (
            dot(dL_dR[0], float3(-2*y, x, w)) +
            dot(dL_dR[1], float3(x, 0, z_rot)) +
            dot(dL_dR[2], float3(-w, z_rot, -2*y))
        );
        dL_dq.w = 2.0f * (
            dot(dL_dR[0], float3(-2*z_rot, -w, x)) +
            dot(dL_dR[1], float3(w, -2*z_rot, y)) +
            dot(dL_dR[2], float3(x, y, 0))
        );
        
        // Per-pixel gradient clamp - CRITICAL to prevent explosion
        // A Gaussian may receive gradients from 1000s of pixels, so this must be small
        float clampVal = 0.01f;
        
        // Atomic accumulation
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].sh[0],
            clamp(dL_dColor.r * SH_C0, -clampVal, clampVal), memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].sh[4],
            clamp(dL_dColor.g * SH_C0, -clampVal, clampVal), memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].sh[8],
            clamp(dL_dColor.b * SH_C0, -clampVal, clampVal), memory_order_relaxed);
        
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].opacity,
            clamp(dL_dRawOpacity, -clampVal, clampVal), memory_order_relaxed);
        
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].position_x,
            clamp(dL_dWorldPos.x, -clampVal, clampVal), memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].position_y,
            clamp(dL_dWorldPos.y, -clampVal, clampVal), memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].position_z,
            clamp(dL_dWorldPos.z, -clampVal, clampVal), memory_order_relaxed);
        
        // Viewspace (screen-space) gradients for density control
        // Official 3DGS uses norm of these 2D gradients for densification
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].viewspace_grad_x,
            clamp(dL_dScreenPos.x, -clampVal, clampVal), memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].viewspace_grad_y,
            clamp(dL_dScreenPos.y, -clampVal, clampVal), memory_order_relaxed);
        
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].scale_x,
            clamp(dL_dLogScale.x, -clampVal, clampVal), memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].scale_y,
            clamp(dL_dLogScale.y, -clampVal, clampVal), memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].scale_z,
            clamp(dL_dLogScale.z, -clampVal, clampVal), memory_order_relaxed);
        
        atomic_fetch_add_explicit(((device atomic_float*)&gradients[gIdx].rotation) + 0,
            clamp(dL_dq.x, -clampVal, clampVal), memory_order_relaxed);
        atomic_fetch_add_explicit(((device atomic_float*)&gradients[gIdx].rotation) + 1,
            clamp(dL_dq.y, -clampVal, clampVal), memory_order_relaxed);
        atomic_fetch_add_explicit(((device atomic_float*)&gradients[gIdx].rotation) + 2,
            clamp(dL_dq.z, -clampVal, clampVal), memory_order_relaxed);
        atomic_fetch_add_explicit(((device atomic_float*)&gradients[gIdx].rotation) + 3,
            clamp(dL_dq.w, -clampVal, clampVal), memory_order_relaxed);
    }
}

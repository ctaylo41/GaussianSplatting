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
struct ProjectedGaussian {
    float2 screenPos;
    float3 conic;       // Inverse 2D covariance
    float depth;
    float opacity;      // AFTER sigmoid (for rendering efficiency)
    float3 color;
    float radius;
    uint tileMinX;
    uint tileMinY;
    uint tileMaxX;
    uint tileMaxY;
    float2 viewPos_xy;
};

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
    float3 cameraPos;
    float _pad1;
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
};

constant float SH_C0 = 0.28209479177387814f;
constant uint TILE_SIZE = 16;
constant float MAX_RADIUS = 64.0f;
constant float MAX_SCALE = 2.0f;  // Log scale range -2 to 2 (exp: 0.14 to 7.4, max 55:1 aspect)

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
    
    // Behind camera or too close
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
    
    // Screen position (Y flipped for screen coordinates)
    proj.screenPos = float2(
        (ndc.x * 0.5 + 0.5) * uniforms.screenSize.x,
        (1.0 - (ndc.y * 0.5 + 0.5)) * uniforms.screenSize.y
    );
    proj.depth = viewPos.z;
    proj.viewPos_xy = viewPos.xy;
    
    // ===== SCALE: Apply exp() to log scale =====
    float3 logScale = clamp(g.scale, -MAX_SCALE, MAX_SCALE);
    float3 scale = exp(logScale);
    
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
    
    // View space projection using same approach as shaders.metal
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
    float3x3 cov2D = T * Sigma3D * transpose(T);
    
    // Metal matrix indexing: matrix[col][row]
    // For symmetric matrix: [0][0]=top-left, [1][1]=bottom-right, [0][1]=[1][0]=off-diagonal
    float a = cov2D[0][0];  // Column 0, Row 0 = (0,0)
    float b = cov2D[1][0];  // Column 1, Row 0 = (0,1) - the off-diagonal term
    float c = cov2D[1][1];  // Column 1, Row 1 = (1,1)
    
    // Low-pass filter
    a += 0.3;
    c += 0.3;
    
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
    proj.radius = min(ceil(rawRadius), MAX_RADIUS);
    
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
    
    // Limit tile coverage
    uint tilesX = proj.tileMaxX - proj.tileMinX + 1;
    uint tilesY = proj.tileMaxY - proj.tileMinY + 1;
    if (tilesX * tilesY > 64) {
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
    
    uint maxIter = min(range.count, 256u);
    
    for (uint i = 0; i < maxIter && T > 0.0001; i++) {
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
    uint endIdx = min(lastIdx + 1, range.start + min(range.count, 256u));
    
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
    float last_alpha = 0.0;
    float3 accum_rec = float3(0.0);  // Accumulated color from Gaussians BEHIND current
    float3 last_color = float3(0.0);
    
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
        T = T / (1.0 - alpha);
        
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
        // dL/dalpha = T * dot(dL_dPixel, color - accum_rec)
        accum_rec = last_alpha * last_color + (1.0 - last_alpha) * accum_rec;
        last_color = p.color;
        
        float dL_dAlpha = T * dot(dL_dPixel, p.color - accum_rec);
        
        last_alpha = alpha;
        
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
        float2 conic_d = float2(
            p.conic.x * d.x + p.conic.y * d.y,
            p.conic.y * d.x + p.conic.z * d.y
        );
        float2 dL_dScreenPos = dL_dG * G * conic_d;
        
        // ===== World position gradient =====
        float z = p.depth;
        float fx = uniforms.focalLength.x;
        float fy = uniforms.focalLength.y;
        
        float3 dL_dViewPos;
        dL_dViewPos.x = dL_dScreenPos.x * fx / z;
        dL_dViewPos.y = -dL_dScreenPos.y * fy / z;  // Y flip
        dL_dViewPos.z = -dL_dScreenPos.x * fx * p.viewPos_xy.x / (z * z)
                        + dL_dScreenPos.y * fy * p.viewPos_xy.y / (z * z);
        
        // Extract view rotation (columns = rotation matrix)
        // viewMatrix is world-to-view, so transpose gives view-to-world
        float3x3 viewRot = float3x3(
            uniforms.viewMatrix[0].xyz,
            uniforms.viewMatrix[1].xyz,
            uniforms.viewMatrix[2].xyz
        );
        // viewRot already contains columns of rotation, transpose gives view-to-world
        float3 dL_dWorldPos = transpose(viewRot) * dL_dViewPos;
        
        // ===== Proper Scale and Rotation Gradients =====
        // 1. Conic gradient
        float3 dL_dConic;
        dL_dConic.x = dL_dG * G * (-0.5 * d.x * d.x);
        dL_dConic.y = dL_dG * G * (-1.0 * d.x * d.y);
        dL_dConic.z = dL_dG * G * (-0.5 * d.y * d.y);
        
        // 2. Cov2D gradient: dL_dCov2D = -Conic * dL_dConic * Conic
        float3 c = p.conic;
        float t00 = dL_dConic.x * c.x + dL_dConic.y * c.y;
        float t01 = dL_dConic.x * c.y + dL_dConic.y * c.z;
        float t10 = dL_dConic.y * c.x + dL_dConic.z * c.y;
        float t11 = dL_dConic.y * c.y + dL_dConic.z * c.z;
        
        float3 dL_dCov2D;
        dL_dCov2D.x = -(c.x * t00 + c.y * t10);
        dL_dCov2D.y = -(c.x * t01 + c.y * t11);
        dL_dCov2D.z = -(c.y * t01 + c.z * t11);
        
        // 3. Cov3D gradient
        float3 t_cam = float3(p.viewPos_xy, p.depth);
        float txtz = t_cam.x / t_cam.z;
        float tytz = t_cam.y / t_cam.z;
        
        float J00 = fx / t_cam.z;
        float J02 = -fx * txtz / t_cam.z;
        float J11 = fy / t_cam.z;
        float J12 = -fy * tytz / t_cam.z;
        
        float3 T0 = float3(J00, 0, J02) * viewRot;
        float3 T1 = float3(0, J11, J12) * viewRot;
        
        float3 A = dL_dCov2D.x * T0 + dL_dCov2D.y * T1;
        float3 B = dL_dCov2D.y * T0 + dL_dCov2D.z * T1;
        
        float3x3 dL_dCov3D;
        dL_dCov3D[0] = A * T0.x + B * T1.x;
        dL_dCov3D[1] = A * T0.y + B * T1.y;
        dL_dCov3D[2] = A * T0.z + B * T1.z;
        
        // 4. Scale and Rotation gradients
        Gaussian g_orig = gaussians[gIdx];
        float3 scale = exp(clamp(g_orig.scale, -MAX_SCALE, MAX_SCALE));
        float3x3 R = quatToMat(g_orig.rotation);
        // Diagonal scale matrix - Metal column constructor
        float3x3 S = float3x3(
            float3(scale.x, 0, 0),  // Column 0
            float3(0, scale.y, 0),  // Column 1
            float3(0, 0, scale.z)   // Column 2
        );
        float3x3 M = R * S;
        
        float3x3 dL_dM = 2.0 * dL_dCov3D * M;
        float3x3 Rt = transpose(R);
        float3 dL_dScale_val;
        dL_dScale_val.x = dot(Rt[0], dL_dM[0]);
        dL_dScale_val.y = dot(Rt[1], dL_dM[1]);
        dL_dScale_val.z = dot(Rt[2], dL_dM[2]);
        
        float3 dL_dLogScale = dL_dScale_val * scale;
        
        // Add scale regularization to prevent extreme aspect ratios
        // Penalize scales that are too different from the average
        // This prevents extreme aspect ratios (rectangles instead of ellipsoids)
        float avgScale = (g_orig.scale.x + g_orig.scale.y + g_orig.scale.z) / 3.0;
        float scaleRegWeight = 0.1;  // Increased regularization strength
        float3 scaleRegGrad = scaleRegWeight * (g_orig.scale - avgScale);
        dL_dLogScale += scaleRegGrad;
        
        float3x3 dL_dR = dL_dM * S;
        
        float4 dL_dq = float4(0);
        float w = g_orig.rotation.x;
        float x = g_orig.rotation.y;
        float y = g_orig.rotation.z;
        float z_rot = g_orig.rotation.w; // Rename to avoid conflict with z coordinate
        
        dL_dq.x = dot(dL_dR[0], float3(0, 2*z_rot, -2*y)) + dot(dL_dR[1], float3(-2*z_rot, 0, 2*x)) + dot(dL_dR[2], float3(2*y, -2*x, 0));
        dL_dq.y = dot(dL_dR[0], float3(0, 2*y, 2*z_rot)) + dot(dL_dR[1], float3(2*y, -4*x, -2*w)) + dot(dL_dR[2], float3(2*z_rot, 2*w, -4*x));
        dL_dq.z = dot(dL_dR[0], float3(-4*y, 2*x, -2*w)) + dot(dL_dR[1], float3(2*x, 0, 2*z_rot)) + dot(dL_dR[2], float3(2*w, 2*z_rot, -4*y));
        dL_dq.w = dot(dL_dR[0], float3(-4*z_rot, 2*w, 2*x)) + dot(dL_dR[1], float3(-2*w, -4*z_rot, 2*y)) + dot(dL_dR[2], float3(2*x, 2*y, 0));
        
        float clampVal = 1.0;
        
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
        
        // Note: T update is already handled at the start of the loop via T = T / (1 - alpha)
    }
}

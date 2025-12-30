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
constant float MAX_SCALE = 3.0f;

// Quaternion to rotation matrix
// q.x=w, q.y=x, q.z=y, q.w=z
float3x3 quatToMat(float4 q) {
    float w = q.x, x = q.y, y = q.z, z = q.w;
    return float3x3(
        1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z), 2.0*(x*z + w*y),
        2.0*(x*y + w*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x),
        2.0*(x*z - w*y), 2.0*(y*z + w*x), 1.0 - 2.0*(x*x + y*y)
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
    float3x3 S = float3x3(scale.x, 0, 0, 0, scale.y, 0, 0, 0, scale.z);
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
    
    // Jacobian of projection (3x3 with zero last row for proper matrix multiply)
    float J00 = fx / z_cam;
    float J02 = -fx * txtz / z_cam;
    float J11 = fy / z_cam;
    float J12 = -fy * tytz / z_cam;
    
    float3x3 J = float3x3(J00, 0, J02,
                          0, J11, J12,
                          0, 0, 0);
    
    // View rotation (upper-left 3x3) - use row access like shaders.metal
    // In Metal, matrix[i] gives row i
    float3x3 W = float3x3(uniforms.viewMatrix[0].xyz, 
                          uniforms.viewMatrix[1].xyz, 
                          uniforms.viewMatrix[2].xyz);
    
    // Combined transform: T = J * W
    float3x3 T = J * W;
    
    // Project 3D covariance to 2D: cov2D = T * Sigma3D * T^T
    float3x3 cov2D = T * Sigma3D * transpose(T);
    
    float a = cov2D[0][0];
    float b = cov2D[0][1];
    float c = cov2D[1][1];
    
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
    
    float T = 1.0;
    
    uint endIdx = min(lastIdx + 1, range.start + min(range.count, 256u));
    
    for (uint sortIdx = range.start; sortIdx < endIdx && T > 0.0001; sortIdx++) {
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
        
        float T_i = T;
        float weight = alpha * T_i;
        
        // ===== Color/SH gradients =====
        float3 dL_dColor = dL_dPixel * weight;
        
        // ===== Alpha gradient =====
        float dL_dAlpha = dot(dL_dPixel, p.color * T_i);
        
        // ===== Opacity gradient =====
        // alpha = sigmoid(raw_opacity) * G
        // dalpha/d_raw_opacity = sigmoid * (1-sigmoid) * G
        float sig = p.opacity;
        float dAlpha_dRawOp = sig * (1.0 - sig) * G;
        float dL_dRawOpacity = dL_dAlpha * dAlpha_dRawOp;
        
        // ===== Gaussian gradient =====
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
        
        // Use row access like forward pass (matrix[i] gives row i in Metal)
        float3x3 viewRot = float3x3(
            uniforms.viewMatrix[0].xyz,
            uniforms.viewMatrix[1].xyz,
            uniforms.viewMatrix[2].xyz
        );
        float3 dL_dWorldPos = transpose(viewRot) * dL_dViewPos;
        
        // Scale gradient (simplified heuristic)
        float scale_grad_magnitude = dL_dG * G * 0.01;
        
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
            clamp(scale_grad_magnitude, -clampVal, clampVal), memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].scale_y,
            clamp(scale_grad_magnitude, -clampVal, clampVal), memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].scale_z,
            clamp(scale_grad_magnitude, -clampVal, clampVal), memory_order_relaxed);
        
        T *= (1.0 - alpha);
    }
}

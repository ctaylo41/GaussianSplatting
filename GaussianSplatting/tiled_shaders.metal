//
//  tiled_shaders.metal
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-28.
//

#include <metal_stdlib>
using namespace metal;

struct Gaussian {
    packed_float3 position;
    float _pad0;
    packed_float3 scale;
    float _pad1;
    float4 rotation;
    float opacity;
    float sh[12];
    float _pad2;
    float _pad3;
    float _pad4;
};

// Projected Gaussian data for tiled rendering
struct ProjectedGaussian {
    float2 screenPos;
    packed_float3 conic;
    float depth;
    float opacity;
    packed_float3 color;
    float radius;
    uint tileMinX;
    uint tileMinY;
    uint tileMaxX;
    uint tileMaxY;
    float _pad1;
    float2 viewPos_xy;
    packed_float3 cov2D;
    packed_float3 viewDir;
};  

// Tile range structure
struct TileRange {
    uint start;
    uint count;
};

// Uniforms for tiled rendering
struct TiledUniforms {
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4x4 viewProjectionMatrix;
    float2 screenSize;
    float2 focalLength;
    float3 cameraPos;        
    uint numTilesX;
    uint numTilesY;
    uint numGaussians;
    uint _pad2;
};

// Gradients for Gaussians
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
    float viewspace_grad_x;
    float viewspace_grad_y;
    float _pad2;
    float _pad3;
};

// Constants
constant float SH_C0 = 0.28209479177387814f;
constant float SH_C1 = 0.4886025119029199f;
constant uint TILE_SIZE = 16;
constant float MAX_RADIUS = 512.0f;
// exp(5), 148 reasonable max scale
constant float MAX_SCALE = 5.0f;  

// Quaternion to rotation matrix
// q.x=w, q.y=x, q.z=y, q.w=z
float3x3 quatToMat(float4 q) {
    float w = q.x, x = q.y, y = q.z, z = q.w;
    // Metal float3x3 constructor takes columns
    return float3x3(
        float3(1.0 - 2.0*(y*y + z*z), 2.0*(x*y + w*z), 2.0*(x*z - w*y)),
        float3(2.0*(x*y - w*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z + w*x)),
        float3(2.0*(x*z + w*y), 2.0*(y*z - w*x), 1.0 - 2.0*(x*x + y*y))
    );
}

// Project Gaussians to screen space and compute projected parameters
kernel void projectGaussians(
    device const Gaussian* gaussians [[buffer(0)]],
    device ProjectedGaussian* projected [[buffer(1)]],
    constant TiledUniforms& uniforms [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= uniforms.numGaussians) return;
    
    // Fetch Gaussian
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

    // Transform to clip space
    float4 worldPos = float4(g.position, 1.0);
    float4 viewPos = uniforms.viewMatrix * worldPos;
    float4 clipPos = uniforms.viewProjectionMatrix * worldPos;
    
    // COLMAP uses opencv convention camera looks down +z axis
    // Objects in front of camera have positive view z
    if (clipPos.w <= 0.1 || viewPos.z <= 0.1) {
        projected[tid] = proj;
        return;
    }
    
    // Normalized Device Coordinates
    float3 ndc = clipPos.xyz / clipPos.w;
    
    // Outside frustum
    if (abs(ndc.x) > 1.2 || abs(ndc.y) > 1.2) {
        projected[tid] = proj;
        return;
    }
    
    // Screen position from NDC
    proj.screenPos = float2(
        (ndc.x * 0.5 + 0.5) * uniforms.screenSize.x,
        (ndc.y * 0.5 + 0.5) * uniforms.screenSize.y 
    );

    // Store positive depth
    proj.depth = viewPos.z;  
    proj.viewPos_xy = viewPos.xy;
    
    // scale is stored in LOG space
    float3 logScale = clamp(g.scale, -MAX_SCALE, MAX_SCALE);
    float3 scale = exp(logScale);
    
    // Prevent extremely elongated Gaussians max 20:1 aspect ratio
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
    
    // Build 3D covariance using official 3DGS convention:
    // M = S * R, Sigma = M^T * M = R^T * S^2 * R
    float3x3 R = quatToMat(q);
    // Diagonal scale matrix
    float3x3 S = float3x3(
        float3(scale.x, 0, 0),
        float3(0, scale.y, 0),
        float3(0, 0, scale.z)
    );
    // Viewer/paper convention: M = R * S
    float3x3 M = R * S;
    // Sigma = M * M^T
    float3x3 Sigma3D = M * transpose(M);
    
    // View space projection using same approach as official 3DGS
    float z_cam = viewPos.z;
    float fx = uniforms.focalLength.x;
    float fy = uniforms.focalLength.y;
    
    // Clamp to avoid numerical issues at edges
    float limx = 1.3 * fx / z_cam;
    float limy = 1.3 * fy / z_cam;
    float txtz = clamp(viewPos.x / z_cam, -limx, limx);
    float tytz = clamp(viewPos.y / z_cam, -limy, limy);
    
    // Jacobian of projection perspective projection derivative
    // Maps 3D view space to 2D screen space
    float J00 = fx / z_cam;
    float J02 = -fx * txtz / z_cam;
    float J11 = fy / z_cam;
    float J12 = -fy * tytz / z_cam;
    
    // Jacobian matrix
    float3x3 J = float3x3(
        float3(J00, 0, 0),     
        float3(0, J11, 0),  
        float3(J02, J12, 0)
    );
    
    // View matrix rotation world-to-view extract 3x3 rotation
    float3x3 W = float3x3(uniforms.viewMatrix[0].xyz,
                          uniforms.viewMatrix[1].xyz,
                          uniforms.viewMatrix[2].xyz);
    
    // Combined transform T = J * W
    float3x3 T = J * W;
    // Project 3D covariance to 2D cov2D = T * Sigma3D * T^T
    float3x3 cov2D_mat = T * Sigma3D * transpose(T);
    
    // Extract 2D covariance components
    float a = cov2D_mat[0][0];  
    float b = cov2D_mat[1][0];
    float c = cov2D_mat[1][1];
    
    // Low-pass filter add before storing for backward pass
    a += 0.3;
    c += 0.3;
    
    // Store cov2D for backward pass after low-pass filter
    proj.cov2D = float3(a, b, c);
    
    // Compute determinant
    float det = a * c - b * b;
    if (det < 0.0001) {
        projected[tid] = proj;
        return;
    }
    
    // Conic
    float inv_det = 1.0 / det;
    proj.conic = float3(c * inv_det, -b * inv_det, a * inv_det);
    
    // Compute radius from eigenvalues
    float mid = 0.5 * (a + c);
    float disc = mid * mid - det;
    float l1 = mid + sqrt(max(0.1f, disc));
    float rawRadius = 3.0 * sqrt(l1);
    proj.radius = min(ceil(rawRadius), MAX_RADIUS);
    
    // Projected radius zero means skip
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
    
    // No tile coverage
    if (minX > maxX || minY > maxY) {
        proj.radius = 0;
        projected[tid] = proj;
        return;
    }
    
    // Tile bounds
    proj.tileMinX = uint(minX) / TILE_SIZE;
    proj.tileMinY = uint(minY) / TILE_SIZE;
    proj.tileMaxX = min(uint(maxX) / TILE_SIZE, uniforms.numTilesX - 1);
    proj.tileMaxY = min(uint(maxY) / TILE_SIZE, uniforms.numTilesY - 1);
    
    // Limit tile coverage increased from 64 to allow larger Gaussians
    uint tilesX = proj.tileMaxX - proj.tileMinX + 1;
    uint tilesY = proj.tileMaxY - proj.tileMinY + 1;
    if (tilesX * tilesY > 256) {
        proj.radius = 0;
        projected[tid] = proj;
        return;
    }
    
    // Apply sigmoid to opacity
    float rawOpacity = clamp(g.opacity, -8.0f, 8.0f);
    proj.opacity = 1.0 / (1.0 + exp(-rawOpacity));

    // Color from DC terms using sigmoid activation (like nerfstudio splatfacto)
    // Sigmoid naturally bounds colors to (0, 1) no clamping needed
    // This prevents RGB channel divergence that causes saturated color artifacts
    float3 rawColor = float3(g.sh[0], g.sh[4], g.sh[8]);
    proj.color = float3(
        1.0f / (1.0f + exp(-rawColor.x)),
        1.0f / (1.0f + exp(-rawColor.y)),
        1.0f / (1.0f + exp(-rawColor.z))
    );

    projected[tid] = proj;
}

// Tiled forward rendering kernel
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

    // Determine tile index   
    uint tileX = gid.x / TILE_SIZE;
    uint tileY = gid.y / TILE_SIZE;
    uint tileIdx = tileY * uniforms.numTilesX + tileX;
    TileRange range = tileRanges[tileIdx];
    
    // Use float precision to match backward pass critical for gradient accuracy
    float3 color = float3(0);
    float T = 1.0f;
    float2 pixelPos = float2(gid) + 0.5;
    
    uint lastIdx = 0;
    bool hasContrib = false;
    
    // Rely purely on T termination instead of artificial cap
    for (uint i = 0; i < range.count && T > 0.0001f; i++) {
        uint sortIdx = range.start + i;
        uint gIdx = sortedIndices[sortIdx];
        
        if (gIdx >= uniforms.numGaussians) continue;
        
        // Fetch projected Gaussian
        ProjectedGaussian p = projected[gIdx];
        
        // Skip invalid Gaussians
        if (p.radius <= 0) continue;
        
        // Compute offset from Gaussian center
        float2 d = pixelPos - p.screenPos;
        
        // Check if conic is valid
        float conicMag = abs(p.conic.x) + abs(p.conic.y) + abs(p.conic.z);
        if (conicMag < 0.0001) continue;
        
        // Gaussian evaluation in float precision to match backward pass
        float power = -0.5f * (p.conic.x * d.x * d.x +
                               2.0f * p.conic.y * d.x * d.y +
                               p.conic.z * d.y * d.y);
        
        // Early skip for negligible contribution
        if (power > 0.0f || power < -4.5f) continue;
        
        // Compute Gaussian weight and alpha
        float G = exp(power);
        float alpha = min(p.opacity * G, 0.99f);
        
        // Skip negligible alpha
        if (alpha < 1.0f / 255.0f) continue;
        
        // Accumulate color using alpha blending
        color += float3(p.color) * alpha * T;
        T *= (1.0f - alpha);
        
        lastIdx = sortIdx;
        hasContrib = true;
    }
    
    // Blend with white background using remaining transmittance
    float3 bgColor = float3(1.0f, 1.0f, 1.0f);
    color = color + bgColor * T;
    
    // Store last contributing index for backward pass
    uint pixelIdx = gid.y * uint(uniforms.screenSize.x) + gid.x;
    lastContribIdx[pixelIdx] = hasContrib ? lastIdx : UINT_MAX;
    
    output.write(float4(color, 1.0), gid);
}

// Tile-local gradient accumulation constants
// 128 Gaussians * 16 floats = 2048 floats = 8KB threadgroup memory
constant uint BACKWARD_CHUNK_SIZE = 128;
constant uint NUM_GRAD_COMPONENTS = 16;

// Helper: atomic float add for threadgroup memory using CAS loop
inline void atomicAddTG(threadgroup atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(
        addr, &expected,
        as_type<uint>(as_type<float>(expected) + val),
        memory_order_relaxed, memory_order_relaxed)) {}
}

// SIMD-reduced atomic add: reduces 256 threads to 8 atomics (32 threads per simdgroup)
inline void simdAtomicAddTG(threadgroup atomic_uint* addr, float val) {
    float sum = simd_sum(val);
    if (simd_is_first()) {
        atomicAddTG(addr, sum);
    }
}

// Tiled backward rendering kernel with tile-local gradient accumulation
// Instead of 256 pixels doing 15 device atomics each per Gaussian (3840 atomics),
// we accumulate in threadgroup memory and do 15 device atomics per Gaussian per tile
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
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    // Threadgroup memory for gradient accumulation (as atomic_uint, bits interpreted as float)
    // Layout: [CHUNK_SIZE][16] - 16 gradient components per Gaussian
    // Components: pos(3), opacity(1), scale(3), rot(4), sh_dc(3), viewspace(2)
    threadgroup atomic_uint tgGrads[BACKWARD_CHUNK_SIZE * NUM_GRAD_COMPONENTS];
    threadgroup uint tgGaussianIdx[BACKWARD_CHUNK_SIZE];

    // Bounds check - but we need all threads for reduction
    bool valid = (gid.x < uint(uniforms.screenSize.x) && gid.y < uint(uniforms.screenSize.y));

    // Get tile range (all threads in tile share this)
    uint tileIdx = tgid.y * uniforms.numTilesX + tgid.x;
    TileRange range = tileRanges[tileIdx];

    // Per-pixel state
    uint pixelIdx = valid ? (gid.y * uint(uniforms.screenSize.x) + gid.x) : 0;
    uint lastIdx = valid ? lastContribIdx[pixelIdx] : UINT_MAX;
    bool hasContrib = valid && (lastIdx != UINT_MAX);
    uint endIdx = hasContrib ? min(lastIdx + 1, range.start + range.count) : 0;

    // Pixel position and gradient computation
    float2 pixelPos = float2(gid) + 0.5;
    float4 r_pix = valid ? rendered.read(gid) : float4(0);
    float4 gt_pix = valid ? groundTruth.read(gid) : float4(0);
    float3 diff = r_pix.rgb - gt_pix.rgb;
    float3 dL_dPixel = sign(diff) / 3.0;

    // Pre-compute T_final (same as before)
    float T_final = 1.0;
    if (hasContrib) {
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
    }

    // Initialize backward pass state
    float T = T_final;
    float3 bgColor = float3(1.0);
    float3 accum_rec = bgColor;

    // Cache view rotation (used for all Gaussians)
    float3x3 viewRot = float3x3(
        uniforms.viewMatrix[0].xyz,
        uniforms.viewMatrix[1].xyz,
        uniforms.viewMatrix[2].xyz
    );
    float fx = uniforms.focalLength.x;
    float fy = uniforms.focalLength.y;

    // Process the tile's Gaussians in chunks
    uint totalCount = range.count;
    uint numChunks = (totalCount + BACKWARD_CHUNK_SIZE - 1) / BACKWARD_CHUNK_SIZE;

    // Process chunks back-to-front (high sortIdx to low)
    for (int chunk = int(numChunks) - 1; chunk >= 0; chunk--) {
        uint chunkStart = range.start + uint(chunk) * BACKWARD_CHUNK_SIZE;
        uint chunkEnd = min(chunkStart + BACKWARD_CHUNK_SIZE, range.start + totalCount);
        uint chunkSize = chunkEnd - chunkStart;

        // Clear threadgroup accumulators (all threads participate)
        for (uint i = tid; i < BACKWARD_CHUNK_SIZE * NUM_GRAD_COMPONENTS; i += 256) {
            atomic_store_explicit(&tgGrads[i], 0u, memory_order_relaxed);
        }

        // Cache Gaussian indices for this chunk
        for (uint i = tid; i < chunkSize; i += 256) {
            tgGaussianIdx[i] = sortedIndices[chunkStart + i];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Process chunk back-to-front (each pixel maintains its own T state)
        for (int sortIdx = int(chunkEnd) - 1; sortIdx >= int(chunkStart); sortIdx--) {
            uint localIdx = uint(sortIdx) - chunkStart;
            uint gIdx = tgGaussianIdx[localIdx];

            // Only process if this pixel contributes to this Gaussian
            bool process = hasContrib && (uint(sortIdx) < endIdx) && (gIdx < uniforms.numGaussians);

            if (process) {
                ProjectedGaussian p = projected[gIdx];

                if (p.radius > 0) {
                    float2 d = pixelPos - p.screenPos;
                    float power = -0.5 * (p.conic.x * d.x * d.x +
                                          2.0 * p.conic.y * d.x * d.y +
                                          p.conic.z * d.y * d.y);

                    if (power <= 0.0 && power >= -4.5) {
                        float G = exp(power);
                        float alpha = min(p.opacity * G, 0.99f);

                        if (alpha >= 1.0 / 255.0) {
                            // Update T
                            T = T / max(1.0 - alpha, 0.01);
                            float weight = alpha * T;

                            // Color gradient
                            float3 dL_dColor = dL_dPixel * weight;
                            float dL_dAlpha = T * dot(dL_dPixel, p.color - accum_rec);
                            accum_rec = alpha * p.color + (1.0 - alpha) * accum_rec;

                            // Opacity gradient
                            float sig = p.opacity;
                            float dAlpha_dRawOp = sig * (1.0 - sig) * G;
                            float dL_dRawOpacity = clamp(dL_dAlpha * dAlpha_dRawOp, -0.1f, 0.1f);

                            // Screen position gradient
                            float dL_dG = dL_dAlpha * sig;
                            float gdx = G * d.x;
                            float gdy = G * d.y;
                            float dG_ddelx = -gdx * p.conic.x - gdy * p.conic.y;
                            float dG_ddely = -gdy * p.conic.z - gdx * p.conic.y;
                            float2 dL_dScreenPos = dL_dG * float2(-dG_ddelx, -dG_ddely);

                            // World position gradient
                            float z = p.depth;
                            float txtz = p.viewPos_xy.x / z;
                            float tytz = p.viewPos_xy.y / z;

                            float3 dL_dViewPos;
                            dL_dViewPos.x = dL_dScreenPos.x * fx / z;
                            dL_dViewPos.y = dL_dScreenPos.y * fy / z;
                            dL_dViewPos.z = -dL_dScreenPos.x * fx * txtz / z
                                            -dL_dScreenPos.y * fy * tytz / z;

                            float3 dL_dWorldPos = transpose(viewRot) * dL_dViewPos;

                            // Conic gradient
                            float3 dL_dConic;
                            dL_dConic.x = -0.5f * dL_dG * G * d.x * d.x;
                            dL_dConic.y = -0.5f * dL_dG * G * 2.0f * d.x * d.y;
                            dL_dConic.z = -0.5f * dL_dG * G * d.y * d.y;

                            // Cov2D gradient
                            float cov_a = p.cov2D.x;
                            float cov_b = p.cov2D.y;
                            float cov_c = p.cov2D.z;
                            float denom = cov_a * cov_c - cov_b * cov_b;
                            float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

                            float3 dL_dCov2D;
                            dL_dCov2D.x = denom2inv * (-cov_c * cov_c * dL_dConic.x
                                                       + 2.0f * cov_b * cov_c * dL_dConic.y
                                                       + (denom - cov_a * cov_c) * dL_dConic.z);
                            dL_dCov2D.z = denom2inv * (-cov_a * cov_a * dL_dConic.z
                                                       + 2.0f * cov_a * cov_b * dL_dConic.y
                                                       + (denom - cov_a * cov_c) * dL_dConic.x);
                            dL_dCov2D.y = denom2inv * 2.0f * (cov_b * cov_c * dL_dConic.x
                                                              - (denom + 2.0f * cov_b * cov_b) * dL_dConic.y
                                                              + cov_a * cov_b * dL_dConic.z);

                            // Cov3D gradient
                            float3 t_cam = float3(p.viewPos_xy, p.depth);
                            float J00 = fx / t_cam.z;
                            float J02 = -fx * (t_cam.x / t_cam.z) / t_cam.z;
                            float J11 = fy / t_cam.z;
                            float J12 = -fy * (t_cam.y / t_cam.z) / t_cam.z;

                            float3x3 J = float3x3(
                                float3(J00, 0, 0),
                                float3(0, J11, 0),
                                float3(J02, J12, 0)
                            );

                            float3x3 T_mat = J * viewRot;
                            float3x3 dL_dCov2D_mat = float3x3(
                                float3(dL_dCov2D.x, dL_dCov2D.y, 0),
                                float3(dL_dCov2D.y, dL_dCov2D.z, 0),
                                float3(0, 0, 0)
                            );
                            float3x3 dL_dCov3D = transpose(T_mat) * dL_dCov2D_mat * T_mat;

                            // Scale and Rotation gradients
                            Gaussian g_orig = gaussians[gIdx];
                            float3 scale = exp(clamp(g_orig.scale, -MAX_SCALE, MAX_SCALE));

                            float4 q = g_orig.rotation;
                            float r = q.x;
                            float x_q = q.y;
                            float y_q = q.z;
                            float z_q = q.w;

                            float3x3 R = quatToMat(q);
                            float3x3 S = float3x3(
                                float3(scale.x, 0, 0),
                                float3(0, scale.y, 0),
                                float3(0, 0, scale.z)
                            );
                            float3x3 M = R * S;
                            float3x3 dL_dM = 2.0f * dL_dCov3D * M;

                            float3x3 Rt = transpose(R);
                            float3x3 Rt_dLdM = Rt * dL_dM;
                            float3 dL_dScale_val = float3(Rt_dLdM[0][0], Rt_dLdM[1][1], Rt_dLdM[2][2]);
                            float3 dL_dLogScale = dL_dScale_val * scale;

                            float3x3 dL_dR = float3x3(
                                dL_dM[0] * scale.x,
                                dL_dM[1] * scale.y,
                                dL_dM[2] * scale.z
                            );
                            float3x3 dL_dMt_scaled = transpose(dL_dR);

                            // Quaternion gradient
                            float4 dL_dq;
                            dL_dq.x = 2.0f * (z_q * (dL_dMt_scaled[0][1] - dL_dMt_scaled[1][0]) +
                                             y_q * (dL_dMt_scaled[2][0] - dL_dMt_scaled[0][2]) +
                                             x_q * (dL_dMt_scaled[1][2] - dL_dMt_scaled[2][1]));
                            dL_dq.y = 2.0f * (y_q * (dL_dMt_scaled[1][0] + dL_dMt_scaled[0][1]) +
                                             z_q * (dL_dMt_scaled[2][0] + dL_dMt_scaled[0][2]) +
                                             r * (dL_dMt_scaled[1][2] - dL_dMt_scaled[2][1]) -
                                             2.0f * x_q * (dL_dMt_scaled[2][2] + dL_dMt_scaled[1][1]));
                            dL_dq.z = 2.0f * (x_q * (dL_dMt_scaled[1][0] + dL_dMt_scaled[0][1]) +
                                             r * (dL_dMt_scaled[2][0] - dL_dMt_scaled[0][2]) +
                                             z_q * (dL_dMt_scaled[1][2] + dL_dMt_scaled[2][1]) -
                                             2.0f * y_q * (dL_dMt_scaled[2][2] + dL_dMt_scaled[0][0]));
                            dL_dq.w = 2.0f * (r * (dL_dMt_scaled[0][1] - dL_dMt_scaled[1][0]) +
                                             x_q * (dL_dMt_scaled[2][0] + dL_dMt_scaled[0][2]) +
                                             y_q * (dL_dMt_scaled[1][2] + dL_dMt_scaled[2][1]) -
                                             2.0f * z_q * (dL_dMt_scaled[1][1] + dL_dMt_scaled[0][0]));

                            // SH gradient (DC terms only)
                            float3 color = float3(p.color);
                            float3 sigmoid_grad = color * (1.0f - color);
                            float3 sh_grad = clamp(dL_dColor * sigmoid_grad, -1.0f, 1.0f);

                            // Accumulate to threadgroup memory using SIMD-reduced atomics
                            // SIMD reduction first: 256 threads -> 8 simd groups -> 8 atomics
                            // This reduces CAS contention by ~32x
                            uint baseIdx = localIdx * NUM_GRAD_COMPONENTS;
                            simdAtomicAddTG(&tgGrads[baseIdx + 0], dL_dWorldPos.x);
                            simdAtomicAddTG(&tgGrads[baseIdx + 1], dL_dWorldPos.y);
                            simdAtomicAddTG(&tgGrads[baseIdx + 2], dL_dWorldPos.z);
                            simdAtomicAddTG(&tgGrads[baseIdx + 3], dL_dRawOpacity);
                            simdAtomicAddTG(&tgGrads[baseIdx + 4], dL_dLogScale.x);
                            simdAtomicAddTG(&tgGrads[baseIdx + 5], dL_dLogScale.y);
                            simdAtomicAddTG(&tgGrads[baseIdx + 6], dL_dLogScale.z);
                            simdAtomicAddTG(&tgGrads[baseIdx + 7], dL_dq.x);
                            simdAtomicAddTG(&tgGrads[baseIdx + 8], dL_dq.y);
                            simdAtomicAddTG(&tgGrads[baseIdx + 9], dL_dq.z);
                            simdAtomicAddTG(&tgGrads[baseIdx + 10], dL_dq.w);
                            simdAtomicAddTG(&tgGrads[baseIdx + 11], sh_grad.r);
                            simdAtomicAddTG(&tgGrads[baseIdx + 12], sh_grad.g);
                            simdAtomicAddTG(&tgGrads[baseIdx + 13], sh_grad.b);
                            simdAtomicAddTG(&tgGrads[baseIdx + 14], dL_dScreenPos.x);
                            simdAtomicAddTG(&tgGrads[baseIdx + 15], dL_dScreenPos.y);
                        }
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Write accumulated gradients to global memory
        // Each thread handles some of the Gaussians in the chunk
        for (uint i = tid; i < chunkSize; i += 256) {
            uint gIdx = tgGaussianIdx[i];
            uint baseIdx = i * NUM_GRAD_COMPONENTS;

            // Load from threadgroup (uint bits -> float) and write non-zero to global
            float val;

            val = as_type<float>(atomic_load_explicit(&tgGrads[baseIdx + 0], memory_order_relaxed));
            if (val != 0) atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].position_x, val, memory_order_relaxed);
            val = as_type<float>(atomic_load_explicit(&tgGrads[baseIdx + 1], memory_order_relaxed));
            if (val != 0) atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].position_y, val, memory_order_relaxed);
            val = as_type<float>(atomic_load_explicit(&tgGrads[baseIdx + 2], memory_order_relaxed));
            if (val != 0) atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].position_z, val, memory_order_relaxed);
            val = as_type<float>(atomic_load_explicit(&tgGrads[baseIdx + 3], memory_order_relaxed));
            if (val != 0) atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].opacity, val, memory_order_relaxed);
            val = as_type<float>(atomic_load_explicit(&tgGrads[baseIdx + 4], memory_order_relaxed));
            if (val != 0) atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].scale_x, val, memory_order_relaxed);
            val = as_type<float>(atomic_load_explicit(&tgGrads[baseIdx + 5], memory_order_relaxed));
            if (val != 0) atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].scale_y, val, memory_order_relaxed);
            val = as_type<float>(atomic_load_explicit(&tgGrads[baseIdx + 6], memory_order_relaxed));
            if (val != 0) atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].scale_z, val, memory_order_relaxed);
            val = as_type<float>(atomic_load_explicit(&tgGrads[baseIdx + 7], memory_order_relaxed));
            if (val != 0) atomic_fetch_add_explicit(((device atomic_float*)&gradients[gIdx].rotation) + 0, val, memory_order_relaxed);
            val = as_type<float>(atomic_load_explicit(&tgGrads[baseIdx + 8], memory_order_relaxed));
            if (val != 0) atomic_fetch_add_explicit(((device atomic_float*)&gradients[gIdx].rotation) + 1, val, memory_order_relaxed);
            val = as_type<float>(atomic_load_explicit(&tgGrads[baseIdx + 9], memory_order_relaxed));
            if (val != 0) atomic_fetch_add_explicit(((device atomic_float*)&gradients[gIdx].rotation) + 2, val, memory_order_relaxed);
            val = as_type<float>(atomic_load_explicit(&tgGrads[baseIdx + 10], memory_order_relaxed));
            if (val != 0) atomic_fetch_add_explicit(((device atomic_float*)&gradients[gIdx].rotation) + 3, val, memory_order_relaxed);
            val = as_type<float>(atomic_load_explicit(&tgGrads[baseIdx + 11], memory_order_relaxed));
            if (val != 0) atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].sh[0], val, memory_order_relaxed);
            val = as_type<float>(atomic_load_explicit(&tgGrads[baseIdx + 12], memory_order_relaxed));
            if (val != 0) atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].sh[4], val, memory_order_relaxed);
            val = as_type<float>(atomic_load_explicit(&tgGrads[baseIdx + 13], memory_order_relaxed));
            if (val != 0) atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].sh[8], val, memory_order_relaxed);
            val = as_type<float>(atomic_load_explicit(&tgGrads[baseIdx + 14], memory_order_relaxed));
            if (val != 0) atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].viewspace_grad_x, val, memory_order_relaxed);
            val = as_type<float>(atomic_load_explicit(&tgGrads[baseIdx + 15], memory_order_relaxed));
            if (val != 0) atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].viewspace_grad_y, val, memory_order_relaxed);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// GPU Pair Generation
// Each thread handles one Gaussian and writes all its tile-pairs atomically
constant float GPU_MIN_OPACITY = 0.005f;
constant uint GPU_MAX_TILES_PER_GAUSSIAN = 256u;

kernel void generateTilePairs(
    device const ProjectedGaussian* projected [[buffer(0)]],
    device ulong* pairKeys [[buffer(1)]],
    device uint* pairValues [[buffer(2)]],
    device atomic_uint* writeCounter [[buffer(3)]],
    constant uint& numGaussians [[buffer(4)]],
    constant uint& numTilesX [[buffer(5)]],
    constant uint& maxPairs [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= numGaussians) return;
    
    ProjectedGaussian p = projected[tid];
    
    // Skip invalid Gaussians
    if (p.radius <= 0) return;
    if (p.tileMinX > p.tileMaxX || p.tileMinY > p.tileMaxY) return;
    if (p.opacity < GPU_MIN_OPACITY) return;
    if (p.tileMinX > 10000 || p.tileMaxX > 10000 || p.tileMinY > 10000 || p.tileMaxY > 10000) return;
    
    // Compute number of tiles covered
    uint tilesX = p.tileMaxX - p.tileMinX + 1;
    uint tilesY = p.tileMaxY - p.tileMinY + 1;
    uint tileCount = tilesX * tilesY;
    
    if (tileCount > GPU_MAX_TILES_PER_GAUSSIAN) return;
    
    // Create depth key for sorting (IEEE float to sortable uint)
    uint depthKey = as_type<uint>(p.depth);
    depthKey = (depthKey & 0x80000000u) ? ~depthKey : (depthKey | 0x80000000u);
    
    // Reserve write positions atomically
    uint writePos = atomic_fetch_add_explicit(writeCounter, tileCount, memory_order_relaxed);
    
    // Check buffer bounds
    if (writePos + tileCount > maxPairs) return;
    
    // Write pairs for all tiles this Gaussian touches
    uint idx = 0;
    for (uint ty = p.tileMinY; ty <= p.tileMaxY; ty++) {
        for (uint tx = p.tileMinX; tx <= p.tileMaxX; tx++) {
            uint tileIdx = ty * numTilesX + tx;
            ulong key = (ulong(tileIdx) << 32) | ulong(depthKey);
            
            pairKeys[writePos + idx] = key;
            pairValues[writePos + idx] = tid;
            idx++;
        }
    }
}

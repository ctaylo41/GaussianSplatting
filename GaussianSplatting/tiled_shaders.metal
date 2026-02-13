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
    packed_uchar3 colorClamped;
    uint8_t _pad1;
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

// Intermediate per-Gaussian render gradients accumulated by tiledBackward, read by preprocessBackward
// Atomic version for multi-tile accumulation via tiledBackward
struct RenderGradientsAtomic {
    atomic_float dL_dColor[3];      // Color gradient (accumulated weight * dL_dpixel)
    atomic_float dL_dConic[3];      // Conic gradient
    atomic_float dL_dOpacity;       // Opacity intermediate (accumulated G * dL_dalpha)
    atomic_float dL_dMean2D[2];     // Screen position gradient
    atomic_float _pad;
};

// Non-atomic read version of RenderGradients same memory layout, for preprocessBackward
struct RenderGradientsRead {
    float dL_dColor[3];
    float dL_dConic[3];
    float dL_dOpacity;
    float dL_dMean2D[2];
    float _pad;
};

// Final Gaussian gradients written by preprocessBackward, read by Adam optimizer
// Plain float no atomics needed since preprocessBackward has exactly one thread per Gaussian
struct GaussianGradients {
    float position_x;
    float position_y;
    float position_z;
    float opacity;
    float scale_x;
    float scale_y;
    float scale_z;
    float _pad1;
    float rotation_x;
    float rotation_y;
    float rotation_z;
    float rotation_w;
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
// exp(4) = 54.6, must match optimizer's MAX_SCALE_TRAIN
constant float MAX_SCALE = 4.0f;

// SSIM gradient constants must match forward SSIM in shaders.metal
constant float SSIM_C1 = 0.01f * 0.01f;
constant float SSIM_C2 = 0.03f * 0.03f;
constant int SSIM_WINDOW_RADIUS = 5;
constant float LAMBDA_DSSIM = 0.2f;
constant float MAX_PIXEL_GRAD_ABS = 100.0f;
// Safety clamp for global atomic accumulation in tiledBackward Stage 1 prevents extreme outliers 
// from causing NaN explosions, at the cost of potentially losing some gradient signal in those cases.
constant bool USE_SAFE_GLOBAL_STAGE1_ACCUM = true;
// Debug mode switches keep OFF by default for root-cause investigation
constant bool ENABLE_BACKWARD_VALUE_CLAMPS = false;
constant bool ENABLE_INTERMEDIATE_MAG_REJECT = false;

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

    // Skip Gaussians with NaN/inf SH
    for (int i = 0; i < 12; i++) {
        if (isnan(g.sh[i]) || isinf(g.sh[i])) {
            projected[tid] = proj;
            return;
        }
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
    
    // Outside frustum — generous margin so off-screen Gaussians with large radii still contribute
    if (abs(ndc.x) > 1.5 || abs(ndc.y) > 1.5) {
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
    
    // Tile coverage cap — 2048 allows ~half-screen coverage (vs 256 which caused holes).
    // Official has no cap, but unlimited causes severe performance regression on Metal.
    uint tilesX = proj.tileMaxX - proj.tileMinX + 1;
    uint tilesY = proj.tileMaxY - proj.tileMinY + 1;
    if (tilesX * tilesY > 2048) {
        proj.radius = 0;
        projected[tid] = proj;
        return;
    }
    
    // Apply sigmoid to opacity
    float rawOpacity = clamp(g.opacity, -8.0f, 8.0f);
    proj.opacity = 1.0 / (1.0 + exp(-rawOpacity));

    // Compute view direction (from camera to Gaussian, normalized)
    float3 toGaussian = float3(g.position) - uniforms.cameraPos;
    float dist = length(toGaussian);
    float3 viewDir = (dist > 0.0001f) ? (toGaussian / dist) : float3(0, 0, 1);
    proj.viewDir = viewDir;

    // color = SH_C0 * dc + 0.5 + SH_C1 * (view-dependent terms)
    // SH layout per channel: [dc, sh1_y, sh1_z, sh1_x]
    // R: sh[0-3], G: sh[4-7], B: sh[8-11]
    float3 dc = float3(g.sh[0], g.sh[4], g.sh[8]);
    float3 sh1_y = float3(g.sh[1], g.sh[5], g.sh[9]);
    float3 sh1_z = float3(g.sh[2], g.sh[6], g.sh[10]);
    float3 sh1_x = float3(g.sh[3], g.sh[7], g.sh[11]);

    float3 color = SH_C0 * dc + 0.5f + SH_C1 * (-sh1_y * viewDir.y + sh1_z * viewDir.z - sh1_x * viewDir.x);

    // Track which channels were clamped negative only 
    // Colors > 1 are valid bright values from SH and should receive gradients
    proj.colorClamped = packed_uchar3(
        (color.x < 0.0f) ? 1 : 0,
        (color.y < 0.0f) ? 1 : 0,
        (color.z < 0.0f) ? 1 : 0
    );

    proj.color = max(color, 0.0f);

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
        
        // Early skip
        if (power > 0.0f) continue;
        
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
    
    // Blend with black background for COLMAP scenes
    float3 bgColor = float3(0.0f, 0.0f, 0.0f);
    color = color + bgColor * T;

    // Store last contributing index for backward pass
    uint pixelIdx = gid.y * uint(uniforms.screenSize.x) + gid.x;
    lastContribIdx[pixelIdx] = hasContrib ? lastIdx : UINT_MAX;

    output.write(float4(color, 1.0), gid);
}

// Tile-local gradient accumulation constants
// Stage 1 accumulates 9 intermediate components per Gaussian:
// dL_dColor[3], dL_dConic[3], dL_dOpacity[1], dL_dMean2D[2]
// 128 Gaussians * 9 floats = 1152 floats = 4.5KB threadgroup memory
constant uint BACKWARD_CHUNK_SIZE = 128;
constant uint NUM_RENDER_GRAD_COMPONENTS = 9;

// Helper: atomic float add for threadgroup memory using CAS loop (atomic_uint stores float bits)
inline void atomicAddTG(threadgroup atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(
        addr, &expected,
        as_type<uint>(as_type<float>(expected) + val),
        memory_order_relaxed, memory_order_relaxed)) {}
}

// SIMD-reduced atomic add reduces 256 threads to 8 atomics 32 threads per simdgroup
inline void simdAtomicAddTG(threadgroup atomic_uint* addr, float val) {
    float sum = simd_sum(val);
    if (simd_is_first()) {
        atomicAddTG(addr, sum);
    }
}

// Two-stage backward pass (matches official 3DGS architecture):
// Stage 1 (tiledBackward): Per-pixel kernel accumulates 9 intermediate gradients per Gaussian
//   - dL_dColor[3], dL_dConic[3], dL_dOpacity[1], dL_dMean2D[2]
//   - Uses threadgroup memory + SIMD reduction, flushes to RenderGradientsAtomic buffer
// Stage 2 preprocessBackward: Per-Gaussian kernel computes all downstream gradients
//   - SH, scale, rotation, position, viewspace gradients — NO atomics needed

kernel void tiledBackward(
    device const ProjectedGaussian* projected [[buffer(0)]],
    device RenderGradientsAtomic* renderGrads [[buffer(1)]],
    device const uint* sortedIndices [[buffer(2)]],
    device const TileRange* tileRanges [[buffer(3)]],
    constant TiledUniforms& uniforms [[buffer(4)]],
    device const uint* lastContribIdx [[buffer(5)]],
    device const float* pixelGradients [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    // Threadgroup memory for gradient accumulation as atomic_uint, bits interpreted as float
    // Layout: [CHUNK_SIZE][9] - 9 intermediate gradient components per Gaussian
    // Components: dL_dColor(3), dL_dConic(3), dL_dOpacity(1), dL_dMean2D(2)
    threadgroup atomic_uint tgGrads[BACKWARD_CHUNK_SIZE * NUM_RENDER_GRAD_COMPONENTS];
    threadgroup uint tgGaussianIdx[BACKWARD_CHUNK_SIZE];

    // Bounds check
    bool valid = (gid.x < uint(uniforms.screenSize.x) && gid.y < uint(uniforms.screenSize.y));

    // Get tile range
    uint tileIdx = tgid.y * uniforms.numTilesX + tgid.x;
    TileRange range = tileRanges[tileIdx];

    // Per-pixel state
    uint pixelIdx = valid ? (gid.y * uint(uniforms.screenSize.x) + gid.x) : 0;
    uint lastIdx = valid ? lastContribIdx[pixelIdx] : UINT_MAX;
    bool hasContrib = valid && (lastIdx != UINT_MAX);
    uint endIdx = hasContrib ? min(lastIdx + 1, range.start + range.count) : 0;

    // Pixel position and pre-computed gradient combined L1 + D-SSIM from separate pass
    float2 pixelPos = float2(gid) + 0.5;
    float3 dL_dPixel = float3(0);
    if (valid) {
        uint gradBase = pixelIdx * 3;
        dL_dPixel = float3(pixelGradients[gradBase], pixelGradients[gradBase + 1], pixelGradients[gradBase + 2]);
        if (!isfinite(dL_dPixel.x) || !isfinite(dL_dPixel.y) || !isfinite(dL_dPixel.z)) {
            dL_dPixel = float3(0.0f);
        } else if (ENABLE_BACKWARD_VALUE_CLAMPS) {
            dL_dPixel = clamp(dL_dPixel, float3(-1.0f), float3(1.0f));
        }
    }

    // Pre-compute T_final by replaying the forward pass exactly
    float T_final = 1.0;
    if (hasContrib) {
        for (uint sortIdx = range.start; sortIdx < endIdx; sortIdx++) {
            uint gIdx = sortedIndices[sortIdx];
            if (gIdx >= uniforms.numGaussians) continue;

            ProjectedGaussian p = projected[gIdx];
            if (p.radius <= 0) continue;

            float conicMag = abs(p.conic.x) + abs(p.conic.y) + abs(p.conic.z);
            if (conicMag < 0.0001) continue;

            float2 d = pixelPos - p.screenPos;
            float power = -0.5 * (p.conic.x * d.x * d.x +
                                  2.0 * p.conic.y * d.x * d.y +
                                  p.conic.z * d.y * d.y);

            if (power > 0.0) continue;

            float G = exp(power);
            float alpha = min(p.opacity * G, 0.99f);

            if (alpha < 1.0 / 255.0) continue;

            T_final *= (1.0 - alpha);
        }
    }

    // Initialize backward pass state
    float T = T_final;
    float3 bgColor = float3(0.0);
    float3 accum_rec = bgColor;

    // Process the tile's Gaussians in chunks
    uint totalCount = range.count;
    uint numChunks = (totalCount + BACKWARD_CHUNK_SIZE - 1) / BACKWARD_CHUNK_SIZE;

    // Process chunks back-to-front high sortIdx to low
    for (int chunk = int(numChunks) - 1; chunk >= 0; chunk--) {
        uint chunkStart = range.start + uint(chunk) * BACKWARD_CHUNK_SIZE;
        uint chunkEnd = min(chunkStart + BACKWARD_CHUNK_SIZE, range.start + totalCount);
        uint chunkSize = chunkEnd - chunkStart;

        // Clear threadgroup accumulators all threads participate
        for (uint i = tid; i < BACKWARD_CHUNK_SIZE * NUM_RENDER_GRAD_COMPONENTS; i += 256) {
            atomic_store_explicit(&tgGrads[i], 0u, memory_order_relaxed);
        }

        // Cache Gaussian indices for this chunk
        for (uint i = tid; i < chunkSize; i += 256) {
            uint idx = sortedIndices[chunkStart + i];
            tgGaussianIdx[i] = (idx < uniforms.numGaussians) ? idx : UINT_MAX;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Process chunk back-to-front (each pixel maintains its own T state)
        // IMPORTANT: simdAtomicAddTG uses simd_sum/simd_is_first which require
        // ALL threads in the simdgroup to be converged. We must call them outside
        // all divergent branches non-contributing threads pass 0.
        for (int sortIdx = int(chunkEnd) - 1; sortIdx >= int(chunkStart); sortIdx--) {
            uint localIdx = uint(sortIdx) - chunkStart;
            uint gIdx = tgGaussianIdx[localIdx];

            // Per-thread gradient contributions 0 for non-contributing threads
            float grad_color_x = 0.0f, grad_color_y = 0.0f, grad_color_z = 0.0f;
            float grad_conic_x = 0.0f, grad_conic_y = 0.0f, grad_conic_z = 0.0f;
            float grad_opacity = 0.0f;
            float grad_mean2d_x = 0.0f, grad_mean2d_y = 0.0f;

            bool process = hasContrib && (uint(sortIdx) < endIdx) && (gIdx < uniforms.numGaussians);

            if (process) {
                ProjectedGaussian p = projected[gIdx];

                if (p.radius > 0) {
                    float conicMag = abs(p.conic.x) + abs(p.conic.y) + abs(p.conic.z);

                    if (conicMag >= 0.0001) {
                        float2 d = pixelPos - p.screenPos;
                        float power = -0.5 * (p.conic.x * d.x * d.x +
                                              2.0 * p.conic.y * d.x * d.y +
                                              p.conic.z * d.y * d.y);

                        if (power <= 0.0) {
                            float G = exp(power);
                            float alpha = min(p.opacity * G, 0.99f);

                            if (alpha >= 1.0 / 255.0) {
                                float one_minus_alpha = max(1.0f - alpha, 0.0001f);
                                float T_before = T / one_minus_alpha;
                                T_before = min(T_before, 1.0f);

                                float weight = alpha * T_before;

                                // dL/dAlpha (official 3DGS formula)
                                float dL_dAlpha = T_before * dot(dL_dPixel, p.color - accum_rec);
                                float bg_dot_dpixel = dot(bgColor, dL_dPixel);
                                dL_dAlpha += (-T_final / one_minus_alpha) * bg_dot_dpixel;
                                if (ENABLE_BACKWARD_VALUE_CLAMPS) dL_dAlpha = clamp(dL_dAlpha, -1e3f, 1e3f);

                                // Update accum_rec and T for back-to-front traversal
                                accum_rec = alpha * p.color + (1.0f - alpha) * accum_rec;
                                T = T_before;


                                // 1. Color gradient: dL/d(color_g) = weight * dL/d(pixel)
                                float3 dL_dColor = dL_dPixel * weight;
                                grad_color_x = dL_dColor.x;
                                grad_color_y = dL_dColor.y;
                                grad_color_z = dL_dColor.z;

                                // 2. Opacity intermediate: G * dL/dAlpha
                                grad_opacity = G * dL_dAlpha;

                                // 3. Screen position gradient
                                float sig = p.opacity;
                                float dL_dG = dL_dAlpha * sig;
                                float gdx = G * d.x;
                                float gdy = G * d.y;
                                float dG_ddelx = -gdx * p.conic.x - gdy * p.conic.y;
                                float dG_ddely = -gdy * p.conic.z - gdx * p.conic.y;
                                grad_mean2d_x = dL_dG * (-dG_ddelx);
                                grad_mean2d_y = dL_dG * (-dG_ddely);

                                // 4. Conic gradient
                                grad_conic_x = -0.5f * dL_dG * G * d.x * d.x;
                                grad_conic_y = -1.0f * dL_dG * G * d.x * d.y;
                                grad_conic_z = -0.5f * dL_dG * G * d.y * d.y;
                                if (ENABLE_BACKWARD_VALUE_CLAMPS) {
                                    grad_color_x = clamp(grad_color_x, -1e3f, 1e3f);
                                    grad_color_y = clamp(grad_color_y, -1e3f, 1e3f);
                                    grad_color_z = clamp(grad_color_z, -1e3f, 1e3f);
                                    grad_opacity = clamp(grad_opacity, -1e3f, 1e3f);
                                    grad_mean2d_x = clamp(grad_mean2d_x, -1e3f, 1e3f);
                                    grad_mean2d_y = clamp(grad_mean2d_y, -1e3f, 1e3f);
                                    grad_conic_x = clamp(grad_conic_x, -1e3f, 1e3f);
                                    grad_conic_y = clamp(grad_conic_y, -1e3f, 1e3f);
                                    grad_conic_z = clamp(grad_conic_z, -1e3f, 1e3f);
                                }
                            }
                        }
                    }
                }
            }

            if (USE_SAFE_GLOBAL_STAGE1_ACCUM) {
                if (process && gIdx < uniforms.numGaussians) {
                    #define SAFE_ADD_GLOBAL(dest_atomic, val) { \
                        if (isfinite(val) && abs(val) < 1e20f && (val != 0.0f)) { \
                            atomic_fetch_add_explicit(&(dest_atomic), (val), memory_order_relaxed); \
                        } \
                    }
                    SAFE_ADD_GLOBAL(renderGrads[gIdx].dL_dColor[0], grad_color_x);
                    SAFE_ADD_GLOBAL(renderGrads[gIdx].dL_dColor[1], grad_color_y);
                    SAFE_ADD_GLOBAL(renderGrads[gIdx].dL_dColor[2], grad_color_z);
                    SAFE_ADD_GLOBAL(renderGrads[gIdx].dL_dConic[0], grad_conic_x);
                    SAFE_ADD_GLOBAL(renderGrads[gIdx].dL_dConic[1], grad_conic_y);
                    SAFE_ADD_GLOBAL(renderGrads[gIdx].dL_dConic[2], grad_conic_z);
                    SAFE_ADD_GLOBAL(renderGrads[gIdx].dL_dOpacity, grad_opacity);
                    SAFE_ADD_GLOBAL(renderGrads[gIdx].dL_dMean2D[0], grad_mean2d_x);
                    SAFE_ADD_GLOBAL(renderGrads[gIdx].dL_dMean2D[1], grad_mean2d_y);
                    #undef SAFE_ADD_GLOBAL
                }
            } else {
                // ALL threads in the simdgroup participate in simd_sum here (converged)
                // Non-contributing threads contribute 0
                uint baseIdx = localIdx * NUM_RENDER_GRAD_COMPONENTS;
                simdAtomicAddTG(&tgGrads[baseIdx + 0], grad_color_x);
                simdAtomicAddTG(&tgGrads[baseIdx + 1], grad_color_y);
                simdAtomicAddTG(&tgGrads[baseIdx + 2], grad_color_z);
                simdAtomicAddTG(&tgGrads[baseIdx + 3], grad_conic_x);
                simdAtomicAddTG(&tgGrads[baseIdx + 4], grad_conic_y);
                simdAtomicAddTG(&tgGrads[baseIdx + 5], grad_conic_z);
                simdAtomicAddTG(&tgGrads[baseIdx + 6], grad_opacity);
                simdAtomicAddTG(&tgGrads[baseIdx + 7], grad_mean2d_x);
                simdAtomicAddTG(&tgGrads[baseIdx + 8], grad_mean2d_y);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Flush accumulated intermediates to global RenderGradients buffer
        if (!USE_SAFE_GLOBAL_STAGE1_ACCUM) {
            for (uint i = tid; i < chunkSize; i += 256) {
                uint gIdx = tgGaussianIdx[i];
                if (gIdx == UINT_MAX) continue;
                uint baseIdx = i * NUM_RENDER_GRAD_COMPONENTS;

                #define FLUSH_RENDER_GRAD(slot, dest_atomic) { \
                    float val = as_type<float>(atomic_load_explicit(&tgGrads[baseIdx + (slot)], memory_order_relaxed)); \
                    if (val != 0 && isfinite(val)) \
                        atomic_fetch_add_explicit(&(dest_atomic), val, memory_order_relaxed); \
                }
                FLUSH_RENDER_GRAD(0, renderGrads[gIdx].dL_dColor[0]);
                FLUSH_RENDER_GRAD(1, renderGrads[gIdx].dL_dColor[1]);
                FLUSH_RENDER_GRAD(2, renderGrads[gIdx].dL_dColor[2]);
                FLUSH_RENDER_GRAD(3, renderGrads[gIdx].dL_dConic[0]);
                FLUSH_RENDER_GRAD(4, renderGrads[gIdx].dL_dConic[1]);
                FLUSH_RENDER_GRAD(5, renderGrads[gIdx].dL_dConic[2]);
                FLUSH_RENDER_GRAD(6, renderGrads[gIdx].dL_dOpacity);
                FLUSH_RENDER_GRAD(7, renderGrads[gIdx].dL_dMean2D[0]);
                FLUSH_RENDER_GRAD(8, renderGrads[gIdx].dL_dMean2D[1]);
                #undef FLUSH_RENDER_GRAD
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Stage 2: Per-Gaussian preprocess backward pass
// Reads accumulated intermediate gradients and computes final Gaussian parameter gradients.
// One thread per Gaussian — no atomics needed.
kernel void preprocessBackward(
    device const Gaussian* gaussians [[buffer(0)]],
    device const ProjectedGaussian* projected [[buffer(1)]],
    device const RenderGradientsRead* renderGrads [[buffer(2)]],
    device GaussianGradients* gradients [[buffer(3)]],
    constant TiledUniforms& uniforms [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= uniforms.numGaussians) return;

    ProjectedGaussian p = projected[tid];

    // Skip invalid Gaussians (no screen contribution)
    if (p.radius <= 0) {
        // Zero out gradients for this Gaussian
        gradients[tid].position_x = 0;
        gradients[tid].position_y = 0;
        gradients[tid].position_z = 0;
        gradients[tid].opacity = 0;
        gradients[tid].scale_x = 0;
        gradients[tid].scale_y = 0;
        gradients[tid].scale_z = 0;
        gradients[tid]._pad1 = 0;
        gradients[tid].rotation_x = 0;
        gradients[tid].rotation_y = 0;
        gradients[tid].rotation_z = 0;
        gradients[tid].rotation_w = 0;
        for (int i = 0; i < 12; i++) gradients[tid].sh[i] = 0;
        gradients[tid].viewspace_grad_x = 0;
        gradients[tid].viewspace_grad_y = 0;
        gradients[tid]._pad2 = 0;
        gradients[tid]._pad3 = 0;
        return;
    }

    // Read accumulated intermediate gradients
    RenderGradientsRead rg = renderGrads[tid];

    // NaN/Inf safety check on intermediates
    bool anyBad = false;
    for (int i = 0; i < 3; i++) {
        if (!isfinite(rg.dL_dColor[i]) || !isfinite(rg.dL_dConic[i])) anyBad = true;
        if (ENABLE_INTERMEDIATE_MAG_REJECT && (abs(rg.dL_dColor[i]) > 1e6f || abs(rg.dL_dConic[i]) > 1e6f)) anyBad = true;
    }
    if (!isfinite(rg.dL_dOpacity) || !isfinite(rg.dL_dMean2D[0]) || !isfinite(rg.dL_dMean2D[1])) anyBad = true;
    if (ENABLE_INTERMEDIATE_MAG_REJECT && (abs(rg.dL_dOpacity) > 1e6f || abs(rg.dL_dMean2D[0]) > 1e6f || abs(rg.dL_dMean2D[1]) > 1e6f)) anyBad = true;

    if (anyBad) {
        // Zero out all gradients for this Gaussian
        gradients[tid].position_x = 0;
        gradients[tid].position_y = 0;
        gradients[tid].position_z = 0;
        gradients[tid].opacity = 0;
        gradients[tid].scale_x = 0;
        gradients[tid].scale_y = 0;
        gradients[tid].scale_z = 0;
        gradients[tid]._pad1 = 0;
        gradients[tid].rotation_x = 0;
        gradients[tid].rotation_y = 0;
        gradients[tid].rotation_z = 0;
        gradients[tid].rotation_w = 0;
        for (int i = 0; i < 12; i++) gradients[tid].sh[i] = 0;
        gradients[tid].viewspace_grad_x = 0;
        gradients[tid].viewspace_grad_y = 0;
        gradients[tid]._pad2 = 0;
        gradients[tid]._pad3 = 0;
        return;
    }

    // === Opacity gradient ===
    // Intermediate stores sum_pixels(G * dL_dAlpha)
    // Final: dL/d(raw_opacity) = intermediate * sig * (1-sig)
    float sig = p.opacity;  // Already sigmoid'd in projection
    float dL_dRawOpacity = rg.dL_dOpacity * sig * (1.0f - sig);
    if (ENABLE_BACKWARD_VALUE_CLAMPS) dL_dRawOpacity = clamp(dL_dRawOpacity, -1e3f, 1e3f);

    // === Color / SH gradients ===
    float3 dL_dColor = float3(rg.dL_dColor[0], rg.dL_dColor[1], rg.dL_dColor[2]);
    if (ENABLE_BACKWARD_VALUE_CLAMPS) dL_dColor = clamp(dL_dColor, float3(-1e3f), float3(1e3f));

    // Apply clamping mask (zero gradient for channels that were clamped negative)
    dL_dColor.x *= (p.colorClamped.x == 0) ? 1.0f : 0.0f;
    dL_dColor.y *= (p.colorClamped.y == 0) ? 1.0f : 0.0f;
    dL_dColor.z *= (p.colorClamped.z == 0) ? 1.0f : 0.0f;

    // DC SH gradient (degree 0)
    float3 sh_dc_grad = dL_dColor * SH_C0;

    // Degree-1 SH gradients
    float3 viewDir = float3(p.viewDir);
    float3 sh1_y_grad = dL_dColor * SH_C1 * (-viewDir.y);
    float3 sh1_z_grad = dL_dColor * SH_C1 * viewDir.z;
    float3 sh1_x_grad = dL_dColor * SH_C1 * (-viewDir.x);

    // === Screen position -> World position gradient ===
    float2 dL_dScreenPos = float2(rg.dL_dMean2D[0], rg.dL_dMean2D[1]);
    if (ENABLE_BACKWARD_VALUE_CLAMPS) dL_dScreenPos = clamp(dL_dScreenPos, float2(-1e3f), float2(1e3f));
    float fx = uniforms.focalLength.x;
    float fy = uniforms.focalLength.y;
    float z = p.depth;
    if (abs(z) < 1e-4f || !isfinite(z)) {
        gradients[tid].position_x = 0;
        gradients[tid].position_y = 0;
        gradients[tid].position_z = 0;
        gradients[tid].opacity = 0;
        gradients[tid].scale_x = 0;
        gradients[tid].scale_y = 0;
        gradients[tid].scale_z = 0;
        gradients[tid]._pad1 = 0;
        gradients[tid].rotation_x = 0;
        gradients[tid].rotation_y = 0;
        gradients[tid].rotation_z = 0;
        gradients[tid].rotation_w = 0;
        for (int i = 0; i < 12; i++) gradients[tid].sh[i] = 0;
        gradients[tid].viewspace_grad_x = 0;
        gradients[tid].viewspace_grad_y = 0;
        gradients[tid]._pad2 = 0;
        gradients[tid]._pad3 = 0;
        return;
    }
    float txtz = p.viewPos_xy.x / z;
    float tytz = p.viewPos_xy.y / z;

    float3 dL_dViewPos;
    dL_dViewPos.x = dL_dScreenPos.x * fx / z;
    dL_dViewPos.y = dL_dScreenPos.y * fy / z;
    dL_dViewPos.z = -dL_dScreenPos.x * fx * txtz / z
                    -dL_dScreenPos.y * fy * tytz / z;

    // View rotation matrix
    float3x3 viewRot = float3x3(
        uniforms.viewMatrix[0].xyz,
        uniforms.viewMatrix[1].xyz,
        uniforms.viewMatrix[2].xyz
    );

    // === Conic -> Cov2D -> Cov3D -> Scale/Rotation gradient ===
    float3 dL_dConic = float3(rg.dL_dConic[0], rg.dL_dConic[1], rg.dL_dConic[2]);
    if (ENABLE_BACKWARD_VALUE_CLAMPS) dL_dConic = clamp(dL_dConic, float3(-1e3f), float3(1e3f));

    // Cov2D gradient from conic
    float cov_a = p.cov2D.x;
    float cov_b = p.cov2D.y;
    float cov_c = p.cov2D.z;
    float denom = cov_a * cov_c - cov_b * cov_b;
    float denom2inv = 1.0f / (denom * denom + 1e-4f);

    float3 dL_dCov2D;
    dL_dCov2D.x = denom2inv * (-cov_c * cov_c * dL_dConic.x
                               + 2.0f * cov_b * cov_c * dL_dConic.y
                               + (denom - cov_a * cov_c) * dL_dConic.z);
    dL_dCov2D.z = denom2inv * (-cov_a * cov_a * dL_dConic.z
                               + 2.0f * cov_a * cov_b * dL_dConic.y
                               + (denom - cov_a * cov_c) * dL_dConic.x);
    dL_dCov2D.y = denom2inv * (2.0f * cov_b * cov_c * dL_dConic.x
                               - (denom + 2.0f * cov_b * cov_b) * dL_dConic.y
                               + 2.0f * cov_a * cov_b * dL_dConic.z);

    // Clamp to prevent near-singular cov2D from creating runaway gradients
    if (ENABLE_BACKWARD_VALUE_CLAMPS) dL_dCov2D = clamp(dL_dCov2D, float3(-1e4f), float3(1e4f));

    // Jacobian
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

    // Cov2D -> Cov3D gradient (0.5 on off-diagonal for symmetric matrix)
    float3x3 dL_dCov2D_mat = float3x3(
        float3(dL_dCov2D.x, 0.5f * dL_dCov2D.y, 0),
        float3(0.5f * dL_dCov2D.y, dL_dCov2D.z, 0),
        float3(0, 0, 0)
    );
    float3x3 dL_dCov3D = transpose(T_mat) * dL_dCov2D_mat * T_mat;

    // Scale and Rotation gradients
    Gaussian g_orig = gaussians[tid];
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

    // Jacobian contribution to position gradient
    {
        float3x3 Sigma3D = M * transpose(M);
        float3x3 dL_dT_mat = 2.0f * dL_dCov2D_mat * T_mat * Sigma3D;
        float3x3 dL_dJ_mat = dL_dT_mat * transpose(viewRot);

        float dL_dJ00 = dL_dJ_mat[0][0];
        float dL_dJ02 = dL_dJ_mat[2][0];
        float dL_dJ11 = dL_dJ_mat[1][1];
        float dL_dJ12 = dL_dJ_mat[2][1];

        float z_sq = z * z;
        dL_dViewPos.x += dL_dJ02 * (-fx / z_sq);
        dL_dViewPos.y += dL_dJ12 * (-fy / z_sq);
        dL_dViewPos.z += dL_dJ00 * (-fx / z_sq)
                      + dL_dJ11 * (-fy / z_sq)
                      + dL_dJ02 * (2.0f * fx * txtz / z_sq)
                      + dL_dJ12 * (2.0f * fy * tytz / z_sq);
    }

    float3 dL_dWorldPos = transpose(viewRot) * dL_dViewPos;
    if (ENABLE_BACKWARD_VALUE_CLAMPS) dL_dWorldPos = clamp(dL_dWorldPos, float3(-1e3f), float3(1e3f));

    // Scale gradient
    float3x3 Rt = transpose(R);
    float3x3 Rt_dLdM = Rt * dL_dM;
    float3 dL_dScale_val = float3(Rt_dLdM[0][0], Rt_dLdM[1][1], Rt_dLdM[2][2]);
    float3 dL_dLogScale = dL_dScale_val * scale;
    if (ENABLE_BACKWARD_VALUE_CLAMPS) dL_dLogScale = clamp(dL_dLogScale, float3(-1e3f), float3(1e3f));

    // Rotation gradient (quaternion)
    float3x3 dL_dR = float3x3(
        dL_dM[0] * scale.x,
        dL_dM[1] * scale.y,
        dL_dM[2] * scale.z
    );
    float3x3 dL_dMt_scaled = transpose(dL_dR);

    float4 dL_dq;
    dL_dq.x = 2.0f * (z_q * (dL_dMt_scaled[1][0] - dL_dMt_scaled[0][1]) +
                     y_q * (dL_dMt_scaled[0][2] - dL_dMt_scaled[2][0]) +
                     x_q * (dL_dMt_scaled[2][1] - dL_dMt_scaled[1][2]));
    dL_dq.y = 2.0f * (y_q * (dL_dMt_scaled[0][1] + dL_dMt_scaled[1][0]) +
                     z_q * (dL_dMt_scaled[0][2] + dL_dMt_scaled[2][0]) +
                     r * (dL_dMt_scaled[2][1] - dL_dMt_scaled[1][2]) -
                     2.0f * x_q * (dL_dMt_scaled[2][2] + dL_dMt_scaled[1][1]));
    dL_dq.z = 2.0f * (x_q * (dL_dMt_scaled[0][1] + dL_dMt_scaled[1][0]) +
                     r * (dL_dMt_scaled[0][2] - dL_dMt_scaled[2][0]) +
                     z_q * (dL_dMt_scaled[2][1] + dL_dMt_scaled[1][2]) -
                     2.0f * y_q * (dL_dMt_scaled[2][2] + dL_dMt_scaled[0][0]));
    dL_dq.w = 2.0f * (r * (dL_dMt_scaled[1][0] - dL_dMt_scaled[0][1]) +
                     x_q * (dL_dMt_scaled[0][2] + dL_dMt_scaled[2][0]) +
                     y_q * (dL_dMt_scaled[2][1] + dL_dMt_scaled[1][2]) -
                     2.0f * z_q * (dL_dMt_scaled[1][1] + dL_dMt_scaled[0][0]));
    if (ENABLE_BACKWARD_VALUE_CLAMPS) dL_dq = clamp(dL_dq, float4(-1e3f), float4(1e3f));

    // Viewspace gradient (NDC-scaled for density control)
    float viewGradX = dL_dScreenPos.x * 0.5f * uniforms.screenSize.x;
    float viewGradY = dL_dScreenPos.y * 0.5f * uniforms.screenSize.y;
    if (ENABLE_BACKWARD_VALUE_CLAMPS) {
        viewGradX = clamp(viewGradX, -1e3f, 1e3f);
        viewGradY = clamp(viewGradY, -1e3f, 1e3f);

        sh_dc_grad = clamp(sh_dc_grad, float3(-1e3f), float3(1e3f));
        sh1_y_grad = clamp(sh1_y_grad, float3(-1e3f), float3(1e3f));
        sh1_z_grad = clamp(sh1_z_grad, float3(-1e3f), float3(1e3f));
        sh1_x_grad = clamp(sh1_x_grad, float3(-1e3f), float3(1e3f));
    }

    // === Write final gradients (non-atomic, one thread per Gaussian) ===
    gradients[tid].position_x = dL_dWorldPos.x;
    gradients[tid].position_y = dL_dWorldPos.y;
    gradients[tid].position_z = dL_dWorldPos.z;
    gradients[tid].opacity = dL_dRawOpacity;
    gradients[tid].scale_x = dL_dLogScale.x;
    gradients[tid].scale_y = dL_dLogScale.y;
    gradients[tid].scale_z = dL_dLogScale.z;
    gradients[tid]._pad1 = 0;
    gradients[tid].rotation_x = dL_dq.x;
    gradients[tid].rotation_y = dL_dq.y;
    gradients[tid].rotation_z = dL_dq.z;
    gradients[tid].rotation_w = dL_dq.w;
    // SH layout: R[dc,y,z,x], G[dc,y,z,x], B[dc,y,z,x]
    gradients[tid].sh[0] = sh_dc_grad.r;
    gradients[tid].sh[1] = sh1_y_grad.r;
    gradients[tid].sh[2] = sh1_z_grad.r;
    gradients[tid].sh[3] = sh1_x_grad.r;
    gradients[tid].sh[4] = sh_dc_grad.g;
    gradients[tid].sh[5] = sh1_y_grad.g;
    gradients[tid].sh[6] = sh1_z_grad.g;
    gradients[tid].sh[7] = sh1_x_grad.g;
    gradients[tid].sh[8] = sh_dc_grad.b;
    gradients[tid].sh[9] = sh1_y_grad.b;
    gradients[tid].sh[10] = sh1_z_grad.b;
    gradients[tid].sh[11] = sh1_x_grad.b;
    gradients[tid].viewspace_grad_x = viewGradX;
    gradients[tid].viewspace_grad_y = viewGradY;
    gradients[tid]._pad2 = 0;
    gradients[tid]._pad3 = 0;
}

// GPU Pair Generation
// Each thread handles one Gaussian and writes all its tile-pairs atomically
constant float GPU_MIN_OPACITY = 0.005f;
// Must match projection kernel's tile cap (2048) — otherwise Gaussians covering 257-2048 tiles
// pass projection (valid radius/conic) but generate zero tile pairs → invisible zombie Gaussians
// that waste memory budget and never get pruned
constant uint GPU_MAX_TILES_PER_GAUSSIAN = 2048u;

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
    
    // Reserve write positions atomically.
    // Important: we intentionally allow the counter to exceed maxPairs, so the CPU can detect
    // overflow (`totalPairs > maxPairs`), grow buffers, and re-run pair generation.
    uint writePos = atomic_fetch_add_explicit(writeCounter, tileCount, memory_order_relaxed);
    
    // Check buffer bounds (avoid OOB writes).
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

// ==== D-SSIM Gradient Computation (Two-Pass Approach) ====
//
// The combined loss is: L = (1-lambda)*L1_mean + lambda*DSSIM_mean
// where DSSIM_mean = mean over pixels of mean over channels of (1-SSIM)/2
//
// The gradient dL/dX(q) decomposes via the SSIM windowed statistics into:
//   dL_DSSIM/dX(q) = conv(K, w)(q) + X(q)*conv(L, w)(q) + Y(q)*conv(M, w)(q)
// where K, L, M are per-pixel coefficient maps computed from SSIM partial derivatives.

// Pass 1: Compute SSIM gradient coefficient maps (K, L, M per pixel per channel)
kernel void computeSSIMGradCoeffs(
    texture2d<float, access::read> rendered [[texture(0)]],
    texture2d<float, access::read> groundTruth [[texture(1)]],
    device float* coeffK [[buffer(0)]],
    device float* coeffL [[buffer(1)]],
    device float* coeffM [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint width = rendered.get_width();
    uint height = rendered.get_height();
    if (gid.x >= width || gid.y >= height) return;

    float numPixels = float(width * height);
    float sigma = 1.5f;
    float two_sigma_sq = 2.0f * sigma * sigma;

    // Compute weighted means (same as forward SSIM)
    float3 mu_x = float3(0);
    float3 mu_y = float3(0);
    float weight_sum = 0.0f;

    for (int dy = -SSIM_WINDOW_RADIUS; dy <= SSIM_WINDOW_RADIUS; dy++) {
        for (int dx = -SSIM_WINDOW_RADIUS; dx <= SSIM_WINDOW_RADIUS; dx++) {
            int px = clamp(int(gid.x) + dx, 0, int(width) - 1);
            int py = clamp(int(gid.y) + dy, 0, int(height) - 1);
            float dist_sq = float(dx * dx + dy * dy);
            float w = exp(-dist_sq / two_sigma_sq);
            weight_sum += w;
            mu_x += w * rendered.read(uint2(px, py)).rgb;
            mu_y += w * groundTruth.read(uint2(px, py)).rgb;
        }
    }
    mu_x /= weight_sum;
    mu_y /= weight_sum;

    // Compute weighted variances and covariance
    float3 sigma_x_sq = float3(0);
    float3 sigma_y_sq = float3(0);
    float3 sigma_xy = float3(0);

    for (int dy = -SSIM_WINDOW_RADIUS; dy <= SSIM_WINDOW_RADIUS; dy++) {
        for (int dx = -SSIM_WINDOW_RADIUS; dx <= SSIM_WINDOW_RADIUS; dx++) {
            int px = clamp(int(gid.x) + dx, 0, int(width) - 1);
            int py = clamp(int(gid.y) + dy, 0, int(height) - 1);
            float dist_sq = float(dx * dx + dy * dy);
            float w = exp(-dist_sq / two_sigma_sq);
            float3 dx_val = rendered.read(uint2(px, py)).rgb - mu_x;
            float3 dy_val = groundTruth.read(uint2(px, py)).rgb - mu_y;
            sigma_x_sq += w * dx_val * dx_val;
            sigma_y_sq += w * dy_val * dy_val;
            sigma_xy += w * dx_val * dy_val;
        }
    }
    sigma_x_sq /= weight_sum;
    sigma_y_sq /= weight_sum;
    sigma_xy /= weight_sum;

    // SSIM components per channel
    float3 N1 = 2.0f * mu_x * mu_y + SSIM_C1;
    float3 N2 = 2.0f * sigma_xy + SSIM_C2;
    float3 D1 = mu_x * mu_x + mu_y * mu_y + SSIM_C1;
    float3 D2 = sigma_x_sq + sigma_y_sq + SSIM_C2;

    // Partial derivatives of SSIM w.r.t. the 3 statistics depending on X
    float3 D1D2 = D1 * D2;
    float3 inv_D1D2 = 1.0f / (D1D2 + 1e-8f);
    float3 SSIM_val = N1 * N2 * inv_D1D2;

    // dSSIM/dmu_x = 2*mu_y*N2/(D1*D2) - 2*mu_x*SSIM/D1
    float3 dSSIM_dmu_x = 2.0f * mu_y * N2 * inv_D1D2
                        - 2.0f * mu_x * SSIM_val / (D1 + 1e-8f);

    // dSSIM/dsigma_x_sq = -SSIM/D2
    float3 dSSIM_dsigma_x_sq = -SSIM_val / (D2 + 1e-8f);

    // dSSIM/dsigma_xy = 2*N1/(D1*D2)
    float3 dSSIM_dsigma_xy = 2.0f * N1 * inv_D1D2;

    // Scale by dL/dSSIM = -lambda / (6 * N_pixels)
    // DSSIM = mean_pixels(mean_channels((1-SSIM)/2))
    // dL/dSSIM_c(p) = lambda * (-1/(2*3*N))
    float dL_dSSIM_scalar = -LAMBDA_DSSIM / (6.0f * numPixels);

    float3 a = dL_dSSIM_scalar * dSSIM_dmu_x;
    float3 b = dL_dSSIM_scalar * dSSIM_dsigma_x_sq;
    float3 c_val = dL_dSSIM_scalar * dSSIM_dsigma_xy;

    // K, L, M coefficients (pre-divided by weight_sum for clean convolution)
    float inv_W = 1.0f / weight_sum;
    float3 K = (a - 2.0f * b * mu_x - c_val * mu_y) * inv_W;
    float3 L = 2.0f * b * inv_W;
    float3 M = c_val * inv_W;

    // Source sanitization: prevent single bad SSIM window from poisoning neighboring pixels
    if (!isfinite(K.x) || !isfinite(K.y) || !isfinite(K.z)) K = float3(0.0f);
    if (!isfinite(L.x) || !isfinite(L.y) || !isfinite(L.z)) L = float3(0.0f);
    if (!isfinite(M.x) || !isfinite(M.y) || !isfinite(M.z)) M = float3(0.0f);
    K = clamp(K, float3(-MAX_PIXEL_GRAD_ABS), float3(MAX_PIXEL_GRAD_ABS));
    L = clamp(L, float3(-MAX_PIXEL_GRAD_ABS), float3(MAX_PIXEL_GRAD_ABS));
    M = clamp(M, float3(-MAX_PIXEL_GRAD_ABS), float3(MAX_PIXEL_GRAD_ABS));

    // Store (flat float layout: 3 floats per pixel)
    uint idx = gid.y * width + gid.x;
    uint base = idx * 3;
    coeffK[base + 0] = K.x; coeffK[base + 1] = K.y; coeffK[base + 2] = K.z;
    coeffL[base + 0] = L.x; coeffL[base + 1] = L.y; coeffL[base + 2] = L.z;
    coeffM[base + 0] = M.x; coeffM[base + 1] = M.y; coeffM[base + 2] = M.z;
}

// Pass 2: Convolve K, L, M with Gaussian kernel and combine with L1 gradient
// Output: per-pixel gradient dL/dX = (1-lambda)*dL1/dX + dL_DSSIM/dX
kernel void computePixelGradient(
    texture2d<float, access::read> rendered [[texture(0)]],
    texture2d<float, access::read> groundTruth [[texture(1)]],
    device const float* coeffK [[buffer(0)]],
    device const float* coeffL [[buffer(1)]],
    device const float* coeffM [[buffer(2)]],
    device float* pixelGradients [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint width = rendered.get_width();
    uint height = rendered.get_height();
    if (gid.x >= width || gid.y >= height) return;

    float numPixels = float(width * height);
    float sigma = 1.5f;
    float two_sigma_sq = 2.0f * sigma * sigma;

    // Convolve K, L, M with Gaussian kernel
    float3 convK = float3(0);
    float3 convL = float3(0);
    float3 convM = float3(0);

    for (int dy = -SSIM_WINDOW_RADIUS; dy <= SSIM_WINDOW_RADIUS; dy++) {
        for (int dx = -SSIM_WINDOW_RADIUS; dx <= SSIM_WINDOW_RADIUS; dx++) {
            int px = clamp(int(gid.x) + dx, 0, int(width) - 1);
            int py = clamp(int(gid.y) + dy, 0, int(height) - 1);
            float dist_sq = float(dx * dx + dy * dy);
            float w = exp(-dist_sq / two_sigma_sq);

            uint nIdx = uint(py) * width + uint(px);
            uint nBase = nIdx * 3;
            convK += w * float3(coeffK[nBase], coeffK[nBase + 1], coeffK[nBase + 2]);
            convL += w * float3(coeffL[nBase], coeffL[nBase + 1], coeffL[nBase + 2]);
            convM += w * float3(coeffM[nBase], coeffM[nBase + 1], coeffM[nBase + 2]);
        }
    }

    // Read pixel values
    float3 X = rendered.read(gid).rgb;
    float3 Y = groundTruth.read(gid).rgb;

    // D-SSIM gradient: conv(K,w) + X*conv(L,w) + Y*conv(M,w)
    float3 dDSSIM_dX = convK + X * convL + Y * convM;

    // L1 gradient: (1-lambda) * sign(X-Y) / (3*N)
    float3 diff = X - Y;
    float3 dL1_dX = (1.0f - LAMBDA_DSSIM) * sign(diff) / (3.0f * numPixels);

    // Combined gradient
    float3 grad = dL1_dX + dDSSIM_dX;

    // Final sanitization before Stage-1 consumption
    if (!isfinite(grad.x) || !isfinite(grad.y) || !isfinite(grad.z)) {
        grad = float3(0.0f);
    } else {
        grad = clamp(grad, float3(-MAX_PIXEL_GRAD_ABS), float3(MAX_PIXEL_GRAD_ABS));
    }

    uint idx = gid.y * width + gid.x;
    uint base = idx * 3;
    pixelGradients[base + 0] = grad.x;
    pixelGradients[base + 1] = grad.y;
    pixelGradients[base + 2] = grad.z;
}

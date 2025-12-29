#include <metal_stdlib>
using namespace metal;

struct Gaussian {
    float3 position;
    float3 scale;
    float4 rotation;
    float opacity;
    float sh[12];
};

struct ProjectedGaussian {
    float2 screenPos;
    float3 conic;
    float depth;
    float opacity;
    float3 color;
    float radius;
    uint tileMinX;
    uint tileMinY;
    uint tileMaxX;
    uint tileMaxY;
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

float3x3 quatToMat(float4 q) {
    float w = q.x, x = q.y, y = q.z, z = q.w;
    return float3x3(
        1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z), 2.0*(x*z + w*y),
        2.0*(x*y + w*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x),
        2.0*(x*z - w*y), 2.0*(y*z + w*x), 1.0 - 2.0*(x*x + y*y)
    );
}

// ============================================================================
// Projection kernel
// ============================================================================

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
    
    float4 worldPos = float4(g.position, 1.0);
    float4 viewPos = uniforms.viewMatrix * worldPos;
    float4 clipPos = uniforms.viewProjectionMatrix * worldPos;
    
    if (clipPos.w <= 0.01) { projected[tid] = proj; return; }
    
    float3 ndc = clipPos.xyz / clipPos.w;
    if (abs(ndc.x) > 1.3 || abs(ndc.y) > 1.3) { projected[tid] = proj; return; }
    
    proj.screenPos = float2(
        (ndc.x * 0.5 + 0.5) * uniforms.screenSize.x,
        (ndc.y * 0.5 + 0.5) * uniforms.screenSize.y
    );
    proj.depth = viewPos.z;
    
    float3 scale = exp(clamp(g.scale, -10.0f, 10.0f));
    float4 q = normalize(g.rotation);
    
    float3x3 R = quatToMat(q);
    float3x3 S = float3x3(scale.x, 0, 0, 0, scale.y, 0, 0, 0, scale.z);
    float3x3 M = R * S;
    float3x3 Sigma3D = M * transpose(M);
    
    float3x3 viewRot = float3x3(
        uniforms.viewMatrix.columns[0].xyz,
        uniforms.viewMatrix.columns[1].xyz,
        uniforms.viewMatrix.columns[2].xyz
    );
    
    float z_cam = viewPos.z;
    if (abs(z_cam) < 0.01) { projected[tid] = proj; return; }
    
    float fx = uniforms.focalLength.x;
    float fy = uniforms.focalLength.y;
    float z2 = z_cam * z_cam;
    
    float2x3 J = float2x3(fx/z_cam, 0, -fx*viewPos.x/z2, 0, fy/z_cam, -fy*viewPos.y/z2);
    float3x3 Sv = viewRot * Sigma3D * transpose(viewRot);
    
    float a = 0, b = 0, c = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            a += J[0][i] * Sv[i][j] * J[0][j];
            b += J[0][i] * Sv[i][j] * J[1][j];
            c += J[1][i] * Sv[i][j] * J[1][j];
        }
    }
    a += 0.3; c += 0.3;
    
    float det = a * c - b * b;
    if (det < 0.0001) { projected[tid] = proj; return; }
    
    float inv_det = 1.0 / det;
    proj.conic = float3(c * inv_det, -b * inv_det, a * inv_det);
    
    float mid = 0.5 * (a + c);
    float l1 = mid + sqrt(max(0.1f, mid*mid - det));
    proj.radius = min(ceil(3.0 * sqrt(l1)), 512.0f);
    
    int2 minB = int2(max(0, int(proj.screenPos.x - proj.radius)), max(0, int(proj.screenPos.y - proj.radius)));
    int2 maxB = int2(min(int(uniforms.screenSize.x)-1, int(proj.screenPos.x + proj.radius)),
                     min(int(uniforms.screenSize.y)-1, int(proj.screenPos.y + proj.radius)));
    
    proj.tileMinX = minB.x / TILE_SIZE;
    proj.tileMinY = minB.y / TILE_SIZE;
    proj.tileMaxX = min(uint(maxB.x / TILE_SIZE), uniforms.numTilesX - 1);
    proj.tileMaxY = min(uint(maxB.y / TILE_SIZE), uniforms.numTilesY - 1);
    
    proj.opacity = 1.0 / (1.0 + exp(-clamp(g.opacity, -10.0f, 10.0f)));
    // SH DC coefficients stored at indices 0,1,2 for R,G,B
    proj.color = max(float3(g.sh[0], g.sh[1], g.sh[2]) * SH_C0 + 0.5f, 0.0f);
    
    projected[tid] = proj;
}

// ============================================================================
// Tile counting
// ============================================================================

kernel void countTilesPerGaussian(
    device const ProjectedGaussian* projected [[buffer(0)]],
    device atomic_uint* tileCounts [[buffer(1)]],
    device atomic_uint* totalPairs [[buffer(2)]],
    constant TiledUniforms& uniforms [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= uniforms.numGaussians) return;
    
    ProjectedGaussian p = projected[tid];
    if (p.radius <= 0 || p.tileMinX > p.tileMaxX) return;
    
    uint num = 0;
    for (uint ty = p.tileMinY; ty <= p.tileMaxY; ty++) {
        for (uint tx = p.tileMinX; tx <= p.tileMaxX; tx++) {
            uint idx = ty * uniforms.numTilesX + tx;
            atomic_fetch_add_explicit(&tileCounts[idx], 1, memory_order_relaxed);
            num++;
        }
    }
    atomic_fetch_add_explicit(totalPairs, num, memory_order_relaxed);
}

// ============================================================================
// Write keys for sorting
// ============================================================================

kernel void writeGaussianKeys(
    device const ProjectedGaussian* projected [[buffer(0)]],
    device ulong* keys [[buffer(1)]],
    device uint* values [[buffer(2)]],
    device atomic_uint* tileWriteOffsets [[buffer(3)]],
    constant TiledUniforms& uniforms [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= uniforms.numGaussians) return;
    
    ProjectedGaussian p = projected[tid];
    if (p.radius <= 0 || p.tileMinX > p.tileMaxX) return;
    
    uint dk = as_type<uint>(p.depth);
    dk = (dk & 0x80000000) ? ~dk : (dk | 0x80000000);
    
    for (uint ty = p.tileMinY; ty <= p.tileMaxY; ty++) {
        for (uint tx = p.tileMinX; tx <= p.tileMaxX; tx++) {
            uint tileIdx = ty * uniforms.numTilesX + tx;
            uint pos = atomic_fetch_add_explicit(&tileWriteOffsets[tileIdx], 1, memory_order_relaxed);
            keys[pos] = (ulong(tileIdx) << 32) | ulong(dk);
            values[pos] = tid;
        }
    }
}

// ============================================================================
// Tiled forward pass
// ============================================================================

kernel void tiledForward(
    device const Gaussian* gaussians [[buffer(0)]],
    device const ProjectedGaussian* projected [[buffer(1)]],
    device const uint* sortedIndices [[buffer(2)]],
    device const TileRange* tileRanges [[buffer(3)]],
    constant TiledUniforms& uniforms [[buffer(4)]],
    device float* transmittance [[buffer(5)]],
    device uint* lastIdx [[buffer(6)]],
    texture2d<float, access::write> output [[texture(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= uint(uniforms.screenSize.x) || gid.y >= uint(uniforms.screenSize.y)) return;
    
    uint tileIdx = (gid.y / TILE_SIZE) * uniforms.numTilesX + (gid.x / TILE_SIZE);
    TileRange range = tileRanges[tileIdx];
    
    float3 color = float3(0);
    float T = 1.0;
    float2 pixelPos = float2(gid) + 0.5;
    
    // Track the last index that actually contributed (not just processed)
    uint lastContributingIdx = range.start;
    uint numContributing = 0;

    for (uint i = range.start; i < range.start + range.count; i++) {
        if (T <= 0.001) break;

        uint gIdx = sortedIndices[i];
        ProjectedGaussian p = projected[gIdx];
        
        if (p.radius <= 0) continue;
        
        float2 d = pixelPos - p.screenPos;
        float power = -0.5 * (p.conic.x * d.x * d.x + 2.0 * p.conic.y * d.x * d.y + p.conic.z * d.y * d.y);
        
        if (power > 0.0 || power < -4.5) continue;
        
        float G = exp(power);
        float alpha = min(p.opacity * G, 0.99f);
        if (alpha < 1.0/255.0) continue;
        
        color += p.color * alpha * T;
        T *= (1.0 - alpha);
        
        // Only update lastContributingIdx for Gaussians that actually contribute
        lastContributingIdx = i;
        numContributing++;
    }
    
    uint pixelIdx = gid.y * uint(uniforms.screenSize.x) + gid.x;
    transmittance[pixelIdx] = T;
    // Store the count of contributing Gaussians instead of just the last index
    // We'll iterate from range.start to range.start + range.count in backward pass
    lastIdx[pixelIdx] = numContributing > 0 ? lastContributingIdx : UINT_MAX;
    output.write(float4(color, 1.0 - T), gid);
}

// ============================================================================
// Tiled backward pass - OPTIMIZED with threadgroup reduction
// ============================================================================

kernel void tiledBackward(
    device const Gaussian* gaussians [[buffer(0)]],
    device GaussianGradients* gradients [[buffer(1)]],
    device const ProjectedGaussian* projected [[buffer(2)]],
    device const uint* sortedIndices [[buffer(3)]],
    device const TileRange* tileRanges [[buffer(4)]],
    constant TiledUniforms& uniforms [[buffer(5)]],
    device const float* transmittance [[buffer(6)]],
    device const uint* lastIdx [[buffer(7)]],
    texture2d<float, access::read> rendered [[texture(0)]],
    texture2d<float, access::read> groundTruth [[texture(1)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    // Shared memory for gradient accumulation within tile
    threadgroup float tg_sh_grads[TILE_SIZE * TILE_SIZE][3];      // SH gradients per thread
    threadgroup float tg_opacity_grads[TILE_SIZE * TILE_SIZE];
    threadgroup float tg_pos_grads[TILE_SIZE * TILE_SIZE][2];
    threadgroup float tg_scale_grads[TILE_SIZE * TILE_SIZE];
    threadgroup uint tg_gaussian_idx[TILE_SIZE * TILE_SIZE];
    
    uint localIdx = tid.y * TILE_SIZE + tid.x;
    uint tileIdx = tgid.y * uniforms.numTilesX + tgid.x;
    TileRange range = tileRanges[tileIdx];
    
    bool validPixel = (gid.x < uint(uniforms.screenSize.x) && gid.y < uint(uniforms.screenSize.y));
    uint pixelIdx = gid.y * uint(uniforms.screenSize.x) + gid.x;
    
    float2 pixelPos = float2(gid) + 0.5;
    float3 dL_dPix = float3(0);
    uint endIdx = 0;
    
    if (validPixel) {
        float4 r_pix = rendered.read(gid);
        float4 gt_pix = groundTruth.read(gid);
        
        // L1 loss gradient
        dL_dPix = float3(
            (r_pix.r > gt_pix.r) ? 1.0 : -1.0,
            (r_pix.g > gt_pix.g) ? 1.0 : -1.0,
            (r_pix.b > gt_pix.b) ? 1.0 : -1.0
        );
        
        endIdx = lastIdx[pixelIdx];
    }
    
    if (range.count == 0) return;
    
    // Process gaussians in batches to reduce atomic contention
    float T = 1.0;
    
    for (uint i = range.start; i < range.start + range.count; i++) {
        // Initialize thread-local gradients
        tg_sh_grads[localIdx][0] = 0;
        tg_sh_grads[localIdx][1] = 0;
        tg_sh_grads[localIdx][2] = 0;
        tg_opacity_grads[localIdx] = 0;
        tg_pos_grads[localIdx][0] = 0;
        tg_pos_grads[localIdx][1] = 0;
        tg_scale_grads[localIdx] = 0;
        
        uint gIdx = sortedIndices[i];
        tg_gaussian_idx[localIdx] = gIdx;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (validPixel && endIdx != UINT_MAX && i <= endIdx && T > 0.001) {
            ProjectedGaussian p = projected[gIdx];
            
            if (p.radius > 0) {
                float2 d = pixelPos - p.screenPos;
                float power = -0.5 * (p.conic.x * d.x * d.x + 2.0 * p.conic.y * d.x * d.y + p.conic.z * d.y * d.y);
                
                if (power <= 0.0 && power >= -4.5) {
                    float G = exp(power);
                    float alpha = min(p.opacity * G, 0.99f);
                    
                    if (alpha >= 1.0/255.0) {
                        float weight = alpha * T;
                        float3 dL_dC = dL_dPix * weight;
                        
                        // Store gradients in threadgroup memory
                        tg_sh_grads[localIdx][0] = dL_dC.r * SH_C0;
                        tg_sh_grads[localIdx][1] = dL_dC.g * SH_C0;
                        tg_sh_grads[localIdx][2] = dL_dC.b * SH_C0;
                        
                        float dL_dAlpha = dot(dL_dPix, p.color * T);
                        float sigmoid_val = p.opacity;
                        float sig_deriv = sigmoid_val * (1.0 - sigmoid_val);
                        tg_opacity_grads[localIdx] = dL_dAlpha * G * sig_deriv;
                        
                        float dP_ddx = -(p.conic.x * d.x + p.conic.y * d.y);
                        float dP_ddy = -(p.conic.y * d.x + p.conic.z * d.y);
                        float dL_dG = dL_dAlpha * sigmoid_val;
                        float dG_dP = G;
                        
                        float inv_z = 1.0 / max(abs(p.depth), 0.1f);
                        float fx = uniforms.focalLength.x;
                        float fy = uniforms.focalLength.y;
                        
                        tg_pos_grads[localIdx][0] = -dL_dG * dG_dP * dP_ddx * inv_z * fx * 0.5;
                        tg_pos_grads[localIdx][1] = -dL_dG * dG_dP * dP_ddy * inv_z * fy * 0.5;
                        
                        float dL_dPower = dL_dG * G;
                        float distSq = d.x * d.x + d.y * d.y;
                        tg_scale_grads[localIdx] = dL_dPower * distSq * 0.001f;
                        
                        T *= (1.0 - alpha);
                    }
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Parallel reduction within threadgroup (only first thread writes)
        if (localIdx == 0) {
            float sum_sh0 = 0, sum_sh1 = 0, sum_sh2 = 0;
            float sum_opacity = 0;
            float sum_pos_x = 0, sum_pos_y = 0;
            float sum_scale = 0;
            
            for (uint j = 0; j < TILE_SIZE * TILE_SIZE; j++) {
                sum_sh0 += tg_sh_grads[j][0];
                sum_sh1 += tg_sh_grads[j][1];
                sum_sh2 += tg_sh_grads[j][2];
                sum_opacity += tg_opacity_grads[j];
                sum_pos_x += tg_pos_grads[j][0];
                sum_pos_y += tg_pos_grads[j][1];
                sum_scale += tg_scale_grads[j];
            }
            
            // Single atomic write per gaussian per tile (instead of per pixel!)
            if (sum_sh0 != 0 || sum_sh1 != 0 || sum_sh2 != 0) {
                atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].sh[0], sum_sh0, memory_order_relaxed);
                atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].sh[1], sum_sh1, memory_order_relaxed);
                atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].sh[2], sum_sh2, memory_order_relaxed);
            }
            if (sum_opacity != 0) {
                atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].opacity, sum_opacity, memory_order_relaxed);
            }
            if (sum_pos_x != 0 || sum_pos_y != 0) {
                atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].position_x, sum_pos_x, memory_order_relaxed);
                atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].position_y, sum_pos_y, memory_order_relaxed);
            }
            if (sum_scale != 0) {
                atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].scale_x, sum_scale, memory_order_relaxed);
                atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].scale_y, sum_scale, memory_order_relaxed);
                atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].scale_z, sum_scale, memory_order_relaxed);
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

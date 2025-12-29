#include <metal_stdlib>
using namespace metal;

struct Gaussian {
    float3 position;
    float3 scale;
    float4 rotation;
    float opacity;
    float sh[12];  // SH layout: [R0,R1,R2,R3, G0,G1,G2,G3, B0,B1,B2,B3]
};

struct ProjectedGaussian {
    float2 screenPos;
    float3 conic;       // (cov_inv_00, cov_inv_01, cov_inv_11)
    float depth;
    float opacity;      // After sigmoid
    float3 color;
    float radius;
    uint tileMinX;
    uint tileMinY;
    uint tileMaxX;
    uint tileMaxY;
    // Store covariance for backward pass
    float3 cov2D;       // (a, b, c) - the 2D covariance before inversion
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
constant float SH_C1 = 0.4886025119029199f;
constant uint TILE_SIZE = 16;

float3x3 quatToMat(float4 q) {
    // q = (w, x, y, z) format - but your struct stores as (x, y, z, w) based on usage
    // Looking at your code: float r = q.x, x = q.y, y = q.z, z = q.w;
    // So q.x = w (scalar), q.y = x, q.z = y, q.w = z
    float w = q.x, x = q.y, y = q.z, z = q.w;
    return float3x3(
        1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z), 2.0*(x*z + w*y),
        2.0*(x*y + w*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x),
        2.0*(x*z - w*y), 2.0*(y*z + w*x), 1.0 - 2.0*(x*x + y*y)
    );
}

float3 evalSHDC(float sh[12], float3 dir) {
    // SH layout: sh[0-3] = R coefficients, sh[4-7] = G, sh[8-11] = B
    // DC term (l=0, m=0) is at index 0, 4, 8
    float3 result = float3(sh[0], sh[4], sh[8]) * SH_C0;
    
    // First order SH (optional, can enable later for better quality)
    // result += float3(sh[1], sh[5], sh[9]) * SH_C1 * dir.y;
    // result += float3(sh[2], sh[6], sh[10]) * SH_C1 * dir.z;
    // result += float3(sh[3], sh[7], sh[11]) * SH_C1 * dir.x;
    
    return max(result + 0.5f, 0.0f);
}

// ============================================================================
// Projection kernel - Fixed version
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
    
    // Check for NaN in input
    if (isnan(g.position.x) || isnan(g.position.y) || isnan(g.position.z)) {
        projected[tid] = proj;
        return;
    }
    
    float4 worldPos = float4(g.position, 1.0);
    float4 viewPos = uniforms.viewMatrix * worldPos;
    float4 clipPos = uniforms.viewProjectionMatrix * worldPos;
    
    // Behind camera check
    if (clipPos.w <= 0.01 || viewPos.z <= 0.01) {
        projected[tid] = proj;
        return;
    }
    
    float3 ndc = clipPos.xyz / clipPos.w;
    
    // Frustum culling with margin
    if (abs(ndc.x) > 1.3 || abs(ndc.y) > 1.3) {
        projected[tid] = proj;
        return;
    }
    
    // Screen position - IMPORTANT: Match the convention used in forward rendering
    // OpenGL/Metal NDC: Y up, Screen: Y down (typically)
    // Using same convention as vertex shader: flip Y
    proj.screenPos = float2(
        (ndc.x * 0.5 + 0.5) * uniforms.screenSize.x,
        (1.0 - (ndc.y * 0.5 + 0.5)) * uniforms.screenSize.y  // Flip Y to match screen coords
    );
    proj.depth = viewPos.z;
    
    // Clamp scale to prevent explosion
    float3 logScale = clamp(g.scale, -10.0f, 10.0f);
    float3 scale = exp(logScale);
    
    // Normalize quaternion
    float4 q = g.rotation;
    float qLen = length(q);
    if (qLen < 0.001) {
        q = float4(1, 0, 0, 0);  // Identity
    } else {
        q = q / qLen;
    }
    
    // Build rotation matrix
    float3x3 R = quatToMat(q);
    
    // Build 3D covariance: Sigma = R * S * S^T * R^T
    float3x3 S = float3x3(scale.x, 0, 0, 0, scale.y, 0, 0, 0, scale.z);
    float3x3 M = R * S;
    float3x3 Sigma3D = M * transpose(M);
    
    // Get view rotation (upper-left 3x3 of view matrix)
    float3x3 viewRot = float3x3(
        uniforms.viewMatrix.columns[0].xyz,
        uniforms.viewMatrix.columns[1].xyz,
        uniforms.viewMatrix.columns[2].xyz
    );
    
    float z_cam = viewPos.z;
    float fx = uniforms.focalLength.x;
    float fy = uniforms.focalLength.y;
    float z2 = z_cam * z_cam;
    
    // Clamp values for numerical stability
    float limx = 1.3f * fx / z_cam;
    float limy = 1.3f * fy / z_cam;
    float txtz = clamp(viewPos.x / z_cam, -limx, limx);
    float tytz = clamp(viewPos.y / z_cam, -limy, limy);
    
    // Jacobian of perspective projection
    // J = | fx/z   0    -fx*x/z^2 |
    //     |  0    fy/z  -fy*y/z^2 |
    float2x3 J = float2x3(
        fx / z_cam, 0, -fx * txtz / z_cam,
        0, fy / z_cam, -fy * tytz / z_cam
    );
    
    // Transform 3D covariance to view space: Sigma_view = W * Sigma * W^T
    float3x3 Sigma_view = viewRot * Sigma3D * transpose(viewRot);
    
    // Project to 2D: Sigma_2D = J * Sigma_view * J^T
    // We only need the 2x2 result
    float a = 0, b = 0, c = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            a += J[0][i] * Sigma_view[i][j] * J[0][j];
            b += J[0][i] * Sigma_view[i][j] * J[1][j];
            c += J[1][i] * Sigma_view[i][j] * J[1][j];
        }
    }
    
    // Add low-pass filter for anti-aliasing (paper section 4)
    a += 0.3;
    c += 0.3;
    
    // Store 2D covariance for backward pass
    proj.cov2D = float3(a, b, c);
    
    // Compute determinant and inverse
    float det = a * c - b * b;
    if (det < 0.0001) {
        projected[tid] = proj;
        return;
    }
    
    float inv_det = 1.0 / det;
    proj.conic = float3(c * inv_det, -b * inv_det, a * inv_det);
    
    // Compute radius from eigenvalues
    float mid = 0.5 * (a + c);
    float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    proj.radius = min(ceil(3.0 * sqrt(max(lambda1, lambda2))), 512.0f);
    
    if (proj.radius <= 0) {
        projected[tid] = proj;
        return;
    }
    
    // Compute tile bounds
    int2 minB = int2(max(0, int(proj.screenPos.x - proj.radius)),
                     max(0, int(proj.screenPos.y - proj.radius)));
    int2 maxB = int2(min(int(uniforms.screenSize.x) - 1, int(proj.screenPos.x + proj.radius)),
                     min(int(uniforms.screenSize.y) - 1, int(proj.screenPos.y + proj.radius)));
    
    if (minB.x > maxB.x || minB.y > maxB.y) {
        proj.radius = 0;
        projected[tid] = proj;
        return;
    }
    
    proj.tileMinX = minB.x / TILE_SIZE;
    proj.tileMinY = minB.y / TILE_SIZE;
    proj.tileMaxX = min(uint(maxB.x / TILE_SIZE), uniforms.numTilesX - 1);
    proj.tileMaxY = min(uint(maxB.y / TILE_SIZE), uniforms.numTilesY - 1);
    
    // Opacity after sigmoid
    float rawOpacity = clamp(g.opacity, -10.0f, 10.0f);
    proj.opacity = 1.0 / (1.0 + exp(-rawOpacity));
    
    // Evaluate SH for color (just DC for now)
    // SH layout: sh[0,4,8] are DC terms for R,G,B
    proj.color = clamp(float3(
        SH_C0 * g.sh[0] + 0.5f,
        SH_C0 * g.sh[4] + 0.5f,
        SH_C0 * g.sh[8] + 0.5f
    ), 0.0f, 1.0f);
    
    projected[tid] = proj;
}

// ============================================================================
// Tile counting - unchanged
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
// Write keys for sorting - unchanged
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
    
    // Convert depth to sortable uint (front-to-back = smaller key first)
    uint dk = as_type<uint>(p.depth);
    // Handle negative floats correctly for sorting
    dk = (dk & 0x80000000) ? ~dk : (dk | 0x80000000);
    
    for (uint ty = p.tileMinY; ty <= p.tileMaxY; ty++) {
        for (uint tx = p.tileMinX; tx <= p.tileMaxX; tx++) {
            uint tileIdx = ty * uniforms.numTilesX + tx;
            uint pos = atomic_fetch_add_explicit(&tileWriteOffsets[tileIdx], 1, memory_order_relaxed);
            // Key: high 32 bits = tile index, low 32 bits = depth
            keys[pos] = (ulong(tileIdx) << 32) | ulong(dk);
            values[pos] = tid;
        }
    }
}

// ============================================================================
// Tiled forward pass - Fixed to store per-pixel accumulated data
// ============================================================================

kernel void tiledForward(
    device const Gaussian* gaussians [[buffer(0)]],
    device const ProjectedGaussian* projected [[buffer(1)]],
    device const uint* sortedIndices [[buffer(2)]],
    device const TileRange* tileRanges [[buffer(3)]],
    constant TiledUniforms& uniforms [[buffer(4)]],
    device float* finalTransmittance [[buffer(5)]],
    device uint* lastContribIdx [[buffer(6)]],
    texture2d<float, access::write> output [[texture(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= uint(uniforms.screenSize.x) || gid.y >= uint(uniforms.screenSize.y)) return;
    
    uint tileX = gid.x / TILE_SIZE;
    uint tileY = gid.y / TILE_SIZE;
    uint tileIdx = tileY * uniforms.numTilesX + tileX;
    TileRange range = tileRanges[tileIdx];
    
    float3 color = float3(0);
    float T = 1.0;  // Accumulated transmittance
    float2 pixelPos = float2(gid) + 0.5;
    
    uint lastIdx = 0;
    bool hasContrib = false;
    
    // Front-to-back blending (already sorted by depth)
    for (uint i = range.start; i < range.start + range.count && T > 0.0001; i++) {
        uint gIdx = sortedIndices[i];
        ProjectedGaussian p = projected[gIdx];
        
        if (p.radius <= 0) continue;
        
        float2 d = pixelPos - p.screenPos;
        
        // Evaluate Gaussian: exp(-0.5 * (ax^2 + 2bxy + cy^2))
        // where conic = (a, b, c) = inverse covariance
        float power = -0.5 * (p.conic.x * d.x * d.x +
                              2.0 * p.conic.y * d.x * d.y +
                              p.conic.z * d.y * d.y);
        
        // Skip if outside Gaussian (power > 0 means d^T * Sigma^-1 * d < 0, which is invalid)
        // or too far in the tail
        if (power > 0.0 || power < -4.5) continue;
        
        float G = exp(power);
        float alpha = min(p.opacity * G, 0.99f);
        
        if (alpha < 1.0 / 255.0) continue;
        
        // Alpha blending: C = sum(c_i * alpha_i * T_i)
        color += p.color * alpha * T;
        
        // Update transmittance: T_{i+1} = T_i * (1 - alpha_i)
        T *= (1.0 - alpha);
        
        lastIdx = i;
        hasContrib = true;
    }
    
    uint pixelIdx = gid.y * uint(uniforms.screenSize.x) + gid.x;
    finalTransmittance[pixelIdx] = T;
    lastContribIdx[pixelIdx] = hasContrib ? lastIdx : UINT_MAX;
    
    // Output accumulated color and alpha
    output.write(float4(color, 1.0 - T), gid);
}

// ============================================================================
// Tiled backward pass - Fixed gradient computation following the paper
// ============================================================================

kernel void tiledBackward(
    device const Gaussian* gaussians [[buffer(0)]],
    device GaussianGradients* gradients [[buffer(1)]],
    device const ProjectedGaussian* projected [[buffer(2)]],
    device const uint* sortedIndices [[buffer(3)]],
    device const TileRange* tileRanges [[buffer(4)]],
    constant TiledUniforms& uniforms [[buffer(5)]],
    device const float* finalTransmittance [[buffer(6)]],
    device const uint* lastContribIdx [[buffer(7)]],
    texture2d<float, access::read> rendered [[texture(0)]],
    texture2d<float, access::read> groundTruth [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= uint(uniforms.screenSize.x) || gid.y >= uint(uniforms.screenSize.y)) return;
    
    uint pixelIdx = gid.y * uint(uniforms.screenSize.x) + gid.x;
    uint lastIdx = lastContribIdx[pixelIdx];
    
    if (lastIdx == UINT_MAX) return;  // No Gaussians contributed to this pixel
    
    uint tileX = gid.x / TILE_SIZE;
    uint tileY = gid.y / TILE_SIZE;
    uint tileIdx = tileY * uniforms.numTilesX + tileX;
    TileRange range = tileRanges[tileIdx];
    
    float2 pixelPos = float2(gid) + 0.5;
    
    // Read rendered and ground truth
    float4 r_pix = rendered.read(gid);
    float4 gt_pix = groundTruth.read(gid);
    
    // L1 loss gradient: dL/dC = sign(rendered - gt)
    float3 dL_dPixelColor = float3(
        (r_pix.r > gt_pix.r) ? 1.0 : ((r_pix.r < gt_pix.r) ? -1.0 : 0.0),
        (r_pix.g > gt_pix.g) ? 1.0 : ((r_pix.g < gt_pix.g) ? -1.0 : 0.0),
        (r_pix.b > gt_pix.b) ? 1.0 : ((r_pix.b < gt_pix.b) ? -1.0 : 0.0)
    );
    
    // Scale by 1/3 since L1 loss averages over channels
    dL_dPixelColor /= 3.0;
    
    // According to paper: we need to traverse back-to-front
    // But we can also go front-to-back and recover T values
    // T_final is stored, we compute T_i by dividing
    
    // First pass: compute all T_i values by going front-to-back
    // T_i = product of (1 - alpha_j) for j < i
    
    float T = 1.0;
    
    // We need to iterate front-to-back to compute T values,
    // then use them for gradient computation
    for (uint i = range.start; i <= lastIdx; i++) {
        uint gIdx = sortedIndices[i];
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
        
        // Current transmittance for this Gaussian
        float T_i = T;
        
        // Weight for this Gaussian's contribution
        float weight = alpha * T_i;
        
        // ========== Color/SH gradients ==========
        // C_pixel = sum(c_i * alpha_i * T_i)
        // dL/dc_i = dL/dC_pixel * alpha_i * T_i
        float3 dL_dColor = dL_dPixelColor * weight;
        
        // SH gradient: c = SH_C0 * sh + 0.5, so dc/dsh = SH_C0
        // dL/dsh = dL/dc * dc/dsh = dL/dc * SH_C0
        float dL_dsh_r = dL_dColor.r * SH_C0;
        float dL_dsh_g = dL_dColor.g * SH_C0;
        float dL_dsh_b = dL_dColor.b * SH_C0;
        
        // ========== Opacity gradient ==========
        // dL/d(alpha_i) = dL/dC_pixel * c_i * T_i + dL/dT_{i+1} * (-T_i)
        // For now, simplified: dL/d(alpha_i) ≈ dot(dL_dPixelColor, c_i) * T_i
        float dL_dAlpha = dot(dL_dPixelColor, p.color) * T_i;
        
        // alpha = opacity * G, where opacity = sigmoid(raw_opacity)
        // d(alpha)/d(raw_opacity) = G * sigmoid' = G * opacity * (1 - opacity)
        float sigmoid_val = p.opacity;
        float sigmoid_deriv = sigmoid_val * (1.0 - sigmoid_val);
        float dL_dRawOpacity = dL_dAlpha * G * sigmoid_deriv;
        
        // ========== Position gradients through screen position ==========
        // dL/d(screenPos) comes from the Gaussian evaluation
        // G = exp(power), power = -0.5 * d^T * Sigma^-1 * d
        // dG/d(screenPos) = G * d(power)/d(screenPos)
        // d(power)/d(d) = -Sigma^-1 * d (since d(power)/d(d) = -0.5 * 2 * Sigma^-1 * d)
        // d(d)/d(screenPos) = -I (since d = pixelPos - screenPos)
        // So: dG/d(screenPos) = G * Sigma^-1 * d
        
        float2 conic_d = float2(
            p.conic.x * d.x + p.conic.y * d.y,  // Sigma^-1 * d, x component
            p.conic.y * d.x + p.conic.z * d.y   // Sigma^-1 * d, y component
        );
        
        // dL/d(screenPos) = dL/d(alpha) * d(alpha)/d(G) * dG/d(screenPos)
        //                 = dL_dAlpha * opacity * G * conic_d
        float2 dL_dScreenPos = dL_dAlpha * sigmoid_val * G * conic_d;
        
        // Now backprop screenPos to world position
        // screenPos = ((ndc.xy * 0.5 + 0.5) * screenSize, with Y flipped
        // ndc = clip.xyz / clip.w
        // clip = VP * worldPos
        
        // For simplicity, approximate position gradient:
        // dL/d(worldPos) ≈ dL/d(screenPos) * d(screenPos)/d(worldPos)
        // d(screenPos)/d(worldPos) involves the Jacobian of projection
        
        // Approximate: position gradient magnitude scales with focal length / depth
        float inv_z = 1.0 / max(p.depth, 0.1f);
        float fx = uniforms.focalLength.x;
        float fy = uniforms.focalLength.y;
        
        // Very rough approximation - proper version needs full Jacobian
        float dL_dPosX = dL_dScreenPos.x * inv_z / (uniforms.screenSize.x * 0.5);
        float dL_dPosY = -dL_dScreenPos.y * inv_z / (uniforms.screenSize.y * 0.5);  // Flip Y back
        float dL_dPosZ = -(dL_dScreenPos.x * d.x + dL_dScreenPos.y * d.y) * inv_z * inv_z * 0.1;
        
        // ========== Scale gradients ==========
        // Scale affects the 2D covariance, which affects the conic
        // This is complex - approximate for now
        // Larger scale = larger 2D covariance = smaller conic values = slower Gaussian falloff
        // dL/d(scale) ≈ dL/d(conic) * d(conic)/d(scale)
        
        // From paper: gradient through covariance is important
        // For now, use a simple approximation based on the power term
        float dL_dPower = dL_dAlpha * sigmoid_val * G;  // dL/dG * dG/dPower
        
        // power = -0.5 * d^T * Sigma^-1 * d
        // Sigma = R * S^2 * R^T, where S = diag(scale)
        // Larger scale -> smaller Sigma^-1 -> less negative power -> larger G
        
        // Approximate: scale gradient proportional to power and current scale
        float distSq = dot(d, d);
        float dL_dScale = dL_dPower * distSq * 0.0001;  // Small factor to prevent instability
        
        // ========== Atomic accumulation ==========
        // SH gradients (DC terms)
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].sh[0], dL_dsh_r, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].sh[4], dL_dsh_g, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].sh[8], dL_dsh_b, memory_order_relaxed);
        
        // Opacity gradient
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].opacity, dL_dRawOpacity, memory_order_relaxed);
        
        // Position gradients
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].position_x, dL_dPosX, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].position_y, dL_dPosY, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].position_z, dL_dPosZ, memory_order_relaxed);
        
        // Scale gradients (uniform for all axes for now)
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].scale_x, dL_dScale, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].scale_y, dL_dScale, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].scale_z, dL_dScale, memory_order_relaxed);
        
        // Update transmittance for next iteration
        T *= (1.0 - alpha);
    }
}

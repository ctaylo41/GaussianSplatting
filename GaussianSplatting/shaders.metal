//
//  mtl_engine.hpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-24.
//

#include <metal_stdlib>
using namespace metal;

// The Gaussian Struct
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

// Uniforms passed to shaders
// Must match C++ Uniforms struct in mtl_engine.hpp exactly
struct Uniforms {
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4x4 viewProjectionMatrix;
    float2 screenSize;
    float2 focalLength;
    float3 cameraPos;
    float _pad;  // Match C++ struct padding
};

// Vertex output structure
struct VertexOut {
    float4 position [[position]];
    float3 color;
    float opacity;
    float2 centerScreenPos;
    float3 conic;
    float2 quadOffset;
};

// Spherical Harmonics constants
constant float SH_C0 = 0.28209479177387814f;
constant float SH_C1 = 0.4886025119029199f;
// MAX_SCALE for viewing higher limit to handle external PLY files
constant float MAX_SCALE = 8.0f; 
// Stricter MAX_SCALE for training to prevent Gaussians from growing too large
constant float MAX_SCALE_TRAIN = 4.0f;
// Keep OFF for root-cause debugging so true gradient amplitudes are visible
constant bool ENABLE_ADAM_GRAD_CLAMP = false;

// Evaluate color from SH coefficients
// Must match training renderer: color = SH_C0 * dc + 0.5 + SH_C1 * (view-dependent terms)
float3 evalSH(float sh[12], float3 dir) {
    // DC component (degree 0)
    float3 dc = float3(sh[0], sh[4], sh[8]);

    // Degree 1 components
    float3 sh1_y = float3(sh[1], sh[5], sh[9]);
    float3 sh1_z = float3(sh[2], sh[6], sh[10]);
    float3 sh1_x = float3(sh[3], sh[7], sh[11]);

    // Official 3DGS formula: SH_C0 * dc + 0.5 + SH_C1 * (view-dependent)
    float3 color = SH_C0 * dc + 0.5f + SH_C1 * (-sh1_y * dir.y + sh1_z * dir.z - sh1_x * dir.x);

    // Only clamp to non-negative (official uses max(result, 0.0f), no upper bound)
    return max(color, 0.0f);
}

// Convert quaternion to rotation matrix
float3x3 quaternionToMatrix(float4 q) {
    // q.x=w, q.y=x, q.z=y, q.w=z
    float w = q.x, x = q.y, y = q.z, z = q.w;
    // Metal float3x3 constructor takes columns
    return float3x3(
        float3(1.0 - 2.0*(y*y + z*z), 2.0*(x*y + w*z), 2.0*(x*z - w*y)), 
        float3(2.0*(x*y - w*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z + w*x)), 
        float3(2.0*(x*z + w*y), 2.0*(y*z - w*x), 1.0 - 2.0*(x*x + y*y)) 
    );
}

// Compute 3D covariance matrix from log scale and rotation
float3x3 computeCovariance3D(float3 logScale, float4 rotation) {
    float3x3 R = quaternionToMatrix(rotation);
    // Apply exp() to log scale
    // Only place exp() is applied
    float3 scale = exp(clamp(logScale, -MAX_SCALE, MAX_SCALE));
    // Diagonal scale matrix Metal column constructor
    float3x3 S = float3x3(
        float3(scale.x, 0, 0),  
        float3(0, scale.y, 0),  
        float3(0, 0, scale.z)   
    );
    // Covariance = R * S^2 * R^T
    float3x3 M = R * S;
    return M * transpose(M);
}

// Project 3D covariance to 2D screen space
float3 computeCovariance2D(float3 mean, float3x3 covariance3D, float4x4 viewMatrix, float2 focalLength) {
    float4 t = viewMatrix * float4(mean, 1.0);
    
    // Using left-hand coordinate system +Z forward, objects in front have positive z
    float z = t.z;
    // Prevent division by zero and behind camera
    if (z < 0.001) z = 0.001;  
    
    // Clamp to avoid numerical issues at edges
    float limx = 1.3 * focalLength.x / z;
    float limy = 1.3 * focalLength.y / z;
    float txtz = clamp(t.x / z, -limx, limx);
    float tytz = clamp(t.y / z, -limy, limy);
    
    // Jacobian of projection perspective projection derivative
    // Maps 3D view space to 2D screen space
    float J00 = focalLength.x / z;
    float J02 = -focalLength.x * txtz / z;
    float J11 = focalLength.y / z;
    float J12 = -focalLength.y * tytz / z;
    
    float3x3 J = float3x3(
        float3(J00, 0, 0),    
        float3(0, J11, 0),    
        float3(J02, J12, 0)   
    );
    
    // View rotation upper-left 3x3
    float3x3 W = float3x3(viewMatrix[0].xyz, viewMatrix[1].xyz, viewMatrix[2].xyz);
    float3x3 T = J * W;
    float3x3 cov = T * covariance3D * transpose(T);
    
    // Low-pass filter to avoid aliasing
    cov[0][0] += 0.3;
    cov[1][1] += 0.3;
    
    // Extract 2D covariance components
    float a = cov[0][0];
    float b = cov[1][0];
    float c = cov[1][1];
    
    return float3(a, b, c);
}

// Compute conic parameters from 2D covariance
float3 computeConic(float3 cov2D) {
    // Inverse of 2D covariance matrix
    float det = cov2D.x * cov2D.z - cov2D.y * cov2D.y;
    // Numerical stability
    if (abs(det) < 1e-6) det = 1e-6;
    float invDet = 1.0 / det;
    return float3(cov2D.z * invDet, -cov2D.y * invDet, cov2D.x * invDet);
}

// Compute radius from 2D covariance eigenvalues
float computeRadius(float3 cov2D) {
    // Eigenvalues of 2x2 covariance matrix
    float det = cov2D.x * cov2D.z - cov2D.y * cov2D.y;
    // Numerical stability
    float mid = 0.5 * (cov2D.x + cov2D.z);
    // Ensure positive eigenvalues
    float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    // Radius = 3 * sqrt(largest eigenvalue)
    return min(ceil(3.0 * sqrt(max(lambda1, lambda2))), 64.0f);
}

// Vertex Shader
vertex VertexOut vertexShader(
    uint vertexID [[vertex_id]],
    uint instanceID [[instance_id]],
    constant Gaussian* gaussians [[buffer(0)]],
    constant Uniforms& uniforms [[buffer(1)]],
    constant uint32_t* indices [[buffer(2)]])
{
    // Fetch Gaussian for this instance
    uint gaussianIdx = indices[instanceID];
    Gaussian g = gaussians[gaussianIdx];
    
    VertexOut out;
    
    // Skip invalid Gaussians
    if (isnan(g.position.x) || isnan(g.position.y) || isnan(g.position.z) ||
        abs(g.position.x) > 1e6 || abs(g.position.y) > 1e6 || abs(g.position.z) > 1e6) {
        out.position = float4(0, 0, -1, 1);
        out.opacity = 0;
        return out;
    }
    
    // Compute clip space position
    float4 clipPos = uniforms.viewProjectionMatrix * float4(g.position, 1.0);
    
    // Behind camera check
    if (clipPos.w <= 0.0) {
        out.position = float4(0, 0, -1, 1);
        out.opacity = 0;
        return out;
    }
    
    // Compute 3D covariance from scale and rotation
    float3x3 cov3D = computeCovariance3D(g.scale, g.rotation);
    
    // Project to 2D screen space
    float3 cov2D = computeCovariance2D(g.position, cov3D, uniforms.viewMatrix, uniforms.focalLength);
    
    // Compute radius from eigenvalues of 2D covariance
    float det = cov2D.x * cov2D.z - cov2D.y * cov2D.y;
    float radius;
    float3 conic;
    
    // Add numerical stability - ensure positive diagonal with minimum variance
    // The low-pass filter (0.3) should already be applied in computeCovariance2D
     // Minimum variance 
    float a = max(cov2D.x, 0.3f);  
    float b = cov2D.y;
     // Minimum variance
    float c = max(cov2D.z, 0.3f); 
    
    // Clamp off-diagonal to ensure positive definiteness
    float maxB = sqrt(a * c) * 0.99f;
    b = clamp(b, -maxB, maxB);
    
    det = a * c - b * b;
    
    // Check for valid covariance
    if (det > 0.0001) {
        // Valid covariance compute radius from eigenvalues
        float mid = 0.5 * (a + c);
        float disc = max(0.0f, mid * mid - det);
        // Largest eigenvalue
        float lambda = mid + sqrt(disc);  
        radius = ceil(3.0 * sqrt(max(lambda, 0.1f)));
        // Allow larger radius for external PLYs
        radius = clamp(radius, 1.0f, 512.0f);  
        
        // Compute conic inverse of 2D covariance
        float invDet = 1.0 / det;
        conic = float3(c * invDet, -b * invDet, a * invDet);
        
        // Safety check conic magnitude is reasonable for the radius
        float expectedConic = 9.0f / (radius * radius);
         // Average diagonal
        float actualConic = 0.5f * (conic.x + conic.z); 
        
        // If actual conic is less than 1% of expected, rescale
        if (actualConic < expectedConic * 0.01f) {
            // Conic is way too small rescale to expected value
            float scale = expectedConic / max(actualConic, 1e-10f);
            // Partial correction to avoid over-adjustment
            conic *= sqrt(scale); 
        }
    } else {
        // Fallback to circular Gaussian
        radius = 15.0;
        conic = float3(0.04, 0.0, 0.04);
    }
    
    // Quad corners in normalized [-1,1] space
    float2 quadOffsets[4] = {
        float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1)
    };
    
    // Compute final vertex position with offset
    float2 quadOffset = quadOffsets[vertexID];
    float2 centerNDC = clipPos.xy / clipPos.w;
    
    // Offset in pixels from center
    float2 offsetPixels = quadOffset * radius;
    
    // Convert pixel offset to NDC offset
    // Note NDC y goes up, but screen y goes down don't flip here, it's handled in fragment
    float2 offsetNDC = offsetPixels / (uniforms.screenSize * 0.5);
    
    // Final position in clip space
    out.position = float4((centerNDC + offsetNDC) * clipPos.w, clipPos.z, clipPos.w);
    
    // Evaluate SH for view-dependent color
    float3 viewDir = normalize(g.position - uniforms.cameraPos);
    out.color = evalSH(g.sh, viewDir);
    
    // Apply sigmoid to raw opacity this is the ONLY place sigmoid() is applied
    out.opacity = 1.0 / (1.0 + exp(-clamp(g.opacity, -8.0f, 8.0f)));
    
    // Center screen position for debugging, not used in current frag shader
    out.centerScreenPos = float2(
        (centerNDC.x * 0.5 + 0.5) * uniforms.screenSize.x,
        (centerNDC.y * 0.5 + 0.5) * uniforms.screenSize.y
    );
    
    out.conic = conic;
    // Pass pixel offset directly fragment shader uses this for Gaussian calculation
    out.quadOffset = offsetPixels;
    
    return out;
}

// Fragment Shader
fragment float4 fragmentShader(VertexOut in [[stage_in]]) {
    float2 d = in.quadOffset;
    
    // Use proper conic-based Gaussian evaluation matches training
    float power = -0.5 * (in.conic.x * d.x * d.x + 
                          2.0 * in.conic.y * d.x * d.y + 
                          in.conic.z * d.y * d.y);
    
    // Skip if power is positive outside ellipse or too negative
    if (power > 0.0 || power < -4.5) discard_fragment();
    
    // Gaussian weight
    float G = exp(power);
    float alpha = in.opacity * G;
    
    // Early discard for very low alpha
    if (alpha < 1.0 / 255.0) discard_fragment();
    
    // Pre-multiplied alpha output for blending
    return float4(in.color * alpha, alpha);
}

// L1 Loss Computation
kernel void computeL1Loss(
    texture2d<float, access::read> rendered [[texture(0)]],
    texture2d<float, access::read> groundTruth [[texture(1)]],
    device float* losses [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    // Bounds check
    if (gid.x >= rendered.get_width() || gid.y >= rendered.get_height()) return;
    
    // Read pixels
    float4 r = rendered.read(gid);
    float4 gt = groundTruth.read(gid);
    
    // Compute L1 loss per pixel
    // Average over RGB channels
    float l1 = (abs(r.r - gt.r) + abs(r.g - gt.g) + abs(r.b - gt.b)) / 3.0;
    
    // Store per-pixel loss
    uint idx = gid.y * rendered.get_width() + gid.x;
    losses[idx] = l1;
}

// Reduce Loss Kernel
kernel void reduceLoss(
    device float* losses [[buffer(0)]],
    device float* totalLoss [[buffer(1)]],
    constant uint& pixelCount [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threadsPerThreadgroup [[threads_per_threadgroup]])
{
    // Single-threadgroup reduction.
    // Intentionally robust to accidental over-dispatch if multiple threadgroups are launched,
    // they will redundantly compute the same sum and write the same value.
    if (threadsPerThreadgroup > 256) return;

    // Avoids atomic ops on a shared float buffer (which is undefined / driver-dependent).
    threadgroup float partial[256];

    float sum = 0.0f;
    for (uint i = tid; i < pixelCount; i += threadsPerThreadgroup) {
        sum += losses[i];
    }

    // Store partial and reduce
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = (threadsPerThreadgroup >> 1); stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        *totalLoss = partial[0];
    }
}

// SSIM Loss Computation
// Computes local SSIM over 11x11 windows with Gaussian weighting
constant float SSIM_C1 = 0.01f * 0.01f;  
constant float SSIM_C2 = 0.03f * 0.03f;  
constant int SSIM_WINDOW_SIZE = 11;
constant int SSIM_WINDOW_RADIUS = 5;

// Compute SSIM per pixel per-channel: R, G, B separately, then average D-SSIM
// Matches official 3DGS which computes SSIM per-channel
kernel void computeSSIM(
    texture2d<float, access::read> rendered [[texture(0)]],
    texture2d<float, access::read> groundTruth [[texture(1)]],
    device float* ssimMap [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    // Bounds check
    uint width = rendered.get_width();
    uint height = rendered.get_height();

    if (gid.x >= width || gid.y >= height) return;

    // Per-channel accumulators (R, G, B)
    float3 mu_x = float3(0);
    float3 mu_y = float3(0);
    float weight_sum = 0.0f;

    // Gaussian kernel with sigma = 1.5
    float sigma = 1.5f;
    float two_sigma_sq = 2.0f * sigma * sigma;

    // First pass: per-channel means
    for (int dy = -SSIM_WINDOW_RADIUS; dy <= SSIM_WINDOW_RADIUS; dy++) {
        for (int dx = -SSIM_WINDOW_RADIUS; dx <= SSIM_WINDOW_RADIUS; dx++) {
            int px = clamp(int(gid.x) + dx, 0, int(width) - 1);
            int py = clamp(int(gid.y) + dy, 0, int(height) - 1);

            float dist_sq = float(dx * dx + dy * dy);
            float w = exp(-dist_sq / two_sigma_sq);
            weight_sum += w;

            float4 r = rendered.read(uint2(px, py));
            float4 gt = groundTruth.read(uint2(px, py));

            mu_x += w * r.rgb;
            mu_y += w * gt.rgb;
        }
    }

    mu_x /= weight_sum;
    mu_y /= weight_sum;

    // Second pass per-channel variances and covariance
    float3 sigma_x_sq = float3(0);
    float3 sigma_y_sq = float3(0);
    float3 sigma_xy = float3(0);
    weight_sum = 0.0f;

    for (int dy = -SSIM_WINDOW_RADIUS; dy <= SSIM_WINDOW_RADIUS; dy++) {
        for (int dx = -SSIM_WINDOW_RADIUS; dx <= SSIM_WINDOW_RADIUS; dx++) {
            int px = clamp(int(gid.x) + dx, 0, int(width) - 1);
            int py = clamp(int(gid.y) + dy, 0, int(height) - 1);

            float dist_sq = float(dx * dx + dy * dy);
            float w = exp(-dist_sq / two_sigma_sq);
            weight_sum += w;

            float4 r = rendered.read(uint2(px, py));
            float4 gt = groundTruth.read(uint2(px, py));

            float3 dx_val = r.rgb - mu_x;
            float3 dy_val = gt.rgb - mu_y;

            sigma_x_sq += w * dx_val * dx_val;
            sigma_y_sq += w * dy_val * dy_val;
            sigma_xy += w * dx_val * dy_val;
        }
    }

    sigma_x_sq /= weight_sum;
    sigma_y_sq /= weight_sum;
    sigma_xy /= weight_sum;

    // Compute per-channel SSIM
    float3 numerator = (2.0f * mu_x * mu_y + SSIM_C1) * (2.0f * sigma_xy + SSIM_C2);
    float3 denominator = (mu_x * mu_x + mu_y * mu_y + SSIM_C1) * (sigma_x_sq + sigma_y_sq + SSIM_C2);
    float3 ssim = numerator / denominator;

    // Per-channel D-SSIM = (1 - SSIM) / 2, then average across channels
    float3 dssim_per_channel = clamp((1.0f - ssim) / 2.0f, 0.0f, 1.0f);
    float dssim = (dssim_per_channel.x + dssim_per_channel.y + dssim_per_channel.z) / 3.0f;

    uint idx = gid.y * width + gid.x;
    ssimMap[idx] = dssim;
}

// Combined loss 0.8 * L1 + 0.2 * D-SSIM
// Compute combined loss per pixel
kernel void computeCombinedLoss(
    texture2d<float, access::read> rendered [[texture(0)]],
    texture2d<float, access::read> groundTruth [[texture(1)]],
    device float* l1Losses [[buffer(0)]],
    device float* ssimLosses [[buffer(1)]],
    device float* combinedLosses [[buffer(2)]],
    constant float& lambda_dssim [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    // Bounds check
    uint width = rendered.get_width();
    uint height = rendered.get_height();
    
    if (gid.x >= width || gid.y >= height) return;
    
    uint idx = gid.y * width + gid.x;
    
    // Read pre-computed losses
    float l1 = l1Losses[idx];
    float dssim = ssimLosses[idx];
    
    // Combined (1 - lambda) * L1 + lambda * D-SSIM
    // Default lambda = 0.2 to match official 3DGS
    combinedLosses[idx] = (1.0f - lambda_dssim) * l1 + lambda_dssim * dssim;
}

// Final Gaussian gradients plain float
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

// Adam optimizer with safeguards
// Using float* instead of float3* to avoid Metal z-component corruption bug
kernel void adamStep(
    device Gaussian* gaussians [[buffer(0)]],
    device const GaussianGradients* gradients [[buffer(1)]],
    device float* m_position [[buffer(2)]],  
    device float* m_scale [[buffer(3)]],      
    device float4* m_rotation [[buffer(4)]],
    device float* m_opacity [[buffer(5)]],
    device float* m_sh [[buffer(6)]],
    device float* v_position [[buffer(7)]],   
    device float* v_scale [[buffer(8)]],      
    device float4* v_rotation [[buffer(9)]],
    device float* v_opacity [[buffer(10)]],
    device float* v_sh [[buffer(11)]],
    constant float* lrs [[buffer(12)]],
    constant float& beta1 [[buffer(13)]],
    constant float& beta2 [[buffer(14)]],
    constant float& epsilon [[buffer(15)]],
    constant uint2& params [[buffer(16)]],
    device float* debugOut [[buffer(17)]], 
    uint tid [[thread_position_in_grid]])
{
    // Unpack parameters
    uint t = params.x;
    uint numGaussians = params.y;
    
    if (tid >= numGaussians) return;
    
    // Fetch gradients plain reads preprocessBackward writes non-atomic
    float g_position_x = gradients[tid].position_x;
    float g_position_y = gradients[tid].position_y;
    float g_position_z = gradients[tid].position_z;
    float g_opacity    = gradients[tid].opacity;
    float g_scale_x    = gradients[tid].scale_x;
    float g_scale_y    = gradients[tid].scale_y;
    float g_scale_z    = gradients[tid].scale_z;
    float4 g_rotation  = float4(
        gradients[tid].rotation_x,
        gradients[tid].rotation_y,
        gradients[tid].rotation_z,
        gradients[tid].rotation_w
    );
    float g_sh[12];
    for (int i = 0; i < 12; i++) {
        g_sh[i] = gradients[tid].sh[i];
    }
    
    // Skip if any gradient is NaN or inf including rotation and SH
    if (!isfinite(g_position_x) || !isfinite(g_position_y) || !isfinite(g_position_z) ||
        !isfinite(g_opacity) || !isfinite(g_scale_x) || !isfinite(g_scale_y) || !isfinite(g_scale_z) ||
        !isfinite(g_rotation.x) || !isfinite(g_rotation.y) || !isfinite(g_rotation.z) || !isfinite(g_rotation.w)) {
        return;
    }
    // Also skip if any SH gradient is non-finite
    for (int i = 0; i < 12; i++) {
        if (!isfinite(g_sh[i])) return;
    }

    if (ENABLE_ADAM_GRAD_CLAMP) {
        // Optional safety clipping for production stabilization
        g_position_x = clamp(g_position_x, -10.0f, 10.0f);
        g_position_y = clamp(g_position_y, -10.0f, 10.0f);
        g_position_z = clamp(g_position_z, -10.0f, 10.0f);
        g_scale_x = clamp(g_scale_x, -10.0f, 10.0f);
        g_scale_y = clamp(g_scale_y, -10.0f, 10.0f);
        g_scale_z = clamp(g_scale_z, -10.0f, 10.0f);
        g_opacity = clamp(g_opacity, -10.0f, 10.0f);
        g_rotation = clamp(g_rotation, float4(-10.0f), float4(10.0f));
        for (int i = 0; i < 12; i++) {
            g_sh[i] = clamp(g_sh[i], -10.0f, 10.0f);
        }
    }
    
    // Skip corrupted Gaussians check position, scale, and SH
    if (isnan(gaussians[tid].position.x) || isinf(gaussians[tid].position.x) ||
        abs(gaussians[tid].position.x) > 1e6) {
        return;
    }

    // SH coefficients are more prone to corruption due to their larger magnitude and lack of clamping in the training renderer, so we sanitize them before applying updates.
    for (int i = 0; i < 12; i++) {
        float shVal = gaussians[tid].sh[i];
        if (!isfinite(shVal) || fabs(shVal) > 1.0e6f) {
            gaussians[tid].sh[i] = 0.0f;
            m_sh[tid * 12 + i] = 0.0f;
            v_sh[tid * 12 + i] = 0.0f;
        }
    }
    
    // Bias correction terms
    float bc1 = 1.0 - pow(beta1, float(t));
    float bc2 = 1.0 - pow(beta2, float(t));

    // Position update
    // Using manual indexing to avoid float3 z-component corruption
    {
        float3 grad = float3(g_position_x, g_position_y, g_position_z);
        
        // Read manually from float* buffer, sanitize corrupted moments
        float3 m_old = float3(m_position[tid * 3 + 0], m_position[tid * 3 + 1], m_position[tid * 3 + 2]);
        float3 v_old = float3(v_position[tid * 3 + 0], v_position[tid * 3 + 1], v_position[tid * 3 + 2]);
        if (!isfinite(m_old.x)) m_old.x = 0; if (!isfinite(m_old.y)) m_old.y = 0; if (!isfinite(m_old.z)) m_old.z = 0;
        if (!isfinite(v_old.x)) v_old.x = 0; if (!isfinite(v_old.y)) v_old.y = 0; if (!isfinite(v_old.z)) v_old.z = 0;

        // Adam moment updates
        float3 m = beta1 * m_old + (1.0 - beta1) * grad;
        float3 v = beta2 * v_old + (1.0 - beta2) * grad * grad;
        
        // Write manually to float* buffer
        m_position[tid * 3 + 0] = m.x;
        m_position[tid * 3 + 1] = m.y;
        m_position[tid * 3 + 2] = m.z;
        v_position[tid * 3 + 0] = v.x;
        v_position[tid * 3 + 1] = v.y;
        v_position[tid * 3 + 2] = v.z;
        
        // Compute bias-corrected estimates
        float3 m_hat = m / bc1;
        float3 v_hat = v / bc2;
        float3 update = lrs[0] * m_hat / (sqrt(v_hat) + epsilon);
        
        // Apply update
        // Adam + LR decay + gradient clipping naturally limit step sizes)
        float3 newPos = gaussians[tid].position - update;
        
        // Sanity check
        if (!isnan(newPos.x) && !isnan(newPos.y) && !isnan(newPos.z) &&
            abs(newPos.x) < 1e6 && abs(newPos.y) < 1e6 && abs(newPos.z) < 1e6) {
            gaussians[tid].position = newPos;
        }
    }
    
    // Scale update stays in log space
    // Use stricter max_scale_train during training to prevent elongation
    // Using manual indexing to avoid float3 z-component corruption
    {
        float3 grad = float3(g_scale_x, g_scale_y, g_scale_z);
        
        // Read manually from float* buffer, sanitize corrupted moments
        float3 m_old = float3(m_scale[tid * 3 + 0], m_scale[tid * 3 + 1], m_scale[tid * 3 + 2]);
        float3 v_old = float3(v_scale[tid * 3 + 0], v_scale[tid * 3 + 1], v_scale[tid * 3 + 2]);
        if (!isfinite(m_old.x)) m_old.x = 0; if (!isfinite(m_old.y)) m_old.y = 0; if (!isfinite(m_old.z)) m_old.z = 0;
        if (!isfinite(v_old.x)) v_old.x = 0; if (!isfinite(v_old.y)) v_old.y = 0; if (!isfinite(v_old.z)) v_old.z = 0;

        // Adam moment updates
        float3 m = beta1 * m_old + (1.0 - beta1) * grad;
        float3 v = beta2 * v_old + (1.0 - beta2) * grad * grad;
        
        // Write manually to float* buffer
        m_scale[tid * 3 + 0] = m.x;
        m_scale[tid * 3 + 1] = m.y;
        m_scale[tid * 3 + 2] = m.z;
        v_scale[tid * 3 + 0] = v.x;
        v_scale[tid * 3 + 1] = v.y;
        v_scale[tid * 3 + 2] = v.z;
        
        // Compute bias-corrected estimates
        float3 m_hat = m / bc1;
        float3 v_hat = v / bc2;
        float3 newScale = gaussians[tid].scale - lrs[1] * m_hat / (sqrt(v_hat) + epsilon);
        if (isfinite(newScale.x) && isfinite(newScale.y) && isfinite(newScale.z)) {
            // lrs[6] optionally supplies an upper cap derived from world-space pruning
            // (e.g. log(0.1*sceneExtent)). Keep it within [-MAX_SCALE_TRAIN, MAX_SCALE_TRAIN].
            float maxLog = lrs[6];
            maxLog = min(maxLog, MAX_SCALE_TRAIN);
            maxLog = max(maxLog, -MAX_SCALE_TRAIN);
            gaussians[tid].scale = clamp(newScale, -MAX_SCALE_TRAIN, maxLog);
        }
    }
    
    // Rotation update
    {
        float4 grad = g_rotation;
        // Sanitize corrupted moments
        float4 m_old_r = m_rotation[tid];
        float4 v_old_r = v_rotation[tid];
        if (!isfinite(m_old_r.x)) m_old_r.x = 0; if (!isfinite(m_old_r.y)) m_old_r.y = 0;
        if (!isfinite(m_old_r.z)) m_old_r.z = 0; if (!isfinite(m_old_r.w)) m_old_r.w = 0;
        if (!isfinite(v_old_r.x)) v_old_r.x = 0; if (!isfinite(v_old_r.y)) v_old_r.y = 0;
        if (!isfinite(v_old_r.z)) v_old_r.z = 0; if (!isfinite(v_old_r.w)) v_old_r.w = 0;
        // Adam moment updates
        float4 m = beta1 * m_old_r + (1.0 - beta1) * grad;
        float4 v = beta2 * v_old_r + (1.0 - beta2) * grad * grad;
        m_rotation[tid] = m;
        v_rotation[tid] = v;

        // Compute bias-corrected estimates
        float4 m_hat = m / bc1;
        float4 v_hat = v / bc2;
        // Update rotation
        float4 newRot = gaussians[tid].rotation - lrs[2] * m_hat / (sqrt(v_hat) + epsilon);
        float rotLen = length(newRot);
        if (isfinite(rotLen) && rotLen > 0.001) {
            gaussians[tid].rotation = newRot / rotLen;
        }
    }
    
    // Opacity update stays in raw space
    {
        float grad = g_opacity;

        // Sanitize corrupted moments
        float m_old_o = m_opacity[tid];
        float v_old_o = v_opacity[tid];
        if (!isfinite(m_old_o)) m_old_o = 0;
        if (!isfinite(v_old_o)) v_old_o = 0;
        // Adam moment updates
        float m = beta1 * m_old_o + (1.0 - beta1) * grad;
        float v = beta2 * v_old_o + (1.0 - beta2) * grad * grad;
        // Write back
        m_opacity[tid] = m;
        v_opacity[tid] = v;

        // Compute bias-corrected estimates
        float m_hat = m / bc1;
        float v_hat = v / bc2;
        float newOp = gaussians[tid].opacity - lrs[3] * m_hat / (sqrt(v_hat) + epsilon);
        if (isfinite(newOp)) {
            gaussians[tid].opacity = clamp(newOp, -8.0f, 8.0f);
        }
    }
    
    // SH update DC terms and degree-1 terms 
    // Official 3DGS uses lr/20 for degree>=1 terms and activates them at iter 1000
    for (int i = 0; i < 12; i++) {
        // DC indices are 0, 4, 8 rest are degree-1
        bool isDC = (i == 0 || i == 4 || i == 8);
        float lr = isDC ? lrs[4] : lrs[5];

        // Skip degree-1 terms when not yet active 
        // Also keep momentum zeroed to prevent stale accumulation
        if (lr == 0.0f) {
            m_sh[tid * 12 + i] = 0;
            v_sh[tid * 12 + i] = 0;
            continue;
        }

        float grad = g_sh[i];

        uint idx = tid * 12 + i;

        // Sanitize corrupted moments
        float m_old_sh = m_sh[idx];
        float v_old_sh = v_sh[idx];
        if (!isfinite(m_old_sh)) m_old_sh = 0;
        if (!isfinite(v_old_sh)) v_old_sh = 0;
        // Adam moment updates
        float m = beta1 * m_old_sh + (1.0 - beta1) * grad;
        float v = beta2 * v_old_sh + (1.0 - beta2) * grad * grad;
        m_sh[idx] = m;
        v_sh[idx] = v;

        // Compute bias-corrected estimates
        float m_hat = m / bc1;
        float v_hat = v / bc2;

        float newSH = gaussians[tid].sh[i] - lr * m_hat / (sqrt(v_hat) + epsilon);

        if (!isnan(newSH) && !isinf(newSH)) {
            gaussians[tid].sh[i] = newSH;
        }
    }
}

// Blit Shader
struct BlitVertexOut {
    float4 position [[position]];
    float2 texCoord;
};

// Full-screen triangle vertex shader
vertex BlitVertexOut blitVertexShader(uint vertexID [[vertex_id]]) {
    BlitVertexOut out;

    // Generate full-screen triangle vertices
    float2 positions[3] = {
        float2(-1, -1),
        float2( 3, -1),
        float2(-1,  3)
    };

    float2 texCoords[3] = {
        float2(0, 1), 
        float2(2, 1),
        float2(0, -1)
    };

    out.position = float4(positions[vertexID], 0, 1);
    out.texCoord = texCoords[vertexID];

    return out;
}

// Fragment shader to sample and output the texture
fragment float4 blitFragmentShader(BlitVertexOut in [[stage_in]],
                                    texture2d<float> sourceTexture [[texture(0)]]) {
    constexpr sampler s(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    float4 color = sourceTexture.sample(s, in.texCoord);
    // Clamp to [0,1] for output to RGBA8 drawable
    return clamp(color, 0.0, 1.0);
}

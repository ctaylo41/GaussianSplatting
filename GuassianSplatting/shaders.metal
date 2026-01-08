//
//  mtl_engine.hpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-24.
//

#include <metal_stdlib>
using namespace metal;

// CRITICAL Must match C++ simd struct layout exactly  
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
struct Uniforms {
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4x4 viewProjectionMatrix;
    float2 screenSize;
    float2 focalLength;
    float3 cameraPos;       
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
// Training uses a stricter limit in tiled_shaders.metal
 // Log scale range -8 to 8 for viewing compatibility
constant float MAX_SCALE = 8.0f; 
// Stricter MAX_SCALE for training to prevent Gaussians from growing too large
// MAX_SCALE = 4.0 gives exp(4) ≈ 54.6 world units max, allows initial scales like -2.9
constant float MAX_SCALE_TRAIN = 4.0f;

// Evaluate Spherical Harmonics (only DC term used)
float3 evalSH(float sh[12], float3 dir) {
    float3 result = float3(sh[0], sh[4], sh[8]) * SH_C0 + 0.5f;
    return clamp(result, 0.0f, 1.0f);
}

// Convert quaternion to rotation matrix
float3x3 quaternionToMatrix(float4 q) {
    // q.x=w, q.y=x, q.z=y, q.w=z
    float w = q.x, x = q.y, y = q.z, z = q.w;
    // Metal float3x3 constructor takes COLUMNS
    return float3x3(
        float3(1.0 - 2.0*(y*y + z*z), 2.0*(x*y + w*z), 2.0*(x*z - w*y)), 
        float3(2.0*(x*y - w*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z + w*x)), 
        float3(2.0*(x*z + w*y), 2.0*(y*z - w*x), 1.0 - 2.0*(x*x + y*y)) 
    );
}

// Compute 3D covariance matrix from log scale and rotation
float3x3 computeCovariance3D(float3 logScale, float4 rotation) {
    float3x3 R = quaternionToMatrix(rotation);
    // Apply exp() to log scale this is the ONLY place exp() is applied
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
    // Maps 3D view space -> 2D screen space
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
        // At edge (d=radius), power should be ~-4.5
        // power = -0.5 * conic_diag * radius^2 = -4.5 => conic_diag = 9/radius^2
        // If conic is too small (huge covariance), scale it up
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
        // Conic for radius 15: at d=15, want power = -4.5
        // power = -0.5 * conic * d^2 = -0.5 * conic * 225 = -4.5
        // conic = 9 / 225 = 0.04
        conic = float3(0.04, 0.0, 0.04);
    }
    
    // Quad corners in normalized [-1,1] space
    // vertexID 0: bottom-left, 1: bottom-right, 2: top-left, 3: top-right
    float2 quadOffsets[4] = {
        float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1)
    };
    
    // Compute final vertex position with offset
    float2 quadOffset = quadOffsets[vertexID];  // This is in [-1, 1] range
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
    
    // Apply sigmoid to raw opacity - this is the ONLY place sigmoid() is applied
    out.opacity = 1.0 / (1.0 + exp(-clamp(g.opacity, -8.0f, 8.0f)));
    
    // Center screen position (for debugging, not used in current frag shader)
    out.centerScreenPos = float2(
        (centerNDC.x * 0.5 + 0.5) * uniforms.screenSize.x,
        (centerNDC.y * 0.5 + 0.5) * uniforms.screenSize.y  // Don't flip here
    );
    
    out.conic = conic;
    // Pass pixel offset directly - fragment shader uses this for Gaussian calculation
    out.quadOffset = offsetPixels;
    
    return out;
}

// Fragment Shader
fragment float4 fragmentShader(VertexOut in [[stage_in]]) {
    float2 d = in.quadOffset;
    
    // Use proper conic-based Gaussian evaluation matches training
    // power = -0.5 * (conic.x * d.x^2 + 2 * conic.y * d.x * d.y + conic.z * d.y^2)
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
    device atomic_float* totalLoss [[buffer(1)]],
    constant uint& pixelCount [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint threads [[threads_per_grid]])
{
    // Parallel reduction
    float sum = 0.0;
    for (uint i = tid; i < pixelCount; i += threads) {
        sum += losses[i];
    }
    atomic_fetch_add_explicit(totalLoss, sum, memory_order_relaxed);
}

// SSIM Loss Computation
// Computes local SSIM over 11x11 windows with Gaussian weighting
// SSIM = (2*mu_x*mu_y + C1)(2*sigma_xy + C2) / ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2))
// D-SSIM = (1 - SSIM) / 2

// (0.01 * L)^2 where L=1 for normalized images
constant float SSIM_C1 = 0.01f * 0.01f;  
// (0.03 * L)^2
constant float SSIM_C2 = 0.03f * 0.03f;  
constant int SSIM_WINDOW_SIZE = 11;
constant int SSIM_WINDOW_RADIUS = 5;

// Precomputed Gaussian weights for 11x11 window (sigma=1.5)
// These are separable: weight[x][y] = gauss1d[x] * gauss1d[y]
constant float SSIM_GAUSS_1D[11] = {
    // Actually use proper Gaussian
    0.0113437f, 0.0838195f, 0.0838195f, 0.000335463f, 0.0f,  
    // Placeholder we compute inline
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f  
};

// Compute SSIM per pixel
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
    
    // Compute Gaussian-weighted local statistics
    float mu_x = 0.0f, mu_y = 0.0f;
    float sigma_x_sq = 0.0f, sigma_y_sq = 0.0f, sigma_xy = 0.0f;
    float weight_sum = 0.0f;
    
    // Gaussian kernel with sigma = 1.5
    float sigma = 1.5f;
    float two_sigma_sq = 2.0f * sigma * sigma;
    
    // First pass for means
    for (int dy = -SSIM_WINDOW_RADIUS; dy <= SSIM_WINDOW_RADIUS; dy++) {
        for (int dx = -SSIM_WINDOW_RADIUS; dx <= SSIM_WINDOW_RADIUS; dx++) {
            int px = int(gid.x) + dx;
            int py = int(gid.y) + dy;
            
            // Clamp to image boundaries
            px = clamp(px, 0, int(width) - 1);
            py = clamp(py, 0, int(height) - 1);
            
            // Gaussian weight
            float dist_sq = float(dx * dx + dy * dy);
            float w = exp(-dist_sq / two_sigma_sq);
            weight_sum += w;
            
            // Read pixels (convert to grayscale luminance for SSIM)
            float4 r = rendered.read(uint2(px, py));
            float4 gt = groundTruth.read(uint2(px, py));
            
            // Use luminance or average RGB
            float x_val = (r.r + r.g + r.b) / 3.0f;
            float y_val = (gt.r + gt.g + gt.b) / 3.0f;
            
            mu_x += w * x_val;
            mu_y += w * y_val;
        }
    }
    
    // Normalize means
    mu_x /= weight_sum;
    mu_y /= weight_sum;
    
    // Second pass for variance and covariance
    weight_sum = 0.0f;
    for (int dy = -SSIM_WINDOW_RADIUS; dy <= SSIM_WINDOW_RADIUS; dy++) {
        for (int dx = -SSIM_WINDOW_RADIUS; dx <= SSIM_WINDOW_RADIUS; dx++) {
            int px = int(gid.x) + dx;
            int py = int(gid.y) + dy;
            
            // Clamp to image boundaries
            px = clamp(px, 0, int(width) - 1);
            py = clamp(py, 0, int(height) - 1);
            
            // Gaussian weight
            float dist_sq = float(dx * dx + dy * dy);
            float w = exp(-dist_sq / two_sigma_sq);
            weight_sum += w;
            
            // Read pixels
            float4 r = rendered.read(uint2(px, py));
            float4 gt = groundTruth.read(uint2(px, py));
            
            // Use luminance or average RGB
            float x_val = (r.r + r.g + r.b) / 3.0f;
            float y_val = (gt.r + gt.g + gt.b) / 3.0f;
            
            // Accumulate variances and covariance
            float dx_val = x_val - mu_x;
            float dy_val = y_val - mu_y;
            
            // Variances and covariance
            sigma_x_sq += w * dx_val * dx_val;
            sigma_y_sq += w * dy_val * dy_val;
            sigma_xy += w * dx_val * dy_val;
        }
    }
    
    // Normalize variances
    sigma_x_sq /= weight_sum;
    sigma_y_sq /= weight_sum;
    sigma_xy /= weight_sum;
    
    // Compute SSIM
    float numerator = (2.0f * mu_x * mu_y + SSIM_C1) * (2.0f * sigma_xy + SSIM_C2);
    float denominator = (mu_x * mu_x + mu_y * mu_y + SSIM_C1) * (sigma_x_sq + sigma_y_sq + SSIM_C2);
    float ssim = numerator / denominator;
    
    // D-SSIM = (1 - SSIM) / 2, clamped to [0, 1]
    float dssim = clamp((1.0f - ssim) / 2.0f, 0.0f, 1.0f);
    
    uint idx = gid.y * width + gid.x;
    ssimMap[idx] = dssim;
}

// Combined loss: 0.8 * L1 + 0.2 * D-SSIM
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

// Gradient structure matching Gaussian parameters
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

// Adam optimizer with safeguards
// NOTE: Using float* instead of float3* to avoid Metal z-component corruption bug
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
    
    // Fetch gradients
    GaussianGradients g = gradients[tid];
    
    // Skip if gradients are invalid
    if (isnan(g.position_x) || isnan(g.opacity) || isnan(g.sh[0]) ||
        isinf(g.position_x) || isinf(g.opacity)) {
        return;
    }
    
    // Skip corrupted Gaussians
    if (isnan(gaussians[tid].position.x) || isinf(gaussians[tid].position.x) ||
        abs(gaussians[tid].position.x) > 1e6) {
        return;
    }
    
    // Bias correction terms
    float bc1 = 1.0 - pow(beta1, float(t));
    float bc2 = 1.0 - pow(beta2, float(t));
    
    float clip = 0.5;
    
    // Position update with magnitude limiting
    // Using manual indexing to avoid float3 z-component corruption
    {
        // Clamp gradients
        float3 grad = clamp(float3(g.position_x, g.position_y, g.position_z), -clip, clip);
        
        // Read manually from float* buffer
        float3 m_old = float3(m_position[tid * 3 + 0], m_position[tid * 3 + 1], m_position[tid * 3 + 2]);
        float3 v_old = float3(v_position[tid * 3 + 0], v_position[tid * 3 + 1], v_position[tid * 3 + 2]);
        
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
        
        // Limit update magnitude
        float updateMag = length(update);
        if (updateMag > 0.1) {
            update = update * (0.1 / updateMag);
        }
        
        // Apply update
        float3 newPos = gaussians[tid].position - update;
        
        // Sanity check
        if (!isnan(newPos.x) && !isnan(newPos.y) && !isnan(newPos.z) &&
            abs(newPos.x) < 1e6 && abs(newPos.y) < 1e6 && abs(newPos.z) < 1e6) {
            gaussians[tid].position = newPos;
        }
    }
    
    // Scale update (stays in log space)
    // Use stricter MAX_SCALE_TRAIN during training to prevent elongation
    // Using manual indexing to avoid float3 z-component corruption
    {
        // Clamp gradients
        float3 rawGrad = float3(g.scale_x, g.scale_y, g.scale_z);
        float3 grad = clamp(rawGrad, -clip, clip);
        
        // Read manually from float* buffer
        float3 m_old = float3(m_scale[tid * 3 + 0], m_scale[tid * 3 + 1], m_scale[tid * 3 + 2]);
        float3 v_old = float3(v_scale[tid * 3 + 0], v_scale[tid * 3 + 1], v_scale[tid * 3 + 2]);
        
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
        gaussians[tid].scale = clamp(newScale, -MAX_SCALE_TRAIN, MAX_SCALE_TRAIN);
    }
    
    // Rotation update
    {
        // Clamp gradients
        float4 grad = clamp(g.rotation, -clip, clip);
        // Adam moment updates
        float4 m = beta1 * m_rotation[tid] + (1.0 - beta1) * grad;
        float4 v = beta2 * v_rotation[tid] + (1.0 - beta2) * grad * grad;
        m_rotation[tid] = m;
        v_rotation[tid] = v;
        
        // Compute bias-corrected estimates
        float4 m_hat = m / bc1;
        float4 v_hat = v / bc2;
        // Update rotation
        float4 newRot = gaussians[tid].rotation - lrs[2] * m_hat / (sqrt(v_hat) + epsilon);
        float rotLen = length(newRot);
        gaussians[tid].rotation = (rotLen > 0.001) ? (newRot / rotLen) : float4(1, 0, 0, 0);
    }
    
    // Opacity update (stays in raw space)
    {
        // Clamp gradient
        float grad = clamp(g.opacity, -clip, clip);
        // Adam moment updates
        float m = beta1 * m_opacity[tid] + (1.0 - beta1) * grad;
        float v = beta2 * v_opacity[tid] + (1.0 - beta2) * grad * grad;
        // Write back
        m_opacity[tid] = m;
        v_opacity[tid] = v;
        
        // Compute bias-corrected estimates
        float m_hat = m / bc1;
        float v_hat = v / bc2;
        gaussians[tid].opacity = clamp(gaussians[tid].opacity - lrs[3] * m_hat / (sqrt(v_hat) + epsilon), -8.0f, 8.0f);
    }
    
    // SH update clamp to reasonable range
    for (int i = 0; i < 12; i++) {
        // Clamp gradient
        float grad = clamp(g.sh[i], -clip, clip);
        uint idx = tid * 12 + i;
        // Adam moment updates
        float m = beta1 * m_sh[idx] + (1.0 - beta1) * grad;
        float v = beta2 * v_sh[idx] + (1.0 - beta2) * grad * grad;
        m_sh[idx] = m;
        v_sh[idx] = v;
        // Compute bias-corrected estimates
        float m_hat = m / bc1;
        float v_hat = v / bc2;
        float newSH = gaussians[tid].sh[i] - lrs[4] * m_hat / (sqrt(v_hat) + epsilon);
        // Clamp SH values to prevent color explosion
        // With SH_C0=0.282: color = SH*0.282 + 0.5
        // SH in [-2, 2] gives color in [0.06, 0.94] - prevents saturation
        gaussians[tid].sh[i] = clamp(newSH, -2.0f, 2.0f);
    }
}

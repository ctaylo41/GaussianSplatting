//
//  shaders.metal
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-26.
//
#define METAL
#include <metal_stdlib>
using namespace metal;

struct Gaussian {
    float3 position;
    float3 scale;
    float4 rotation;  // (w, x, y, z) quaternion
    float opacity;
    float sh[12];     // SH layout: [R0,R1,R2,R3, G0,G1,G2,G3, B0,B1,B2,B3]
};

struct Uniforms {
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4x4 viewProjectionMatrix;
    float2 screenSize;
    float2 focalLength;
    float3 cameraPos;
    float _pad;
};

struct VertexOut {
    float4 position [[position]];
    float3 color;
    float opacity;
    float2 centerScreenPos;
    float3 conic;
    float2 quadOffset;
};

constant float SH_C0 = 0.28209479177387814f;
constant float SH_C1 = 0.4886025119029199f;

float3 evalSH(float sh[12], float3 dir) {
    // SH layout: sh[0-3] = R coefficients, sh[4-7] = G, sh[8-11] = B
    // Index 0, 4, 8 are DC (l=0, m=0) terms
    float3 result = float3(sh[0], sh[4], sh[8]) * SH_C0;
    
    // First order SH for view-dependent effects
    result += float3(sh[1], sh[5], sh[9]) * SH_C1 * dir.y;
    result += float3(sh[2], sh[6], sh[10]) * SH_C1 * dir.z;
    result += float3(sh[3], sh[7], sh[11]) * SH_C1 * dir.x;
    
    return max(result + 0.5f, 0.0f);
}

float3x3 quaternionToMatrix(float4 q) {
    // Assuming q = (w, x, y, z) stored as (q.x=w, q.y=x, q.z=y, q.w=z)
    float w = q.x, x = q.y, y = q.z, z = q.w;
    
    return float3x3(
        1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z), 2.0*(x*z + w*y),
        2.0*(x*y + w*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x),
        2.0*(x*z - w*y), 2.0*(y*z + w*x), 1.0 - 2.0*(x*x + y*y)
    );
}

float3x3 computeCovariance3D(float3 logScale, float4 rotation) {
    float3x3 R = quaternionToMatrix(rotation);
    float3 scale = exp(logScale);
    // Covariance = R * S * S^T * R^T = R * diag(s^2) * R^T
    float3x3 S = float3x3(scale.x * scale.x, 0, 0,
                          0, scale.y * scale.y, 0,
                          0, 0, scale.z * scale.z);
    return R * S * transpose(R);
}

float3 computeCovariance2D(float3 mean, float3x3 covariance3D, float4x4 viewMatrix, float2 focalLength) {
    // Transform mean to view space
    float4 t = viewMatrix * float4(mean, 1.0);
    
    // Clamp to avoid numerical issues
    float limx = 1.3 * focalLength.x / t.z;
    float limy = 1.3 * focalLength.y / t.z;
    float txtz = clamp(t.x / t.z, -limx, limx);
    float tytz = clamp(t.y / t.z, -limy, limy);
    
    // Jacobian of perspective projection
    float J00 = focalLength.x / t.z;
    float J02 = -focalLength.x * txtz / t.z;
    float J11 = focalLength.y / t.z;
    float J12 = -focalLength.y * tytz / t.z;
    
    float3x3 J = float3x3(J00, 0, J02,
                          0, J11, J12,
                          0, 0, 0);
    
    // View rotation (upper-left 3x3 of view matrix)
    float3x3 W = float3x3(viewMatrix[0].xyz,
                          viewMatrix[1].xyz,
                          viewMatrix[2].xyz);
    
    // T = J * W
    float3x3 T = J * W;
    
    // 2D covariance: T * cov3d * T^T
    float3x3 cov = T * covariance3D * transpose(T);
    
    // Add small value for numerical stability (low-pass filter)
    cov[0][0] += 0.3;
    cov[1][1] += 0.3;
    
    // Return upper left 2x2 as float3 (a, b, c) where [[a,b],[b,c]]
    return float3(cov[0][0], cov[0][1], cov[1][1]);
}

float3 computeConic(float3 cov2D) {
    float det = cov2D.x * cov2D.z - cov2D.y * cov2D.y;
    if (abs(det) < 1e-6) det = 1e-6;
    float invDet = 1.0 / det;
    
    return float3(cov2D.z * invDet,
                  -cov2D.y * invDet,
                  cov2D.x * invDet);
}

float computeRadius(float3 covariance2D) {
    float det = covariance2D.x * covariance2D.z - covariance2D.y * covariance2D.y;
    float mid = 0.5 * (covariance2D.x + covariance2D.z);
    float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    float radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));
    return min(radius, 1024.0f);
}

vertex VertexOut vertexShader(
    uint vertexID [[vertex_id]],
    uint instanceID [[instance_id]],
    constant Gaussian* gaussians [[buffer(0)]],
    constant Uniforms& uniforms [[buffer(1)]],
    constant uint32_t* indices [[buffer(2)]])
{
    uint gaussianIdx = indices[instanceID];
    Gaussian g = gaussians[gaussianIdx];
    
    VertexOut out;
    
    float4 clipPos = uniforms.viewProjectionMatrix * float4(g.position, 1.0);
    
    // Skip if behind camera
    if (clipPos.w <= 0.0) {
        out.position = float4(0, 0, -1, 1);
        out.opacity = 0;
        return out;
    }
    
    // Compute 3D covariance
    float3x3 cov3d = computeCovariance3D(g.scale, g.rotation);
    
    // Project to 2D covariance
    float3 cov2d = computeCovariance2D(g.position, cov3d, uniforms.viewMatrix, uniforms.focalLength);
    
    // Compute conic (inverse covariance)
    float3 conic = computeConic(cov2d);
    
    // Compute screen space radius
    float radius = computeRadius(cov2d);
    
    float2 quadOffsets[4] = {
        float2(-1, -1),
        float2(1, -1),
        float2(-1, 1),
        float2(1, 1)
    };
    
    float2 quadOffset = quadOffsets[vertexID];
    
    // Center in NDC
    float2 centerNDC = clipPos.xy / clipPos.w;
    
    // Offset in pixels then convert back to NDC
    float2 offsetPixels = quadOffset * radius;
    float2 offsetNDC = offsetPixels / (uniforms.screenSize * 0.5);
    
    out.position = float4((centerNDC + offsetNDC) * clipPos.w, clipPos.z, clipPos.w);
    
    float3 viewDir = normalize(g.position - uniforms.cameraPos);
    
    out.color = evalSH(g.sh, viewDir);
    out.opacity = 1.0 / (1.0 + exp(-g.opacity));  // Sigmoid
    out.centerScreenPos = float2(
        (centerNDC.x * 0.5 + 0.5) * uniforms.screenSize.x,
        (1.0 - (centerNDC.y * 0.5 + 0.5)) * uniforms.screenSize.y  // Flip Y for screen coords
    );
    out.conic = conic;
    out.quadOffset = offsetPixels;
    
    return out;
}

fragment float4 fragmentShader(VertexOut in [[stage_in]]) {
    // Compute gaussian falloff: g(x,y) = exp(-0.5 * (ax^2 + 2bxy + cy^2))
    float2 d = in.quadOffset;
    float power = -0.5 * (in.conic.x * d.x * d.x +
                          2.0 * in.conic.y * d.x * d.y +
                          in.conic.z * d.y * d.y);
    
    if (power > 0.0) discard_fragment();
    
    float alpha = in.opacity * exp(power);
    
    if (alpha < 1.0 / 255.0) discard_fragment();
    
    return float4(in.color * alpha, alpha);
}

// ============================================================================
// Loss computation kernels
// ============================================================================

kernel void computeL1Loss(
    texture2d<float, access::read> rendered [[texture(0)]],
    texture2d<float, access::read> groundTruth [[texture(1)]],
    device float* losses [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= rendered.get_width() || gid.y >= rendered.get_height()) return;
    
    float4 r = rendered.read(gid);
    float4 gt = groundTruth.read(gid);
    
    float l1 = (abs(r.r - gt.r) + abs(r.g - gt.g) + abs(r.b - gt.b)) / 3.0;
    
    uint idx = gid.y * rendered.get_width() + gid.x;
    losses[idx] = l1;
}

kernel void reduceLoss(
    device float* losses [[buffer(0)]],
    device atomic_float* totalLoss [[buffer(1)]],
    constant uint& pixelCount [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint threads [[threads_per_grid]])
{
    float sum = 0.0;
    for (uint i = tid; i < pixelCount; i += threads) {
        sum += losses[i];
    }
    
    atomic_fetch_add_explicit(totalLoss, sum, memory_order_relaxed);
}

// ============================================================================
// Gradient structs - must match C++ side
// ============================================================================

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

// ============================================================================
// Adam optimizer kernel
// ============================================================================

kernel void adamStep(
    device Gaussian* gaussians [[buffer(0)]],
    device const GaussianGradients* gradients [[buffer(1)]],
    device float3* m_position [[buffer(2)]],
    device float3* m_scale [[buffer(3)]],
    device float4* m_rotation [[buffer(4)]],
    device float* m_opacity [[buffer(5)]],
    device float* m_sh [[buffer(6)]],
    device float3* v_position [[buffer(7)]],
    device float3* v_scale [[buffer(8)]],
    device float4* v_rotation [[buffer(9)]],
    device float* v_opacity [[buffer(10)]],
    device float* v_sh [[buffer(11)]],
    constant float* lrs [[buffer(12)]],       // [pos, scale, rot, opacity, sh]
    constant float& beta1 [[buffer(13)]],
    constant float& beta2 [[buffer(14)]],
    constant float& epsilon [[buffer(15)]],
    constant uint2& params [[buffer(16)]],    // [timestep, numGaussians]
    uint tid [[thread_position_in_grid]])
{
    uint t = params.x;
    uint numGaussians = params.y;
    
    if (tid >= numGaussians) return;
    
    GaussianGradients g = gradients[tid];
    
    // Check for NaN gradients - skip update if found
    if (isnan(g.position_x) || isnan(g.position_y) || isnan(g.position_z) ||
        isnan(g.opacity) || isnan(g.scale_x)) {
        return;
    }
    
    // Bias correction factors
    float bc1 = 1.0 - pow(beta1, float(t));
    float bc2 = 1.0 - pow(beta2, float(t));
    
    // Clamp gradients to prevent explosion
    float grad_clip = 1.0;
    
    // Position
    {
        float3 grad = float3(g.position_x, g.position_y, g.position_z);
        grad = clamp(grad, -grad_clip, grad_clip);
        
        float3 m = beta1 * m_position[tid] + (1.0 - beta1) * grad;
        float3 v = beta2 * v_position[tid] + (1.0 - beta2) * grad * grad;
        m_position[tid] = m;
        v_position[tid] = v;
        
        float3 m_hat = m / bc1;
        float3 v_hat = v / bc2;
        gaussians[tid].position -= lrs[0] * m_hat / (sqrt(v_hat) + epsilon);
    }
    
    // Scale
    {
        float3 grad = float3(g.scale_x, g.scale_y, g.scale_z);
        grad = clamp(grad, -grad_clip, grad_clip);
        
        float3 m = beta1 * m_scale[tid] + (1.0 - beta1) * grad;
        float3 v = beta2 * v_scale[tid] + (1.0 - beta2) * grad * grad;
        m_scale[tid] = m;
        v_scale[tid] = v;
        
        float3 m_hat = m / bc1;
        float3 v_hat = v / bc2;
        float3 newScale = gaussians[tid].scale - lrs[1] * m_hat / (sqrt(v_hat) + epsilon);
        // Clamp scale to reasonable range
        gaussians[tid].scale = clamp(newScale, -10.0f, 10.0f);
    }
    
    // Rotation
    {
        float4 grad = g.rotation;
        grad = clamp(grad, -grad_clip, grad_clip);
        
        float4 m = beta1 * m_rotation[tid] + (1.0 - beta1) * grad;
        float4 v = beta2 * v_rotation[tid] + (1.0 - beta2) * grad * grad;
        m_rotation[tid] = m;
        v_rotation[tid] = v;
        
        float4 m_hat = m / bc1;
        float4 v_hat = v / bc2;
        float4 newRot = gaussians[tid].rotation - lrs[2] * m_hat / (sqrt(v_hat) + epsilon);
        
        // Normalize to keep unit quaternion
        float rotLen = length(newRot);
        gaussians[tid].rotation = (rotLen > 0.001) ? (newRot / rotLen) : float4(1, 0, 0, 0);
    }
    
    // Opacity
    {
        float grad = clamp(g.opacity, -grad_clip, grad_clip);
        
        float m = beta1 * m_opacity[tid] + (1.0 - beta1) * grad;
        float v = beta2 * v_opacity[tid] + (1.0 - beta2) * grad * grad;
        m_opacity[tid] = m;
        v_opacity[tid] = v;
        
        float m_hat = m / bc1;
        float v_hat = v / bc2;
        float newOpacity = gaussians[tid].opacity - lrs[3] * m_hat / (sqrt(v_hat) + epsilon);
        // Clamp to reasonable range for sigmoid input
        gaussians[tid].opacity = clamp(newOpacity, -10.0f, 10.0f);
    }
    
    // SH coefficients
    for (int i = 0; i < 12; i++) {
        float grad = clamp(g.sh[i], -grad_clip, grad_clip);
        uint idx = tid * 12 + i;
        
        float m = beta1 * m_sh[idx] + (1.0 - beta1) * grad;
        float v = beta2 * v_sh[idx] + (1.0 - beta2) * grad * grad;
        m_sh[idx] = m;
        v_sh[idx] = v;
        
        float m_hat = m / bc1;
        float v_hat = v / bc2;
        gaussians[tid].sh[i] -= lrs[4] * m_hat / (sqrt(v_hat) + epsilon);
    }
}

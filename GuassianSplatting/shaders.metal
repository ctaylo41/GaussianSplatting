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
    float4 rotation;
    float opacity;
    float sh[12];
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
    float3 result = float3(sh[0], sh[4], sh[8]) * SH_C0;
    
    result += float3(sh[1], sh[5], sh[9]) * SH_C1 * dir.y;
    result += float3(sh[2], sh[6], sh[10]) * SH_C1 * dir.z;
    result += float3(sh[3], sh[7], sh[11]) * SH_C1 * dir.x;
    
    return max(result + 0.5f, 0.0f);

}

float3x3 quaternionToMatrix(float4 q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    
    return float3x3(1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z), 2.0*(x*z + w*y),
                    2.0*(x*y + w*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x),
                    2.0*(x*z - w*y), 2.0*(y*z + w*x), 1.0 - 2.0*(x*x + y*y));
}

float3x3 computeCovariance3D(float3 scale, float4 rotation) {
    float3x3 R = quaternionToMatrix(rotation);
    
    // S is a diagonal scale matrix we compute R *S * S^T * R^T = R * S^2 * R^T
    float3x3 S = float3x3(scale.x * scale.x,0,0,
                          0, scale.y * scale.y, 0,
                          0, 0, scale.z * scale.z);
    return R * S * transpose(R);
}

float3 computeCovariance2D(float3 mean, float3x3 covariance3D, float4x4 viewMatrix, float2 focalLength) {
    // transform mean to view space
    float4 t = viewMatrix * float4(mean, 1.0);
    
    //clamp to avoid numerical issues
    float limx = 1.3 * focalLength.x / t.z;
    float limy = 1.3 * focalLength.y / t.z;
    float txtz = t.x / t.z;
    float tytz = t.y / t.z;
    
    txtz = clamp(txtz, -limx, limx);
    tytz = clamp(tytz, -limy, limy);
    
    // jacobian of perspective projection
    float J00 = focalLength.x / t.z;
    float J02 = -focalLength.x * txtz / t.z;
    float J11 = focalLength.y / t.z;
    float J12 = -focalLength.y * tytz / t.z;
    
    float3x3 J = float3x3(J00, 0, J02,
                          0, J11, J12,
                          0, 0, 0);
    
    // view rotation upper-left 3x3 of view matrix
    float3x3 W = float3x3(viewMatrix[0].xyz,
                           viewMatrix[1].xyz,
                           viewMatrix[2].xyz);
    
    // T = J * W
    float3x3 T = J * W;
    
    // 2D covariance T * cov3d * T^T
    // only need upper left 2x2
    float3x3 cov = T * covariance3D * transpose(T);
    
    //add small value for numerical stability
    cov[0][0] += 0.3;
    cov[1][1] += 0.3;
    
    //return upper left 2x2
    return float3(cov[0][0],cov[0][1],cov[1][1]);
}

float3 computeConic(float3 cov2D) {
    float det = cov2D.x * cov2D.z - cov2D.y * cov2D.y;
    if (abs(det) < 1e-6) det = 1e-6;
    float invDet = 1.0 / det;
    
    return float3(cov2D.z * invDet,
                  -cov2D.y * invDet,
                  cov2D.x * invDet);
}

// compute rradius that covers the gaussian
float computeRadius(float3 covariance2D) {
    float det = covariance2D.x * covariance2D.z - covariance2D.y * covariance2D.y;
    float mid = 0.5 * (covariance2D.x + covariance2D.z);
    float lamda1 = mid + sqrt(max(0.1, mid*mid - det));
    float lamda2 = mid - sqrt(max(0.1, mid*mid - det));
    float radius = ceil(3.0 * sqrt(max(lamda1, lamda2)));
    return min(radius,1024.0);
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
    
    //skip if behind camera
    if (clipPos.w <= 0.0) {
        out.position = float4(0,0,-1,1);
        out.opacity = 0;
        return out;
    }
    
    //compute 3d covariance
    float3x3 cov3d = computeCovariance3D(g.scale, g.rotation);
    
    //project to 2D covariance
    float3 cov2d = computeCovariance2D(g.position, cov3d, uniforms.viewMatrix, uniforms.focalLength);
    
    //compute conic
    float3 conic = computeConic(cov2d);
    
    //compute screen space radius
    float radius = computeRadius(cov2d);
    
    float2 quadOffsets[4] = {
        float2(-1,-1),
        float2(1,-1),
        float2(-1,1),
        float2(1,1)
    };
    
    float2 quadOffset = quadOffsets[vertexID];
    
    //Center in NDC
    float2 centerNDC = clipPos.xy/clipPos.w;
    
    //offset in pixels then convert back to ndc
    float2 offsetPixels = quadOffset * radius;
    float2 offsetNDC = offsetPixels / (uniforms.screenSize * 0.5);
    
    out.position = float4((centerNDC + offsetNDC) * clipPos.w, clipPos.z, clipPos.w);
    
    float3 viewDir = normalize(g.position - uniforms.cameraPos);
    
    out.color = evalSH(g.sh, viewDir);
    out.opacity = g.opacity;
    out.centerScreenPos = (centerNDC * 0.5 + 0.5) * uniforms.screenSize;
    out.conic = conic;
    out.quadOffset = offsetPixels;
    
    return out;
}

fragment float4 fragmentShader(VertexOut in [[stage_in]]) {
    //compute gaussian falloff
    // g(x,y) = exp(-0.5 * (ax^2 + 2bxy + cy^2))
    float2 d = in.quadOffset;
    float power = -0.5 * (in.conic.x * d.x * d.x + 2.0 * in.conic.y * d.x * d.y + in.conic.z * d.y * d.y);
    
    if (power > 0.0) discard_fragment();
    
    float alpha = in.opacity * exp(power);
    
    if (alpha < 1.0/255.0) discard_fragment();
    
    return float4(in.color*alpha, alpha);
}

kernel void computeL1Loss(texture2d<float, access::read> rendered [[texture(0)]],
                          texture2d<float, access::read> groundTruth [[texture(1)]],
                          device float* losses [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= rendered.get_width() || gid.y >= rendered.get_height()) return;
    
    float4 r = rendered.read(gid);
    float4 gt = groundTruth.read(gid);
    
    float l1 = (abs(r.r - gt.r) + abs(r.g - gt.g) + abs(r.b - gt.b)) / 3.0;
    
    uint idx = gid.y * rendered.get_width() + gid.x;
    losses[idx] = l1;
}

kernel void reduceLoss(device float* losses [[buffer(0)]],
                       device float* totalLoss [[buffer(1)]],
                       constant uint& pixelCount [[buffer(2)]],
                       uint tid [[thread_position_in_grid]],
                       uint threads [[threads_per_grid]]) {
    float sum = 0.0;
    for (uint i=tid; i<pixelCount; i+= threads) {
        sum += losses[i];
    }
    
    atomic_fetch_add_explicit((device atomic_float*)totalLoss, sum, memory_order_relaxed);
}

kernel void computePixelGradient(texture2d<float, access::read> rendered [[texture(0)]],
                                 texture2d<float, access::read> groundTruth [[texture(1)]],
                                 texture2d<float, access::write> gradient [[texture(2)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= rendered.get_width() || gid.y >= rendered.get_height()) return;
    
    float4 r = rendered.read(gid);
    float4 gt = groundTruth.read(gid);
    
    // dL/dC = sign(rendered - gt) for l1
    float4 grad;
    grad.r = (r.r > gt.r) ? 1.0 : -1.0;
    grad.g = (r.g > gt.g) ? 1.0 : -1.0;
    grad.b = (r.b > gt.b) ? 1.0 : -1.0;
    grad.a = 0.0;
    
    gradient.write(grad,gid);
}

struct GaussianGradients {
    float3 position;
    float opacity;
    float3 scale;
    float _pad1;
    float4 rotation;
    float sh[12];
};

kernel void backwardPass(device const Gaussian* gaussians [[buffer(0)]],
                         device GaussianGradients* gradients [[buffer(1)]],
                         device const uint32_t* sortedIndices [[buffer(2)]],
                         constant Uniforms& uniforms [[buffer(3)]],
                         constant uint32_t& numGaussians [[buffer(4)]],
                         texture2d<float, access::read> rendered [[texture(0)]],
                         texture2d<float, access::read> groundTruth [[texture(1)]],
                         uint tid [[thread_position_in_grid]]) {
    if (tid >= numGaussians) return;
    
    uint gaussiansIdx = sortedIndices[tid];
    Gaussian g = gaussians[gaussiansIdx];
    
    // project gaussian to screen for forward pass
    float4 worldPos = float4(g.position, 1.0);
    float4 viewPos = uniforms.viewMatrix * worldPos;
    float4 clipPos = uniforms.viewProjectionMatrix * worldPos;
    
    if(clipPos.w <= 0.0) {
        //behind camera zero gradients
        gradients[gaussiansIdx] = GaussianGradients{};
        return;
    }
    
    float z_cam = viewPos.z;
    float3 ndc = clipPos.xyz / clipPos.w;
    float2 screenPos = float2((ndc.x*0.5 + 0.5) * uniforms.screenSize.x,
                              (1.0 - (ndc.y * 0.5 +0.5)) * uniforms.screenSize.y);
    
    float3 scale = exp(g.scale);
    float4 q = normalize(g.rotation);
    float r  = q.x, x = q.y, y = q.z, z = q.w;
    
    float3x3 R = float3x3(1.0 - 2.0*(y*y + z*z), 2.0*(x*y - r*z), 2.0*(x*z + r*y),
                          2.0 * (x*y + r*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z - r*x),
                          2.0 * (x*z - r*y), 2.0 * (y*z + r*x), 1.0 - 2.0 * (x*x + y*y));
    
    float3x3 S = float3x3(scale.x, 0, 0,
                          0, scale.y, 0,
                          0, 0, scale.z);
    
    float3x3 M = R * S;
    float3x3 Sigma3D = M * transpose(M);
    
    float3x3 viewRot = float3x3(uniforms.viewMatrix.columns[0].xyz,
                                uniforms.viewMatrix.columns[1].xyz,
                                uniforms.viewMatrix.columns[2].xyz);
    
    
    float fx = uniforms.focalLength.x;
    float fy = uniforms.focalLength.y;
    float z2 = z_cam * z_cam;
    
    float2x3 J = float2x3(fx / z_cam, 0, -fx * viewPos.x / z2,
                           0, fy / z_cam, -fy * viewPos.y / z2);
    
    
    float3x3 Sigma_view  = viewRot * Sigma3D * transpose(viewRot);
    
    
    float a = 0, b = 0, c = 0;
    for(int i=0;i<3;i++) {
        for(int j=0;j<3;j++) {
            a+=J[0][i] * Sigma_view[i][j] * J[0][j];
            b+=J[0][i] * Sigma_view[i][j] * J[1][j];
            c+=J[1][i] * Sigma_view[i][j] * J[1][j];
        }
    }
    
    a+=0.3;
    c+=0.3;
    
    float det = a * c - b*b;
    if (det <= 0.0) {
        gradients[gaussiansIdx] = GaussianGradients{};
        return;
    }
    
    float inv_det = 1.0/det;
    float cov_inv_00 = c * inv_det;
    float cov_inv_01 = -b * inv_det;
    float cov_inv_11 = a * inv_det;
    
    float radius = 3.0 * sqrt(max(a,c));
    
    int2 minBound = int2(max(0, int(screenPos.x - radius)), max(0, int(screenPos.y - radius)));
    int2 maxBound = int2(min(int(uniforms.screenSize.x) - 1, int(screenPos.x + radius)),
                         min(int(uniforms.screenSize.y) - 1, int(screenPos.y + radius)));

    
    float opacity = 1.0 / (1.0 + exp(-g.opacity));
    
    float SH_C0 = 0.28209479177387814;
    float3 color = clamp(float3(
        SH_C0 * g.sh[0] + 0.5,
        SH_C0 * g.sh[1] + 0.5,
        SH_C0 * g.sh[2] + 0.5
    ),0.0, 1.0);
    

    GaussianGradients grad = {};
    
    float2 dL_dScreenPos = float2(0);
    float dL_dCov00 = 0, dL_dCov01 = 0, dL_dCov11 = 0;
    for (int py = minBound.y; py<= maxBound.y;py++) {
        for(int px=minBound.x;px<=maxBound.x;px++) {
            float2 d = float2(px,py) - screenPos;
            
            float power = -0.5 * (cov_inv_00 * d.x * d.x + 2.0 * cov_inv_01 * d.x * d.y + cov_inv_11 * d.y * d.y);
            if (power > 0.0) continue;
            
            float G = exp(power);
            float alpha = opacity * G;
            
            if(alpha<1.0/255.0) continue;
            
            float4 r = rendered.read(uint2(px, py));
            float4 gt = groundTruth.read(uint2(px, py));
            
            float3 dL_dC = float3(
                (r.r > gt.r) ? 1.0 : -1.0,
                (r.g > gt.g) ? 1.0 : -1.0,
                (r.b > gt.b) ? 1.0 : -1.0
            );
            
            float T = max(1.0 - r.a, 0.01);
            
            float weight = alpha * T;
            float dL_dAlpha = dot(dL_dC, color) * T;
            
            float dL_dG = dL_dAlpha * opacity;
            
            float2 cov_inv_d = float2(cov_inv_00 * d.x + cov_inv_01 * d.y,
                                      cov_inv_01 * d.x + cov_inv_11 * d.y);
            
            dL_dScreenPos += dL_dG * G * cov_inv_d;
            
            float dL_dPower = dL_dG * G;
            
            float d_power_d_cov_inv_00 = -0.5 * d.x * d.x;
            float d_power_d_cov_inv_01 = -d.x * d.y;
            float d_power_d_cov_inv_11 = -0.5 * d.y * d.y;
            
            float inv_det2 = inv_det * inv_det;
            
            float dL_dCovInv00 = dL_dPower * d_power_d_cov_inv_00;
            float dL_dCovInv01 = dL_dPower * d_power_d_cov_inv_01;
            float dL_dCovInv11 = dL_dPower * d_power_d_cov_inv_11;
            
            dL_dCov00 += -dL_dCovInv00 * c * c * inv_det2
                         + dL_dCovInv01 * b * c * inv_det2
                         + dL_dCovInv11 * (inv_det - a * c * inv_det2);
            dL_dCov01 += dL_dCovInv00 * 2 * b * c * inv_det2
                         + dL_dCovInv01 * (-inv_det - 2 * b * b * inv_det2)
                         + dL_dCovInv11 * 2 * a * b * inv_det2;
            dL_dCov11 += dL_dCovInv00 * (inv_det - a * c * inv_det2)
                         + dL_dCovInv01 * a * b * inv_det2
                         - dL_dCovInv11 * a * a * inv_det2;
            
            grad.sh[0] += dL_dC.r * weight * SH_C0;
            grad.sh[1] += dL_dC.g * weight * SH_C0;
            grad.sh[2] += dL_dC.b * weight * SH_C0;
            
            float sigmoid_deriv = opacity * (1.0 - opacity);  // sigmoid derivative
            grad.opacity += dL_dAlpha * G * sigmoid_deriv;
        }
    }
    
    //backprop from screen space to 3d
    
    float3x3 VP3 = float3x3(uniforms.viewProjectionMatrix.columns[0].xyz,
                            uniforms.viewProjectionMatrix.columns[1].xyz,
                            uniforms.viewProjectionMatrix.columns[2].xyz);
    
    float w = clipPos.w;
    float2 d_screen_d_ndcX = float2(uniforms.screenSize.x * 0.5, 0);
    float2 d_screen_d_ndcY = float2(0, -uniforms.screenSize.y * 0.5);
    
    float3 dL_dClip;
    dL_dClip.x = (dL_dScreenPos.x * d_screen_d_ndcX.x) / w;
    dL_dClip.y = (dL_dScreenPos.y * d_screen_d_ndcY.y) / w;
    dL_dClip.z = 0;
    
    grad.position = transpose(VP3) * dL_dClip;
    
    float3x3 dL_dSigmaView = float3x3(0);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            dL_dSigmaView[i][j] = J[0][i] * J[0][j] * dL_dCov00
                                 + (J[0][i] * J[1][j] + J[1][i] * J[0][j]) * dL_dCov01
                                 + J[1][i] * J[1][j] * dL_dCov11;
        }
    }
    
    float3x3 dL_dSigma3D = transpose(viewRot) * dL_dSigmaView * viewRot;

    float3x3 dL_dM = 2.0 * dL_dSigma3D * M;

    float3x3 dL_dS = transpose(R) * dL_dM;

    grad.scale = float3(dL_dS[0][0], dL_dS[1][1], dL_dS[2][2]) * scale;

    float3x3 dL_dR = dL_dM * transpose(S);

    float dL_dr = 2.0 * (-z * (dL_dR[0][1] - dL_dR[1][0]) + y * (dL_dR[0][2] - dL_dR[2][0]) - x * (dL_dR[1][2] - dL_dR[2][1]));
    float dL_dx = 2.0 * (y * (dL_dR[0][1] + dL_dR[1][0]) + z * (dL_dR[0][2] + dL_dR[2][0]) - 2*x * (dL_dR[1][1] + dL_dR[2][2]));
    float dL_dy = 2.0 * (x * (dL_dR[0][1] + dL_dR[1][0]) - 2*y * (dL_dR[0][0] + dL_dR[2][2]) + z * (dL_dR[1][2] + dL_dR[2][1]));
    float dL_dz = 2.0 * (x * (dL_dR[0][2] + dL_dR[2][0]) + y * (dL_dR[1][2] + dL_dR[2][1]) - 2*z * (dL_dR[0][0] + dL_dR[1][1]));
    
    grad.rotation = float4(dL_dr, dL_dx, dL_dy, dL_dz);
    
    gradients[gaussiansIdx] = grad;


}

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
    float _pad0;
    float _pad1;
    float _pad2;
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
        float2(-1,1),
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

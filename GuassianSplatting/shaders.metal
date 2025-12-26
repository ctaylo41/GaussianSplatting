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
    float sh_dc;
};

struct Uniforms {
    float4x4 viewProjection;
};

struct VertexOut {
    float4 position [[position]];
    float3 color;
};

vertex VertexOut vertexShader(
                            uint vertexID [[vertex_id]],
                            constant Gaussian* gaussians [[buffer(0)]],
                            constant Uniforms& uniforms [[buffer(1)]])
{
    Gaussian g = gaussians[vertexID];
    
    VertexOut out;
    out.position = uniforms.viewProjection * float4(g.position, 1.0f);
    out.color = g.sh_dc;
    
    return out;
}

fragment float4 fragmentShader(VertexOut in [[stage_in]]) {
    return float4(in.color.x,in.color.y,in.color.z,1.0f);
}

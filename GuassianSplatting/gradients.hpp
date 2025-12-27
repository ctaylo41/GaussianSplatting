//
//  gradients.hpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-27.
//

#include <simd/simd.h>

struct GaussianGradients {
    simd_float3 position;
    float opacity;
    simd_float3 scale;
    float _pad1;
    simd_float4 rotation;
    float sh[12];
};

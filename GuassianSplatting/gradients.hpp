//
//  gradients.hpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-27.
//
#pragma once
#include <simd/simd.h>

struct GaussianGradients {
    float position_x;
    float position_y;
    float position_z;
    float opacity;
    float scale_x;
    float scale_y;
    float scale_z;
    float _pad1;
    simd_float4 rotation;
    float sh[12];
    
    // Viewspace (screen-space) gradients for density control
    // Official 3DGS uses these for densification decisions, not world-space gradients
    float viewspace_grad_x;  // dL/dScreenPos.x
    float viewspace_grad_y;  // dL/dScreenPos.y
    float _pad2;
    float _pad3;
};

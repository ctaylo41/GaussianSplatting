//
//  gradients.hpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-27.
//
#pragma once
#include <simd/simd.h>

// Structure to hold Gaussian gradients for optimization
// MUST match Metal shader struct layout exactly (112 bytes)
struct GaussianGradients {
    float position_x;        // offset 0
    float position_y;        // offset 4
    float position_z;        // offset 8
    float opacity;           // offset 12
    float scale_x;           // offset 16
    float scale_y;           // offset 20
    float scale_z;           // offset 24
    float _pad1;             // offset 28
    simd_float4 rotation;    // offset 32 (16 bytes)
    float sh[12];            // offset 48 (48 bytes)
    
    // Viewspace gradients for density control
    float viewspace_grad_x;  // offset 96 - dL/dScreenPos.x
    float viewspace_grad_y;  // offset 100 - dL/dScreenPos.y
    float _pad2;             // offset 104 - explicit padding
    float _pad3;             // offset 108
    // Total: 112 bytes
};

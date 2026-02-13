//
//  gradients.hpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-27.
//
#pragma once
#include <simd/simd.h>

// Intermediate per-Gaussian render gradients 
// Must match Metal shader RenderGradientsAtomic/RenderGradientsRead layout exactly (40 bytes)
struct RenderGradients {
    float dL_dColor[3];  
    float dL_dConic[3]; 
    float dL_dOpacity;       
    float dL_dMean2D[2];     
    float _pad;
};

// Structure to hold Gaussian gradients for optimization
// Must match Metal shader struct layout exactly 112 bytes
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
    
    // Viewspace gradients for density control
    float viewspace_grad_x;
    float viewspace_grad_y;
    float _pad2;             
    float _pad3;             
};

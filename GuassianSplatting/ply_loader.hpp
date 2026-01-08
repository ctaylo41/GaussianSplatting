//
//  mtl_engine.hpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-24.
//

#pragma once
#include <simd/simd.h>
#include <string>
#include <vector>

// Structure to hold Gaussian parameters
struct Gaussian {
    simd_float3 position;
    simd_float3 scale;
    simd_float4 rotation;
    float opacity;
    float sh[12];
};

std::vector<Gaussian> load_ply(const std::string& file_path);

std::vector<Gaussian> load_ply(const std::string& file_path);

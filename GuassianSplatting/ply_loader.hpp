//
//  ply_loader.hpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-26.
//
#pragma once
#include <simd/simd.h>
#include <string>
#include <vector>

struct Gaussian {
    simd_float3 position;
    simd_float3 scale;
    simd_float4 rotation;
    float opacity;
    float sh[12];
};

std::vector<Gaussian> load_ply(const std::string& file_path);

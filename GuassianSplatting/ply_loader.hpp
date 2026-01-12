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
// MUST match Metal shader struct layout exactly (112 bytes)
struct Gaussian {
    simd_float3 position;   // 16 bytes (offset 0)
    simd_float3 scale;      // 16 bytes (offset 16)
    simd_float4 rotation;   // 16 bytes (offset 32)
    float opacity;          // 4 bytes (offset 48)
    float sh[12];           // 48 bytes (offset 52)
    float _pad2;            // 4 bytes (offset 100) - explicit padding to match Metal
    float _pad3;            // 4 bytes (offset 104)
    float _pad4;            // 4 bytes (offset 108)
    // Total: 112 bytes
};

std::vector<Gaussian> load_ply(const std::string& file_path);

std::vector<Gaussian> load_ply(const std::string& file_path);

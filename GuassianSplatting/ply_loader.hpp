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

// IMPORTANT: simd_float3 is 16 bytes in C++ (includes padding)
// Metal float3 in structs is also 16 bytes (includes padding)
// Layout verified from user output:
//   position: offset 0, scale: offset 16, rotation: offset 32
//   opacity: offset 48, sh: offset 52
struct Gaussian {
    simd_float3 position;   // offset 0, 16 bytes (12 data + 4 implicit padding)
    simd_float3 scale;      // offset 16, 16 bytes (12 data + 4 implicit padding)
    simd_float4 rotation;   // offset 32, 16 bytes
    float opacity;          // offset 48, 4 bytes
    float sh[12];           // offset 52, 48 bytes
    // Total: 100 bytes, padded to 112 for 16-byte struct alignment
};

std::vector<Gaussian> load_ply(const std::string& file_path);

std::vector<Gaussian> load_ply(const std::string& file_path);

//
//  image_loader.hpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-27.
//
#pragma once
#include <simd/simd.h>
#include <Metal/Metal.hpp>
#include <string>
#include <vector>
#include "colmap_loader.hpp"

// Structure to hold training image data
struct TrainingImage {
    uint32_t imageId;
    uint32_t cameraId;
    MTL::Texture* texture;
    simd_float4 rotation;
    simd_float3 translation;
};

// Load an image from disk and create a Metal texture
MTL::Texture* loadImageAsTexture(MTL::Device* device, const std::string& path);
// Load all training images based on COLMAP data
std::vector<TrainingImage> loadTrainingImages(MTL::Device* device, const ColmapData& colmap, const std::string& imagePath);

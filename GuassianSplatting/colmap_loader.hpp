//
//  colmap_loader.hpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-27.
//

#pragma once
#include <simd/simd.h>
#include <string>
#include <vector>
#include <map>

// Data structures to hold COLMAP data
struct ColmapCamera {
    uint32_t id;
    uint32_t width;
    uint32_t height;
    float fx, fy;
    float cx, cy;
};

// Image structure holding pose and camera association
struct ColmapImage {
    uint32_t id;
    simd_float4 rotation;
    simd_float3 translation;
    uint32_t cameraId;
    std::string filename;
};

// 3D Point structure holding position, color, and reprojection error
struct ColmapPoint {
    simd_float3 position;
    simd_float3 color;
    float error;
};

// Aggregate COLMAP data
struct ColmapData {
    std::map<uint32_t, ColmapCamera> cameras;
    std::vector<ColmapImage> images;
    std::vector<ColmapPoint> points;
};

// Function declarations for loading COLMAP data and computing scene extent
ColmapData loadColmap(const std::string& path);
float computeSceneExtent(const ColmapData& colmap);
simd_float3 getCameraWorldPosition(const ColmapImage& img);

//
//  colmap_loader.cpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-27.
//

#include "colmap_loader.hpp"
#include <fstream>
#include <iostream>
#include <string>

// Helper function to get parameter count based on camera model ID
int getParamCount(int modelId) {
    switch (modelId) {
        case 0: return 3; //pinhole
        case 1: return 4; // pinhole fx fy
        case 2: return 4; // simple radial
        case 3: return 5; // radial k1 k2
        case 4: return 8; // open cv
        default: return 4;
    }
}

// Load cameras from COLMAP binary file
std::map<uint32_t, ColmapCamera> loadCameras(const std::string& path) {
    std::map<uint32_t, ColmapCamera> cameras;
    
    // Open the binary file for reading camera data
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << path << std::endl;
        return cameras;
    }
    
    // Read number of cameras
    uint64_t numCameras;
    // Cast to uint64_t to match COLMAP binary format
    file.read(reinterpret_cast<char*>(&numCameras), sizeof(uint64_t));
    
    std::cout << "Loading " << numCameras << " cameras..." << std::endl;
    
    // Read each camera entry
    for(uint64_t i=0;i<numCameras;i++) {
        uint32_t cameraId;
        int32_t modelId;
        uint64_t width, height;
        
        // Cast to each fields size according to COLMAP binary format
        file.read(reinterpret_cast<char*>(&cameraId), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&modelId), sizeof(int32_t));
        file.read(reinterpret_cast<char*>(&width), sizeof(uint64_t));
        file.read(reinterpret_cast<char*>(&height), sizeof(uint64_t));
        
        // Read camera parameters
        int paramCount = getParamCount(modelId);
        std::vector<double> params(paramCount);
        file.read(reinterpret_cast<char*>(params.data()), sizeof(double)*paramCount);
        
        // Populate ColmapCamera structure
        ColmapCamera cam;
        cam.id = cameraId;
        cam.width= static_cast<uint32_t>(width);
        cam.height = static_cast<uint32_t>(height);
        
        // Assign intrinsic parameters based on model
        if (modelId == 0 || modelId == 2 || modelId == 3) {
            cam.fx = cam.fy = static_cast<float>(params[0]);
            cam.cx = static_cast<float>(params[1]);
            cam.cy = static_cast<float>(params[2]);
        } else {
            cam.fx = static_cast<float>(params[0]);
            cam.fy = static_cast<float>(params[1]);
            cam.cx = static_cast<float>(params[2]);
            cam.cy = static_cast<float>(params[3]);
        }
        cameras[cameraId] = cam;
    }
    
    return cameras;
}

// Load images from COLMAP binary file
std::vector<ColmapImage> loadImages(const std::string& path) {
    std::vector<ColmapImage> images;
    
    // Open the binary file for reading image data
    std::ifstream file(path, std::ios::binary);
    if(!file.is_open()) {
        std::cerr << "Failed to open: " << path << std::endl;
        return images;
    }
    
    // Read number of images
    uint64_t numImages;
    file.read(reinterpret_cast<char*>(&numImages),sizeof(uint64_t));
    
    std::cout << "Loading " << numImages << " images..." << std::endl;
    
    // Read each image entry
    for(int i=0;i<numImages;i++) {
        uint32_t image_id;
        double qw, qx, qy, qz;
        double tx, ty, tz;
        uint32_t cameraId;
        std::string name;
        uint64_t numPoints2D;
        
        // Read image data according to COLMAP binary format
        file.read(reinterpret_cast<char*>(&image_id), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&qw),sizeof(double));
        file.read(reinterpret_cast<char*>(&qx),sizeof(double));
        file.read(reinterpret_cast<char*>(&qy),sizeof(double));
        file.read(reinterpret_cast<char*>(&qz),sizeof(double));
        file.read(reinterpret_cast<char*>(&tx),sizeof(double));
        file.read(reinterpret_cast<char*>(&ty),sizeof(double));
        file.read(reinterpret_cast<char*>(&tz),sizeof(double));
        file.read(reinterpret_cast<char*>(&cameraId), sizeof(uint32_t));
        
        // Read null-terminated string for image name
        char c;
        while (file.read(&c, 1) && c!= '\0') {
            name+=c;
        }
        
        // Skip 2D points data
        file.read(reinterpret_cast<char*>(&numPoints2D), sizeof(uint64_t));
        file.seekg(numPoints2D * 24, std::ios::cur);

        // Populate ColmapImage structure
        ColmapImage image;
        image.id = image_id;
        // Store quaternion as float4(.x=w, .y=x, .z=y, .w=z) to match Gaussian convention
        image.rotation = simd_make_float4(static_cast<float>(qw), static_cast<float>(qx), static_cast<float>(qy), static_cast<float>(qz));
        image.translation = simd_make_float3(static_cast<float>(tx), static_cast<float>(ty), static_cast<float>(tz));
        image.cameraId = cameraId;
        image.filename = name;
        
        images.push_back(image);
    }
    return images;
}

// Load 3D points from COLMAP binary file
std::vector<ColmapPoint> loadPoints(const std::string& path) {
    std::vector<ColmapPoint> points;
    
    // Open the binary file for reading 3D point data
    std::ifstream file(path, std::ios::binary);
    if(!file.is_open()) {
        std::cerr << "Failed to open: " << path << std::endl;
        return points;
    }
    
    // Read number of 3D points
    uint64_t numPoints;
    file.read(reinterpret_cast<char*>(&numPoints), sizeof(uint64_t));
    std::cout << "Loading " << numPoints << " points..." << std::endl;
    
    // Read each 3D point entry
    for(int i=0;i<numPoints;i++) {
        uint64_t point3dId;
        double x, y, z;
        uint8_t r, g, b;
        double error;
        uint64_t track_length;
        
        // Read point data according to COLMAP binary format
        file.read(reinterpret_cast<char*>(&point3dId), sizeof(uint64_t));
        file.read(reinterpret_cast<char*>(&x), sizeof(double));
        file.read(reinterpret_cast<char*>(&y), sizeof(double));
        file.read(reinterpret_cast<char*>(&z), sizeof(double));
        file.read(reinterpret_cast<char*>(&r), sizeof(uint8_t));
        file.read(reinterpret_cast<char*>(&g), sizeof(uint8_t));
        file.read(reinterpret_cast<char*>(&b), sizeof(uint8_t));
        file.read(reinterpret_cast<char*>(&error), sizeof(double));
        file.read(reinterpret_cast<char*>(&track_length), sizeof(uint64_t));
        file.seekg(track_length * 8, std::ios::cur);

        ColmapPoint point;
        point.position = simd_make_float3(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
        point.color = simd_make_float3(r / 255.0f, g / 255.0f, b / 255.0f);
        point.error = static_cast<float>(error);
        
        points.push_back(point);
    }
    
    return points;
}
// Main function to load COLMAP data
ColmapData loadColmap(const std::string& path) {
    ColmapData data;
    data.cameras = loadCameras(path + "/cameras.bin");
    data.images = loadImages(path + "/images.bin");
    data.points = loadPoints(path + "/points3D.bin");
    return data;
}

// Compute the world position of a camera from its COLMAP image data
simd_float3 getCameraWorldPosition(const ColmapImage& img) {
    // COLMAP stores quaternion as (qw, qx, qy, qz)
    // Build rotation matrix from quaternion
    float qw = img.rotation.x; 
    float qx = img.rotation.y;
    float qy = img.rotation.z;
    float qz = img.rotation.w;
    
    // Rotation matrix R from quaternion
    float r00 = 1 - 2*(qy*qy + qz*qz);
    float r01 = 2*(qx*qy - qz*qw);
    float r02 = 2*(qx*qz + qy*qw);
    float r10 = 2*(qx*qy + qz*qw);
    float r11 = 1 - 2*(qx*qx + qz*qz);
    float r12 = 2*(qy*qz - qx*qw);
    float r20 = 2*(qx*qz - qy*qw);
    float r21 = 2*(qy*qz + qx*qw);
    float r22 = 1 - 2*(qx*qx + qy*qy);
    
    // Camera center C = -R^T * t
    float tx = img.translation.x;
    float ty = img.translation.y;
    float tz = img.translation.z;
    
    // R^T * t (transpose of R times t)
    float cx = -(r00*tx + r10*ty + r20*tz);
    float cy = -(r01*tx + r11*ty + r21*tz);
    float cz = -(r02*tx + r12*ty + r22*tz);
    
    return simd_make_float3(cx, cy, cz);
}

float computeSceneExtent(const ColmapData& colmap) {
    // Get all camera world positions
    std::vector<simd_float3> camPositions;
    for (const auto& img : colmap.images) {
        camPositions.push_back(getCameraWorldPosition(img));
    }
    
    // Compute centroid
    simd_float3 centroid = {0, 0, 0};
    for (const auto& p : camPositions) {
        centroid.x += p.x;
        centroid.y += p.y;
        centroid.z += p.z;
    }
    centroid.x /= camPositions.size();
    centroid.y /= camPositions.size();
    centroid.z /= camPositions.size();
    
    // Find max distance from centroid
    float maxDist = 0;
    for (const auto& p : camPositions) {
        float dist = simd_length(p - centroid);
        maxDist = std::max(maxDist, dist);
    }
    
    // This is the "nerf_normalization" radius from official code
    float radius = maxDist * 1.1f;
    
    std::cout << "Camera centroid: (" << centroid.x << ", " << centroid.y << ", " << centroid.z << ")" << std::endl;
    std::cout << "Camera extent (radius): " << radius << std::endl;
    
    return radius;
}

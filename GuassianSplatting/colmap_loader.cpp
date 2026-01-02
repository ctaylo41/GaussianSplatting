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

std::map<uint32_t, ColmapCamera> loadCameras(const std::string& path) {
    std::map<uint32_t, ColmapCamera> cameras;
    
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << path << std::endl;
        return cameras;
    }
    
    uint64_t numCameras;
    file.read(reinterpret_cast<char*>(&numCameras), sizeof(uint64_t));
    
    std::cout << "Loading " << numCameras << " cameras..." << std::endl;
    
    for(uint64_t i=0;i<numCameras;i++) {
        uint32_t cameraId;
        int32_t modelId;
        uint64_t width, height;
        
        file.read(reinterpret_cast<char*>(&cameraId), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&modelId), sizeof(int32_t));
        file.read(reinterpret_cast<char*>(&width), sizeof(uint64_t));
        file.read(reinterpret_cast<char*>(&height), sizeof(uint64_t));
        
        int paramCount = getParamCount(modelId);
        std::vector<double> params(paramCount);
        file.read(reinterpret_cast<char*>(params.data()), sizeof(double)*paramCount);
        
        ColmapCamera cam;
        cam.id = cameraId;
        cam.width= static_cast<uint32_t>(width);
        cam.height = static_cast<uint32_t>(height);
        
        
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

std::vector<ColmapImage> loadImages(const std::string& path) {
    std::vector<ColmapImage> images;
    
    std::ifstream file(path, std::ios::binary);
    if(!file.is_open()) {
        std::cerr << "Failed to open: " << path << std::endl;
        return images;
    }
    
    uint64_t numImages;
    file.read(reinterpret_cast<char*>(&numImages),sizeof(uint64_t));
    
    std::cout << "Loading " << numImages << " images..." << std::endl;
    
    for(int i=0;i<numImages;i++) {
        uint32_t image_id;
        double qw, qx, qy, qz;
        double tx, ty, tz;
        uint32_t cameraId;
        std::string name;
        uint64_t numPoints2D;
        
        file.read(reinterpret_cast<char*>(&image_id), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&qw),sizeof(double));
        file.read(reinterpret_cast<char*>(&qx),sizeof(double));
        file.read(reinterpret_cast<char*>(&qy),sizeof(double));
        file.read(reinterpret_cast<char*>(&qz),sizeof(double));
        file.read(reinterpret_cast<char*>(&tx),sizeof(double));
        file.read(reinterpret_cast<char*>(&ty),sizeof(double));
        file.read(reinterpret_cast<char*>(&tz),sizeof(double));
        file.read(reinterpret_cast<char*>(&cameraId), sizeof(uint32_t));
        
        char c;
        while (file.read(&c, 1) && c!= '\0') {
            name+=c;
        }
        
        file.read(reinterpret_cast<char*>(&numPoints2D), sizeof(uint64_t));
        file.seekg(numPoints2D * 24, std::ios::cur);

        ColmapImage image;
        image.id = image_id;
        // Store quaternion as float4(.x=w, .y=x, .z=y, .w=z) to match Gaussian convention
        // COLMAP binary format reads: qw, qx, qy, qz (in that order)
        image.rotation = simd_make_float4(static_cast<float>(qw), static_cast<float>(qx), static_cast<float>(qy), static_cast<float>(qz));
        image.translation = simd_make_float3(static_cast<float>(tx), static_cast<float>(ty), static_cast<float>(tz));
        image.cameraId = cameraId;
        image.filename = name;
        
        images.push_back(image);
    }
    return images;
}

std::vector<ColmapPoint> loadPoints(const std::string& path) {
    std::vector<ColmapPoint> points;
    
    std::ifstream file(path, std::ios::binary);
    if(!file.is_open()) {
        std::cerr << "Failed to open: " << path << std::endl;
        return points;
    }
    
    uint64_t numPoints;
    file.read(reinterpret_cast<char*>(&numPoints), sizeof(uint64_t));
    std::cout << "Loading " << numPoints << " points..." << std::endl;
    
    for(int i=0;i<numPoints;i++) {
        uint64_t point3dId;
        double x, y, z;
        uint8_t r, g, b;
        double error;
        uint64_t track_length;
        
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

ColmapData loadColmap(const std::string& path) {
    ColmapData data;
    data.cameras = loadCameras(path + "/cameras.bin");
    data.images = loadImages(path + "/images.bin");
    data.points = loadPoints(path + "/points3D.bin");
    return data;
}

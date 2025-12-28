//
//  main.cpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-24.
//


#include <iostream>
#include <Metal/Metal.hpp>
#include "mtl_engine.hpp"
#include "ply_loader.hpp"
#include <cstddef>
#include "colmap_loader.hpp"
#include "image_loader.hpp"

std::vector<Gaussian> gaussiansFromColmap(const ColmapData& colmap) {
    std::vector<Gaussian> gaussians;
    gaussians.reserve(colmap.points.size());
    
    for (const auto& pt : colmap.points) {
        Gaussian g;
        g.position = pt.position;
        g.scale = simd_make_float3(-5.0f, -5.0f, -5.01f);  // Small initial scale
        g.rotation = simd_make_float4(1, 0, 0, 0);         // Identity quaternion (w,x,y,z)
        g.opacity = 0.0f;
        
        // Initialize SH from point color (DC term only)
        float SH_C0 = 0.28209479177387814f;
        g.sh[0] = (pt.color.x - 0.5f) / SH_C0;  // R
        g.sh[1] = (pt.color.y - 0.5f) / SH_C0;  // G
        g.sh[2] = (pt.color.z - 0.5f) / SH_C0;  // B
        for (int i = 3; i < 12; i++) g.sh[i] = 0.0f;
        
        gaussians.push_back(g);
    }
    
    return gaussians;
}


int main() {
    
    ColmapData colmap = loadColmap("/Users/colintaylortaylor/Documents/GuassianSplatting/GuassianSplatting/scenes/sparse/0/");
    
    auto gaussians = gaussiansFromColmap(colmap);
    printf("sizeof(Gaussian): %zu\n", sizeof(Gaussian));
    printf("offsetof position: %zu\n", offsetof(Gaussian, position));
    printf("offsetof scale: %zu\n", offsetof(Gaussian, scale));
    printf("offsetof rotation: %zu\n", offsetof(Gaussian, rotation));
    printf("offsetof opacity: %zu\n", offsetof(Gaussian, opacity));
    printf("offsetof sh: %zu\n", offsetof(Gaussian, sh));

    float min_x = FLT_MAX, min_y = FLT_MAX, min_z = FLT_MAX;
    float max_x = -FLT_MAX, max_y = -FLT_MAX, max_z = -FLT_MAX;

    for(Gaussian g : gaussians) {
        min_x = fmin(min_x, g.position.x);
        max_x = fmax(max_x, g.position.x);
        
        min_y = fmin(min_y, g.position.y);
        max_y = fmax(max_y, g.position.y);
        
        min_z = fmin(min_z, g.position.z);
        max_z = fmax(max_z, g.position.z);
    }
    simd_float3 center = simd_make_float3((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2);
    float diagonal = simd_length(simd_make_float3(max_x - min_x, max_y - min_y, max_z - min_z));
    
    Camera camera = Camera(center, 0, 0.3, 1.5*diagonal, 45.0f * M_PI/180.0f, 800.0f/600.0f, 0.1f, 10*diagonal);
    
    printf("Camera position: (%.3f, %.3f, %.3f)\n",
           camera.get_position().x,
           camera.get_position().y,
           camera.get_position().z);
    printf("Target: (%.3f, %.3f, %.3f)\n", center.x, center.y, center.z);
    printf("Distance: %.3f\n", 1.5f * diagonal);

    printf("Bounding box: (%.3f, %.3f, %.3f) to (%.3f, %.3f, %.3f)\n",
           min_x, min_y, min_z, max_x, max_y, max_z);
    printf("Diagonal: %.3f\n", diagonal);
    
    printf("sizeof(Gaussian): %zu\n", sizeof(Gaussian));
    printf("sizeof(simd_float3): %zu\n", sizeof(simd_float3));
    printf("sizeof(simd_float4): %zu\n", sizeof(simd_float4));

    
    MTLEngine engine;
    engine.initHeadless();
    engine.loadTrainingData(colmap, "/Users/colintaylortaylor/Documents/GuassianSplatting/GuassianSplatting/scenes/images");
    engine.loadGaussians(gaussians);
    engine.train(10);
    engine.cleanup();
    return 0;
}

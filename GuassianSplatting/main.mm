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


int main() {
    auto gaussians = load_ply("/Users/colintaylortaylor/Documents/GuassianSplatting/GuassianSplatting/scenes/father-day.ply");
    
    if(!gaussians.empty()) {
        printf("First gaussian position: (%.3f, %.3f, %.3f)\n",
               gaussians[0].position.x,
               gaussians[0].position.y,
               gaussians[0].position.z);
        
        printf("First gaussian color: (%.3f, %.3f, %.3f)\n",
               gaussians[0].sh_dc.x,
               gaussians[0].sh_dc.y,
               gaussians[0].sh_dc.z);
    }
    
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
    engine.init();
    engine.loadGaussians(gaussians);
    engine.run(camera);
    engine.cleanup();
    return 0;
}

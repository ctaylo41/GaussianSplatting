//
//  main.mm
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
    
    // SH_C0 constant for DC term
    const float SH_C0 = 0.28209479177387814f;
    
    for (const auto& pt : colmap.points) {
        Gaussian g;
        g.position = pt.position;
        
        // Initial scale: exp(-4) ≈ 0.018, gives small starting size
        // Paper suggests initializing based on mean distance to nearest points
        g.scale = simd_make_float3(-4.0f, -4.0f, -4.0f);
        
        // Identity quaternion in (w, x, y, z) format stored as (x=w, y=x, z=y, w=z)
        // Based on your quaternionToMatrix: float w = q.x, x = q.y, y = q.z, z = q.w;
        g.rotation = simd_make_float4(1.0f, 0.0f, 0.0f, 0.0f);  // (w=1, x=0, y=0, z=0)
        
        // Initial opacity: sigmoid(-2) ≈ 0.12, gives moderately transparent starting point
        g.opacity = -2.0f;
        
        // Initialize SH coefficients
        // SH layout: sh[0-3] = R coefficients (DC, then first order)
        //            sh[4-7] = G coefficients
        //            sh[8-11] = B coefficients
        // DC term: color = SH_C0 * sh_dc + 0.5
        // So: sh_dc = (color - 0.5) / SH_C0
        
        // Initialize all to zero first
        for (int i = 0; i < 12; i++) {
            g.sh[i] = 0.0f;
        }
        
        // Set DC terms (indices 0, 4, 8)
        g.sh[0] = (pt.color.x - 0.5f) / SH_C0;  // R DC
        g.sh[4] = (pt.color.y - 0.5f) / SH_C0;  // G DC
        g.sh[8] = (pt.color.z - 0.5f) / SH_C0;  // B DC
        
        gaussians.push_back(g);
    }
    
    std::cout << "Created " << gaussians.size() << " Gaussians from COLMAP points" << std::endl;
    
    return gaussians;
}


int main() {
    printf("sizeof(ProjectedGaussian): %zu\n", sizeof(ProjectedGaussian));
    ColmapData colmap = loadColmap("/Users/colintaylortaylor/Documents/GuassianSplatting/GuassianSplatting/scenes/sparse/0/");
    
    auto gaussians = gaussiansFromColmap(colmap);
    
    // Debug struct layout
    printf("=== Struct Layout ===\n");
    printf("sizeof(Gaussian): %zu\n", sizeof(Gaussian));
    printf("offsetof position: %zu\n", offsetof(Gaussian, position));
    printf("offsetof scale: %zu\n", offsetof(Gaussian, scale));
    printf("offsetof rotation: %zu\n", offsetof(Gaussian, rotation));
    printf("offsetof opacity: %zu\n", offsetof(Gaussian, opacity));
    printf("offsetof sh: %zu\n", offsetof(Gaussian, sh));
    
    // Debug SH values
    if (!gaussians.empty()) {
        printf("\n=== Sample Gaussian SH values ===\n");
        printf("SH[0] (R DC): %.4f\n", gaussians[0].sh[0]);
        printf("SH[4] (G DC): %.4f\n", gaussians[0].sh[4]);
        printf("SH[8] (B DC): %.4f\n", gaussians[0].sh[8]);
        
        // Verify color recovery
        const float SH_C0 = 0.28209479177387814f;
        float r = SH_C0 * gaussians[0].sh[0] + 0.5f;
        float g = SH_C0 * gaussians[0].sh[4] + 0.5f;
        float b = SH_C0 * gaussians[0].sh[8] + 0.5f;
        printf("Recovered color: (%.4f, %.4f, %.4f)\n", r, g, b);
    }

    // Compute bounding box
    float min_x = FLT_MAX, min_y = FLT_MAX, min_z = FLT_MAX;
    float max_x = -FLT_MAX, max_y = -FLT_MAX, max_z = -FLT_MAX;

    for (const Gaussian& g : gaussians) {
        min_x = fmin(min_x, g.position.x);
        max_x = fmax(max_x, g.position.x);
        
        min_y = fmin(min_y, g.position.y);
        max_y = fmax(max_y, g.position.y);
        
        min_z = fmin(min_z, g.position.z);
        max_z = fmax(max_z, g.position.z);
    }
    
    simd_float3 center = simd_make_float3(
        (min_x + max_x) / 2.0f,
        (min_y + max_y) / 2.0f,
        (min_z + max_z) / 2.0f
    );
    float diagonal = simd_length(simd_make_float3(max_x - min_x, max_y - min_y, max_z - min_z));
    
    printf("\n=== Scene Bounds ===\n");
    printf("Bounding box: (%.3f, %.3f, %.3f) to (%.3f, %.3f, %.3f)\n",
           min_x, min_y, min_z, max_x, max_y, max_z);
    printf("Center: (%.3f, %.3f, %.3f)\n", center.x, center.y, center.z);
    printf("Diagonal: %.3f\n", diagonal);
    
    Camera camera = Camera(center, 0, 0.3f, 1.5f * diagonal,
                           45.0f * M_PI / 180.0f, 800.0f / 600.0f,
                           0.1f, 10.0f * diagonal);
    
    printf("\n=== Camera Setup ===\n");
    printf("Camera position: (%.3f, %.3f, %.3f)\n",
           camera.get_position().x,
           camera.get_position().y,
           camera.get_position().z);
    printf("Target: (%.3f, %.3f, %.3f)\n", center.x, center.y, center.z);
    printf("Distance: %.3f\n", 1.5f * diagonal);

    MTLEngine engine;
    engine.initHeadless();
    engine.loadTrainingData(colmap, "/Users/colintaylortaylor/Documents/GuassianSplatting/GuassianSplatting/scenes/images");
    engine.loadGaussians(gaussians);
    
    printf("\n=== Starting Training ===\n");
    engine.train(10);
    
    engine.cleanup();
    return 0;
}

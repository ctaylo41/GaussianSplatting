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
#include "ply_exporter.hpp"

std::vector<Gaussian> gaussiansFromColmap(const ColmapData& colmap) {
    std::vector<Gaussian> gaussians;
    gaussians.reserve(colmap.points.size());
    
    // SH_C0 constant for DC term
    const float SH_C0 = 0.28209479177387814f;
    
    for (const auto& pt : colmap.points) {
        Gaussian g;
        
        g.position = pt.position;
        
        // Initial scale in LOG space: log(0.018) ≈ -4
        g.scale = simd_make_float3(-4.0f, -4.0f, -4.0f);
        
        // Identity quaternion: (w=1, x=0, y=0, z=0)
        // Stored as float4(.x=w, .y=x, .z=y, .w=z)
        g.rotation = simd_make_float4(1.0f, 0.0f, 0.0f, 0.0f);
        
        // Initial opacity in RAW space: sigmoid(-2) ≈ 0.12
        g.opacity = -2.0f;
        
        // Initialize SH coefficients
        for (int i = 0; i < 12; i++) {
            g.sh[i] = 0.0f;
        }
        
        // Set DC terms (indices 0, 4, 8)
        // color = SH_C0 * sh_dc + 0.5, so sh_dc = (color - 0.5) / SH_C0
        g.sh[0] = (pt.color.x - 0.5f) / SH_C0;  // R DC
        g.sh[4] = (pt.color.y - 0.5f) / SH_C0;  // G DC
        g.sh[8] = (pt.color.z - 0.5f) / SH_C0;  // B DC
        
        gaussians.push_back(g);
    }
    
    std::cout << "Created " << gaussians.size() << " Gaussians from COLMAP points" << std::endl;
    
    return gaussians;
}


int main(int argc, char* argv[]) {
    
    // Default paths - can be overridden with command line args
    std::string colmapPath = "/Users/colintaylortaylor/Documents/GuassianSplatting/GuassianSplatting/scenes/sparse/0/";
    std::string imagePath = "/Users/colintaylortaylor/Documents/GuassianSplatting/GuassianSplatting/scenes/images_4";
    std::string outputPath = "/Users/colintaylortaylor/Documents/GuassianSplatting/GuassianSplatting/output.ply";
    size_t numEpochs = 3;
    bool viewOnly = false;
    std::string viewPlyPath = "";
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--colmap" && i + 1 < argc) {
            colmapPath = argv[++i];
        } else if (arg == "--images" && i + 1 < argc) {
            imagePath = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (arg == "--epochs" && i + 1 < argc) {
            numEpochs = std::stoi(argv[++i]);
        } else if (arg == "--view" && i + 1 < argc) {
            viewOnly = true;
            viewPlyPath = argv[++i];
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --colmap PATH   Path to COLMAP sparse reconstruction\n"
                      << "  --images PATH   Path to training images\n"
                      << "  --output PATH   Output PLY file path\n"
                      << "  --epochs N      Number of training epochs (default: 10)\n"
                      << "  --view PATH     View-only mode: load and display PLY file\n"
                      << "  --help          Show this help message\n";
            return 0;
        }
    }
    
    // View-only mode
    if (viewOnly) {
        std::cout << "=== View Mode ===" << std::endl;
        std::cout << "Loading: " << viewPlyPath << std::endl;
        
        auto gaussians = load_ply(viewPlyPath);
        if (gaussians.empty()) {
            std::cerr << "Error: Failed to load PLY file!" << std::endl;
            return 1;
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
        
        std::cout << "Scene center: (" << center.x << ", " << center.y << ", " << center.z << ")" << std::endl;
        std::cout << "Scene diagonal: " << diagonal << std::endl;
        
        Camera camera = Camera(center, 0, 0.3f, 1.5f * diagonal,
                               45.0f * M_PI / 180.0f, 800.0f / 600.0f,
                               0.1f, 10.0f * diagonal);
        
        MTLEngine engine;
        engine.init();
        engine.loadGaussians(gaussians);
        
        std::cout << "\nControls:" << std::endl;
        std::cout << "  Left mouse drag: Orbit camera" << std::endl;
        std::cout << "  Right mouse drag: Pan camera" << std::endl;
        std::cout << "  Scroll: Zoom in/out" << std::endl;
        std::cout << "  ESC: Exit" << std::endl;
        
        engine.run(camera);
        engine.cleanup();
        return 0;
    }
    
    // Training mode
    std::cout << "=== Gaussian Splatting Training ===" << std::endl;
    std::cout << "COLMAP path: " << colmapPath << std::endl;
    std::cout << "Images path: " << imagePath << std::endl;
    std::cout << "Output PLY: " << outputPath << std::endl;
    std::cout << "Epochs: " << numEpochs << std::endl;
    std::cout << std::endl;
    
    ColmapData colmap = loadColmap(colmapPath);
    
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
    engine.loadTrainingData(colmap, imagePath);
    engine.loadGaussians(gaussians);
    
    printf("\n=== Starting Training ===\n");
    engine.train(numEpochs);
    
    // Export trained Gaussians to PLY
    printf("\n=== Exporting PLY ===\n");
    const Gaussian* trainedGaussians = engine.getGaussians();
    size_t gaussianCount = engine.getGaussianCount();
    
    if (trainedGaussians && gaussianCount > 0) {
        PLYExporter::exportPLY(outputPath, trainedGaussians, gaussianCount);
    } else {
        std::cerr << "Error: No Gaussians to export!" << std::endl;
        engine.cleanup();
        return 1;
    }
    
    engine.cleanup();
    
    // Now open viewer with the exported PLY
    printf("\n=== Starting Viewer ===\n");
    printf("Controls:\n");
    printf("  Left mouse drag: Orbit camera\n");
    printf("  Right mouse drag: Pan camera\n");
    printf("  Scroll: Zoom in/out\n");
    printf("  ESC: Exit\n\n");
    
    // Load the exported PLY
    auto loadedGaussians = load_ply(outputPath);
    if (loadedGaussians.empty()) {
        std::cerr << "Error: Failed to load exported PLY!" << std::endl;
        return 1;
    }
    
    std::cout << "Loaded " << loadedGaussians.size() << " Gaussians for viewing" << std::endl;
    
    // Recompute bounds for loaded Gaussians
    min_x = FLT_MAX; min_y = FLT_MAX; min_z = FLT_MAX;
    max_x = -FLT_MAX; max_y = -FLT_MAX; max_z = -FLT_MAX;
    
    for (const Gaussian& g : loadedGaussians) {
        min_x = fmin(min_x, g.position.x);
        max_x = fmax(max_x, g.position.x);
        min_y = fmin(min_y, g.position.y);
        max_y = fmax(max_y, g.position.y);
        min_z = fmin(min_z, g.position.z);
        max_z = fmax(max_z, g.position.z);
    }
    
    center = simd_make_float3(
        (min_x + max_x) / 2.0f,
        (min_y + max_y) / 2.0f,
        (min_z + max_z) / 2.0f
    );
    diagonal = simd_length(simd_make_float3(max_x - min_x, max_y - min_y, max_z - min_z));
    
    std::cout << "Viewer scene center: (" << center.x << ", " << center.y << ", " << center.z << ")" << std::endl;
    std::cout << "Viewer scene diagonal: " << diagonal << std::endl;
    
    Camera viewerCamera = Camera(center, 0, 0.3f, 1.5f * diagonal,
                                  45.0f * M_PI / 180.0f, 800.0f / 600.0f,
                                  0.1f, 10.0f * diagonal);
    
    MTLEngine viewerEngine;
    viewerEngine.init();
    viewerEngine.loadGaussians(loadedGaussians);
    viewerEngine.run(viewerCamera);
    viewerEngine.cleanup();
    
    return 0;
}

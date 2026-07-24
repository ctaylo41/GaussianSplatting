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
#include <algorithm>
#include <cmath>
#include <queue>
#include <vector>
#include <dispatch/dispatch.h>

// k-d tree over the COLMAP point cloud for k-nearest-neighbour queries.
//
// A brute-force scan is O(N^2), which is why this used to be sampled down to one global
// median. A uniform grid is not a safe replacement either: COLMAP reconstructions contain
// far-flung outlier points that stretch the bounding box (the bicycle scene spans 201
// units diagonally while nearly every point sits within ~10 of the centre), so uniform
// cells put almost the whole cloud in a handful of buckets and degrade back to O(N^2).
// A k-d tree splits on the actual point distribution, so it is insensitive to that skew.
struct KDTree {
    struct Node {
        int lo, hi;         // range in `order`, used only for leaves
        int axis;           // split axis, -1 for a leaf
        float split;        // split coordinate
        int left, right;    // child node indices, -1 for a leaf
    };

    static constexpr int LEAF_SIZE = 16;

    const std::vector<ColmapPoint>& points;
    std::vector<uint32_t> order;   // permutation of point indices
    std::vector<Node> nodes;

    explicit KDTree(const std::vector<ColmapPoint>& pts) : points(pts) {
        order.resize(pts.size());
        for (uint32_t i = 0; i < pts.size(); i++) order[i] = i;
        nodes.reserve(2 * (pts.size() / LEAF_SIZE + 1));
        build(0, (int)pts.size());
    }

    // Returns the index of the node covering order[lo, hi)
    int build(int lo, int hi) {
        int nodeIdx = (int)nodes.size();
        nodes.push_back({lo, hi, -1, 0.0f, -1, -1});

        if (hi - lo <= LEAF_SIZE) return nodeIdx;

        // Split on the axis with the widest spread within this range
        simd_float3 mn = points[order[lo]].position;
        simd_float3 mx = mn;
        for (int i = lo + 1; i < hi; i++) {
            mn = simd_min(mn, points[order[i]].position);
            mx = simd_max(mx, points[order[i]].position);
        }
        simd_float3 span = mx - mn;
        int axis = (span.x > span.y) ? ((span.x > span.z) ? 0 : 2)
                                     : ((span.y > span.z) ? 1 : 2);

        // Degenerate range (all points coincident) - keep it as a leaf
        if (span[axis] <= 0.0f) return nodeIdx;

        int mid = lo + (hi - lo) / 2;
        std::nth_element(order.begin() + lo, order.begin() + mid, order.begin() + hi,
                         [&](uint32_t a, uint32_t b) {
                             return points[a].position[axis] < points[b].position[axis];
                         });

        nodes[nodeIdx].axis = axis;
        nodes[nodeIdx].split = points[order[mid]].position[axis];

        int left = build(lo, mid);
        int right = build(mid, hi);
        // build() can reallocate `nodes`, so assign through a fresh reference
        nodes[nodeIdx].left = left;
        nodes[nodeIdx].right = right;
        return nodeIdx;
    }

    // k-NN search. `heap` is a max-heap of the k smallest squared distances found so far.
    void search(int nodeIdx, simd_float3 q, size_t selfIdx, int k,
                std::priority_queue<float>& heap) const {
        const Node& n = nodes[nodeIdx];

        if (n.axis < 0) {
            for (int i = n.lo; i < n.hi; i++) {
                uint32_t idx = order[i];
                if (idx == selfIdx) continue;
                simd_float3 d = points[idx].position - q;
                float d2 = simd_dot(d, d);
                if ((int)heap.size() < k) {
                    heap.push(d2);
                } else if (d2 < heap.top()) {
                    heap.pop();
                    heap.push(d2);
                }
            }
            return;
        }

        // Descend the near side first so the pruning bound tightens as early as possible
        float delta = q[n.axis] - n.split;
        int near = (delta < 0.0f) ? n.left : n.right;
        int far  = (delta < 0.0f) ? n.right : n.left;

        search(near, q, selfIdx, k, heap);
        if ((int)heap.size() < k || delta * delta < heap.top()) {
            search(far, q, selfIdx, k, heap);
        }
    }
};

// Compute Mean Nearest Neighbor Distance for a point. Exact k-NN via the k-d tree.
float computeMeanNearestNeighborDistance(const KDTree& tree,
                                          size_t pointIdx,
                                          int k = 3) {
    std::priority_queue<float> heap;  // squared distances
    tree.search(0, tree.points[pointIdx].position, pointIdx, k, heap);

    // Mean of the k nearest distances
    float sum = 0.0f;
    int count = 0;
    while (!heap.empty()) {
        sum += sqrtf(heap.top());
        heap.pop();
        count++;
    }

    // Default if no neighbors
    return (count > 0) ? (sum / count) : 0.1f;
}

// Create Gaussians from COLMAP points
std::vector<Gaussian> gaussiansFromColmap(const ColmapData& colmap, float sceneExtent) {
    std::vector<Gaussian> gaussians;
    gaussians.reserve(colmap.points.size());
    
    // SH_C0 constant for DC term
    const float SH_C0 = 0.28209479177387814f;
    
    // Scene extent is now passed in that was computed from camera positions
    std::cout << "Using scene extent: " << sceneExtent << std::endl;
    
    // Print out first 5 camera world positions for debugging
    std::cout << "\n=== Camera Position Debug ===" << std::endl;
    for (int i = 0; i < std::min(5, (int)colmap.images.size()); i++) {
        simd_float3 pos = getCameraWorldPosition(colmap.images[i]);
        std::cout << "Camera " << i << " world pos: ("
                  << pos.x << ", " << pos.y << ", " << pos.z << ")" << std::endl;
        std::cout << "  Raw translation: ("
                  << colmap.images[i].translation.x << ", "
                  << colmap.images[i].translation.y << ", "
                  << colmap.images[i].translation.z << ")" << std::endl;
    }

    std::cout << "Computing initial scales from nearest neighbor distances..." << std::endl;
    
    // Precompute initial scales from each point's own nearest-neighbour distance.
    // Assigning one global median to every point (the previous behaviour) makes Gaussians
    // in dense, detailed regions far too large and destroys high-frequency detail.
    std::vector<float> initialScales(colmap.points.size());

    KDTree tree(colmap.points);

    // Blocks capture C++ objects by const copy, so hand the parallel loop raw pointers.
    // Dispatch a fixed number of stripes rather than one block per point: 240k blocks of
    // a few microseconds each is mostly dispatch overhead.
    const KDTree* treePtr = &tree;
    float* scalesPtr = initialScales.data();
    const size_t numPoints = colmap.points.size();
    const size_t numStripes = 8;
    const size_t stripeSize = (numPoints + numStripes - 1) / numStripes;

    dispatch_apply(numStripes, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t s) {
        size_t begin = s * stripeSize;
        size_t end = std::min(begin + stripeSize, numPoints);
        for (size_t i = begin; i < end; i++) {
            scalesPtr[i] = computeMeanNearestNeighborDistance(*treePtr, i, 3);
        }
    });

    // Report the spread so a degenerate initialization is visible in the logs
    {
        std::vector<float> sorted = initialScales;
        std::sort(sorted.begin(), sorted.end());
        std::cout << "Per-point nearest neighbor distance: min=" << sorted.front()
                  << " p50=" << sorted[sorted.size() / 2]
                  << " max=" << sorted.back() << std::endl;
    }
    
    // Create Gaussians using initial scales
    for (size_t i = 0; i < colmap.points.size(); i++) {
        const auto& pt = colmap.points[i];
        Gaussian g;
        
        g.position = pt.position;
        
        // Get initial scale
        float scale = initialScales[i];

        // Apply a global scaling factor to better fit the scene and ensure gradients flow at the start
        scale *= 0.7f;

        // Clamp scale to reasonable range relative to scene
        float minScale = 0.0001f * sceneExtent;
        float maxScale = 0.1f * sceneExtent;
        scale = std::clamp(scale, minScale, maxScale);
        
        // Convert to log space
        float logScale = std::log(scale);
        g.scale = simd_make_float3(logScale, logScale, logScale);
        
        // Identity quaternion (w=1, x=0, y=0, z=0)
        // Stored as float4(.x=w, .y=x, .z=y, .w=z)
        g.rotation = simd_make_float4(1.0f, 0.0f, 0.0f, 0.0f);
        
        // Initial opacity in raw space: sigmoid(0) = 0.5
        // Start with visible opacity so gradients can flow
        g.opacity = 0.0f;
        
        // Initialize SH coefficients
        for (int j = 0; j < 12; j++) {
            g.sh[j] = 0.0f;
        }

        // Set DC terms indices 0, 4, 8
        // Official 3DGS formula: color = SH_C0 * dc + 0.5
        // So dc = (color - 0.5) / SH_C0 (RGB2SH conversion)
        auto RGB2SH = [SH_C0](float c) -> float {
            return (c - 0.5f) / SH_C0;
        };
        g.sh[0] = RGB2SH(pt.color.x);
        g.sh[4] = RGB2SH(pt.color.y);
        g.sh[8] = RGB2SH(pt.color.z);
        
        gaussians.push_back(g);
    }
    
    std::cout << "Created " << gaussians.size() << " Gaussians from COLMAP points" << std::endl;
    
    // Print scale statistics
    float minLogScale = FLT_MAX, maxLogScale = -FLT_MAX, avgLogScale = 0;
    for (const auto& g : gaussians) {
        minLogScale = fmin(minLogScale, g.scale.x);
        maxLogScale = fmax(maxLogScale, g.scale.x);
        avgLogScale += g.scale.x;
    }
    avgLogScale /= gaussians.size();
    
    std::cout << "Scale stats (log): min=" << minLogScale
              << " max=" << maxLogScale
              << " avg=" << avgLogScale << std::endl;
    std::cout << "Scale stats (world): min=" << expf(minLogScale)
              << " max=" << expf(maxLogScale)
              << " avg=" << expf(avgLogScale) << std::endl;
    
    return gaussians;
}





int main(int argc, char* argv[]) {
    // Default paths can be overridden with command line args
    std::string colmapPath = "/Users/colintaylortaylor/Documents/GaussianSplatting/GaussianSplatting/scenes/sparse/0";
    std::string imagePath = "/Users/colintaylortaylor/Documents/GaussianSplatting/GaussianSplatting/scenes/images_4";
    std::string outputPath = "/Users/colintaylortaylor/Documents/GaussianSplatting/GaussianSplatting/output_bike.ply";
    // ~30k iterations for the 194-image bicycle scene (194 * 155 ≈ 30,070)
    size_t numEpochs = 155;
    bool viewOnly = false;
    std::string viewPlyPath = "/Users/colintaylortaylor/Documents/GaussianSplatting/GaussianSplatting/output_bike.ply";
    
    // Parse command line arguments
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
    
    // View only mode
    if (viewOnly) {
        std::cout << "=== View Mode ===" << std::endl;
        std::cout << "Loading: " << viewPlyPath << std::endl;
        
        // Load Gaussians from PLY
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
        // Diagonal size
        float diagonal = simd_length(simd_make_float3(max_x - min_x, max_y - min_y, max_z - min_z));
        
        std::cout << "Scene center: (" << center.x << ", " << center.y << ", " << center.z << ")" << std::endl;
        std::cout << "Scene diagonal: " << diagonal << std::endl;
        
        // Start closer for better initial view, with slight downward angle
        // At least 1 unit away
        float viewDistance = std::max(0.5f * diagonal, 1.0f);  
         // Closer near plane for small scenes
        float nearPlane = std::max(0.01f, diagonal * 0.001f); 
        // Far enough to see everything
        float farPlane = std::max(100.0f, diagonal * 10.0f);   
        
        // Print camera setup
        std::cout << "Initial view distance: " << viewDistance << std::endl;
        std::cout << "Near/Far planes: " << nearPlane << " / " << farPlane << std::endl;
        
        Camera camera = Camera(center, 0, 0.4f, viewDistance,
                               45.0f * M_PI / 180.0f, 800.0f / 600.0f,
                               nearPlane, farPlane);
        
        // Initialize and run engine
        MTLEngine engine;
        engine.init();
        engine.loadGaussians(gaussians, diagonal);
        
        std::cout << "\nControls:" << std::endl;
        std::cout << "  Left mouse drag: Orbit camera" << std::endl;
        std::cout << "  Right mouse drag: Pan camera" << std::endl;
        std::cout << "  Scroll: Zoom in/out" << std::endl;
        std::cout << "  Note: Training view (T key) not available without COLMAP data" << std::endl;
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
    
    // Load COLMAP data
    ColmapData colmap = loadColmap(colmapPath);
    
    // Compute scene extent from camera positions before creating gaussians
    float sceneExtent = computeSceneExtent(colmap);
    std::cout << "Scene extent (from cameras): " << sceneExtent << std::endl;

    // Create Gaussians from COLMAP points
    auto gaussians = gaussiansFromColmap(colmap,sceneExtent);
    
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

        // Verify color recovery using official 3DGS formula: color = SH_C0 * dc + 0.5
        const float SH_C0 = 0.28209479177387814f;
        float r = SH_C0 * gaussians[0].sh[0] + 0.5f;
        float g = SH_C0 * gaussians[0].sh[4] + 0.5f;
        float b = SH_C0 * gaussians[0].sh[8] + 0.5f;
        printf("Recovered color (SH_C0*dc+0.5): (%.4f, %.4f, %.4f)\n", r, g, b);
    }

    // Compute bounding box
    float min_x = FLT_MAX, min_y = FLT_MAX, min_z = FLT_MAX;
    float max_x = -FLT_MAX, max_y = -FLT_MAX, max_z = -FLT_MAX;

    // Iterate through Gaussians to find bounds
    for (const Gaussian& g : gaussians) {
        min_x = fmin(min_x, g.position.x);
        max_x = fmax(max_x, g.position.x);
        
        min_y = fmin(min_y, g.position.y);
        max_y = fmax(max_y, g.position.y);
        
        min_z = fmin(min_z, g.position.z);
        max_z = fmax(max_z, g.position.z);
    }
    
    // Compute center and diagonal
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

    // Initialize engine for training
    MTLEngine engine;
    engine.initHeadless();
    // Load training data
    engine.loadTrainingData(colmap, imagePath);
    // Load Gaussians
    engine.loadGaussians(gaussians, sceneExtent);
    
    printf("\n=== Starting Training ===\n");
    engine.train(numEpochs);
    
    // Export rendered views from each training camera
    std::string rendersFolder = outputPath;
    // Replace filename with renders folder
    size_t lastSlash = rendersFolder.rfind('/');
    if (lastSlash != std::string::npos) {
        rendersFolder = rendersFolder.substr(0, lastSlash) + "/renders";
    } else {
        rendersFolder = "renders";
    }
    // Export training views
    engine.exportTrainingViews(rendersFolder);
    
    // Export trained Gaussians to PLY
    printf("\n=== Exporting PLY ===\n");
    const Gaussian* trainedGaussians = engine.getGaussians();
    size_t gaussianCount = engine.getGaussianCount();
    
    // Export only if we have Gaussians
    if (trainedGaussians && gaussianCount > 0) {
        PLYExporter::exportPLY(outputPath, trainedGaussians, gaussianCount);
    } else {
        std::cerr << "Error: No Gaussians to export!" << std::endl;
        engine.cleanup();
        return 1;
    }
    
    // Cleanup engine
    engine.cleanup();
    
    // Now open viewer with the exported PLY
    printf("\n=== Starting Viewer ===\n");
    printf("Controls:\n");
    printf("  Left mouse drag: Orbit camera\n");
    printf("  Right mouse drag: Pan camera\n");
    printf("  Scroll: Zoom in/out\n");
    printf("  T: Toggle training view (snap to training camera positions)\n");
    printf("  Left/Right arrows: Navigate training images (when in training view)\n");
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
    
    // Iterate through Gaussians to find bounds
    for (const Gaussian& g : loadedGaussians) {
        min_x = fmin(min_x, g.position.x);
        max_x = fmax(max_x, g.position.x);
        min_y = fmin(min_y, g.position.y);
        max_y = fmax(max_y, g.position.y);
        min_z = fmin(min_z, g.position.z);
        max_z = fmax(max_z, g.position.z);
    }
    
    // Compute center and diagonal
    center = simd_make_float3(
        (min_x + max_x) / 2.0f,
        (min_y + max_y) / 2.0f,
        (min_z + max_z) / 2.0f
    );
    diagonal = simd_length(simd_make_float3(max_x - min_x, max_y - min_y, max_z - min_z));
    
    std::cout << "Viewer scene center: (" << center.x << ", " << center.y << ", " << center.z << ")" << std::endl;
    std::cout << "Viewer scene diagonal: " << diagonal << std::endl;
    
    // Use distance based on scene size and position camera closer to see details
    float viewDistance = std::max(0.5f * diagonal, 1.0f);
    float nearPlane = std::max(0.01f, diagonal * 0.001f);
    float farPlane = std::max(100.0f, diagonal * 10.0f);
    std::cout << "Viewer distance: " << viewDistance << std::endl;
    std::cout << "Near/Far planes: " << nearPlane << " / " << farPlane << std::endl;
    
    // Position camera to look at the actual scene center
    simd_float3 viewTarget = center;
    
    // Create viewer camera
    Camera viewerCamera = Camera(viewTarget, 0, 0.4f, viewDistance,
                                  45.0f * M_PI / 180.0f, 800.0f / 600.0f,
                                  nearPlane, farPlane);
    
    // Initialize and run viewer engine
    MTLEngine viewerEngine;
    viewerEngine.init();
    // Load training data so T key and arrow keys work for navigating training views
    viewerEngine.loadTrainingData(colmap, imagePath);
    viewerEngine.loadGaussians(loadedGaussians, sceneExtent);
    viewerEngine.run(viewerCamera);
    viewerEngine.cleanup();
    
    return 0;
}

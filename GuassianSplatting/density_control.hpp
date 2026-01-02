//
//  density_control.hpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-28.
//

#pragma once
#include <Metal/Metal.hpp>
#include "ply_loader.hpp"

struct DensityStats {
    uint32_t numPruned;
    uint32_t numCloned;
    uint32_t numSplit;
};

class DensityController {
public:
    DensityController(MTL::Device* device, MTL::Library* library);
    ~DensityController();
    
    DensityStats apply(MTL::CommandQueue* queue,
                       MTL::Buffer*& gaussianBuffer,
                       MTL::Buffer*& positionBuffer,
                       MTL::Buffer* gradientAccum,
                       size_t& gaussianCount,
                       size_t iteration,
                       float gradThreshold = 0.0002f,
                       float minOpacity = 0.005f,
                       float maxScale = 0.5f,
                       float focalLength = 500.0f,
                       float imageWidth = 800.0f,
                       float avgDepth = 5.0f);
    
    void accumulateGradients(MTL::CommandQueue* queue,
                             MTL::Buffer* gradients,
                             size_t gaussianCount);
    
    void resetAccumulator(size_t gaussianCount);
    
    // Set scene extent for scene-relative thresholds (call before training)
    static void setSceneExtent(float extent);

private:
    MTL::Device* device;
    
    MTL::Buffer* gradientAccum;
    MTL::Buffer* gradientCount;
    MTL::Buffer* markerBuffer;
    
    // NEW: Store position gradients for gradient-directed cloning
    MTL::Buffer* positionGradAccum;
    
    size_t maxGaussians;
};

//
//  density_control.mm
//  GuassianSplatting
//
//  Adaptive density control for Gaussian splatting training
//
//  IMPORTANT: Scale is in LOG space internally!
//  - To get actual scale: exp(g.scale)
//  - When splitting, new log scale = old log scale + log(scaleFactor)
//

#include "density_control.hpp"
#include <iostream>
#include <cmath>
#include "gradients.hpp"

// Paper's recommended thresholds (from official 3DGS)
static constexpr float GRAD_THRESHOLD = 0.0002f;      // densify_grad_threshold
static constexpr float OPACITY_PRUNE_THRESHOLD = 0.005f;  // min_opacity
static constexpr float PERCENT_DENSE = 0.01f;         // percent_dense for clone vs split
static constexpr size_t MAX_GAUSSIANS = 500000;
static constexpr size_t DENSIFY_STOP_ITER = 15000;

// Scene extent - will be set during initialization
static float sceneExtent = 1.0f;  // Default, should be set to scene diagonal

void DensityController::setSceneExtent(float extent) {
    sceneExtent = extent;
    std::cout << "Density control scene extent set to: " << extent << std::endl;
    std::cout << "  Split threshold: " << (PERCENT_DENSE * extent) << " world units" << std::endl;
    std::cout << "  Prune threshold: " << (0.1f * extent) << " world units" << std::endl;
}

DensityController::DensityController(MTL::Device* device, MTL::Library* library)
    : device(device)
    , maxGaussians(MAX_GAUSSIANS)
{
    gradientAccum = device->newBuffer(maxGaussians * sizeof(float), MTL::ResourceStorageModeShared);
    gradientCount = device->newBuffer(maxGaussians * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    markerBuffer = device->newBuffer(maxGaussians * sizeof(uint32_t), MTL::ResourceStorageModeShared);
}

DensityController::~DensityController() {
    gradientAccum->release();
    gradientCount->release();
    markerBuffer->release();
}

void DensityController::resetAccumulator(size_t gaussianCount) {
    memset(gradientAccum->contents(), 0, gaussianCount * sizeof(float));
    memset(gradientCount->contents(), 0, gaussianCount * sizeof(uint32_t));
}

void DensityController::accumulateGradients(MTL::CommandQueue *queue, MTL::Buffer *gradients, size_t gaussianCount) {
    GaussianGradients* grads = (GaussianGradients*)gradients->contents();
    float* accum = (float*)gradientAccum->contents();
    uint32_t* count = (uint32_t*)gradientCount->contents();
    
    for (size_t i = 0; i < gaussianCount; i++) {
        float gradMag = sqrtf(grads[i].position_x * grads[i].position_x +
                              grads[i].position_y * grads[i].position_y +
                              grads[i].position_z * grads[i].position_z);
        
        if (!std::isnan(gradMag) && !std::isinf(gradMag)) {
            accum[i] += gradMag;
            count[i]++;
        }
    }
}

DensityStats DensityController::apply(MTL::CommandQueue* queue,
                                      MTL::Buffer*& gaussianBuffer,
                                      MTL::Buffer*& positionBuffer,
                                      MTL::Buffer* gradientAccumBuffer,
                                      size_t& gaussianCount,
                                      size_t iteration,
                                      float gradThreshold,
                                      float minOpacity,
                                      float maxScale) {
    DensityStats stats = {0, 0, 0};
    
    bool canDensify = (iteration < DENSIFY_STOP_ITER) && (gaussianCount < MAX_GAUSSIANS);
    
    Gaussian* gaussians = (Gaussian*)gaussianBuffer->contents();
    float* accumGrad = (float*)gradientAccum->contents();
    uint32_t* counts = (uint32_t*)gradientCount->contents();
    uint32_t* markers = (uint32_t*)markerBuffer->contents();
    
    const float MAX_SCALE_LOG = 10.0f;  // Must match tiled_shaders.metal MAX_SCALE
    
    // First pass: mark Gaussians
    for (size_t i = 0; i < gaussianCount; i++) {
        Gaussian& g = gaussians[i];
        
        // Prune corrupted Gaussians
        if (std::isnan(g.position.x) || std::isnan(g.position.y) || std::isnan(g.position.z) ||
            std::isinf(g.position.x) || std::isinf(g.position.y) || std::isinf(g.position.z) ||
            std::abs(g.position.x) > 1e6 || std::abs(g.position.y) > 1e6 || std::abs(g.position.z) > 1e6) {
            markers[i] = 1;  // Prune
            stats.numPruned++;
            continue;
        }
        
        // Compute opacity after sigmoid (scale is log, opacity is raw)
        float rawOp = std::clamp(g.opacity, -8.0f, 8.0f);
        float opacity = 1.0f / (1.0f + expf(-rawOp));
        
        float avgGrad = (counts[i] > 0) ? (accumGrad[i] / counts[i]) : 0.0f;
        
        // Compute max scale in WORLD units (apply exp to log scale)
        float maxScaleVal = fmaxf(fmaxf(
            expf(std::clamp(g.scale.x, -MAX_SCALE_LOG, MAX_SCALE_LOG)),
            expf(std::clamp(g.scale.y, -MAX_SCALE_LOG, MAX_SCALE_LOG))),
            expf(std::clamp(g.scale.z, -MAX_SCALE_LOG, MAX_SCALE_LOG)));
        
        // Official 3DGS thresholds (scene-relative)
        float splitThreshold = PERCENT_DENSE * sceneExtent;  // Clone small, split large
        float pruneThreshold = 0.1f * sceneExtent;           // Prune extremely large Gaussians
        
        // Decision logic (matches official 3DGS densify_and_prune)
        bool shouldPrune = (opacity < OPACITY_PRUNE_THRESHOLD);
        
        // Also prune Gaussians that are too large in world space
        if (maxScaleVal > pruneThreshold) {
            shouldPrune = true;
        }
        
        if (shouldPrune) {
            markers[i] = 1;
            stats.numPruned++;
        } else if (canDensify && avgGrad > GRAD_THRESHOLD) {
            // Official logic: clone if small, split if large
            if (maxScaleVal > splitThreshold) {
                markers[i] = 3;  // Split large Gaussians
                stats.numSplit++;
            } else {
                markers[i] = 2;  // Clone small Gaussians
                stats.numCloned++;
            }
        } else {
            markers[i] = 0;  // Keep
        }
    }
    
    // Check capacity
    size_t newCount = gaussianCount - stats.numPruned + stats.numCloned + stats.numSplit;
    
    if (newCount > MAX_GAUSSIANS) {
        size_t excess = newCount - MAX_GAUSSIANS;
        size_t reduced = 0;
        
        for (size_t i = 0; i < gaussianCount && reduced < excess; i++) {
            if (markers[i] == 2) {
                markers[i] = 0;
                stats.numCloned--;
                reduced++;
            }
        }
        
        for (size_t i = 0; i < gaussianCount && reduced < excess; i++) {
            if (markers[i] == 3) {
                markers[i] = 0;
                stats.numSplit--;
                reduced++;
            }
        }
        
        newCount = gaussianCount - stats.numPruned + stats.numCloned + stats.numSplit;
    }
    
    // Allocate new buffers
    MTL::Buffer* newGaussianBuffer = device->newBuffer(newCount * sizeof(Gaussian), MTL::ResourceStorageModeShared);
    MTL::Buffer* newPositionsBuffer = device->newBuffer(newCount * sizeof(simd_float3), MTL::ResourceStorageModeShared);
    
    Gaussian* newGaussians = (Gaussian*)newGaussianBuffer->contents();
    simd_float3* newPositions = (simd_float3*)newPositionsBuffer->contents();
    
    // Second pass: build new arrays
    size_t writeIdx = 0;
    for (size_t i = 0; i < gaussianCount; i++) {
        Gaussian& g = gaussians[i];
        uint32_t marker = markers[i];
        
        if (marker == 1) {
            continue;  // Pruned
        }
        
        if (marker == 0) {
            // Keep as-is
            newGaussians[writeIdx] = g;
            newPositions[writeIdx] = g.position;
            writeIdx++;
        } else if (marker == 2) {
            // Clone
            newGaussians[writeIdx] = g;
            newPositions[writeIdx] = g.position;
            writeIdx++;
            
            Gaussian cloned = g;
            float perturbScale = 0.001f;
            cloned.position.x += perturbScale * ((float)rand() / RAND_MAX - 0.5f);
            cloned.position.y += perturbScale * ((float)rand() / RAND_MAX - 0.5f);
            cloned.position.z += perturbScale * ((float)rand() / RAND_MAX - 0.5f);
            newGaussians[writeIdx] = cloned;
            newPositions[writeIdx] = cloned.position;
            writeIdx++;
        } else if (marker == 3) {
            // Split: create two smaller Gaussians
            float scaleFactor = 1.0f / 1.6f;  // Paper uses φ = 1.6
            float logScaleFactor = logf(scaleFactor);  // Add this to log scale
            
            // Get actual scale (apply exp to log scale)
            // Use same MAX_SCALE_LOG as the rest of the codebase
            simd_float3 scale = simd_make_float3(
                expf(std::clamp(g.scale.x, -MAX_SCALE_LOG, MAX_SCALE_LOG)),
                expf(std::clamp(g.scale.y, -MAX_SCALE_LOG, MAX_SCALE_LOG)),
                expf(std::clamp(g.scale.z, -MAX_SCALE_LOG, MAX_SCALE_LOG))
            );
            
            // Offset along major axis
            simd_float3 offset;
            if (scale.x >= scale.y && scale.x >= scale.z) {
                offset = simd_make_float3(scale.x * 0.5f, 0, 0);
            } else if (scale.y >= scale.z) {
                offset = simd_make_float3(0, scale.y * 0.5f, 0);
            } else {
                offset = simd_make_float3(0, 0, scale.z * 0.5f);
            }
            
            // Rotate offset by Gaussian's rotation
            // q.x=w, q.y=x, q.z=y, q.w=z
            simd_float4 q = g.rotation;
            float w = q.x, x = q.y, y = q.z, z = q.w;
            simd_float3x3 R = {{
                {1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)},
                {2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x)},
                {2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)}
            }};
            offset = simd_mul(R, offset);
            
            // Child 1: positive offset, smaller scale
            Gaussian child1 = g;
            child1.position = g.position + offset;
            // Scale is in LOG space - add log(scaleFactor) to make smaller
            child1.scale = simd_make_float3(
                g.scale.x + logScaleFactor,
                g.scale.y + logScaleFactor,
                g.scale.z + logScaleFactor
            );
            newGaussians[writeIdx] = child1;
            newPositions[writeIdx] = child1.position;
            writeIdx++;
            
            // Child 2: negative offset, same smaller scale
            Gaussian child2 = g;
            child2.position = g.position - offset;
            child2.scale = child1.scale;
            newGaussians[writeIdx] = child2;
            newPositions[writeIdx] = child2.position;
            writeIdx++;
        }
    }
    
    // Swap buffers
    gaussianBuffer->release();
    positionBuffer->release();
    gaussianBuffer = newGaussianBuffer;
    positionBuffer = newPositionsBuffer;
    gaussianCount = writeIdx;
    
    resetAccumulator(gaussianCount);
    
    std::cout << "Density control: pruned=" << stats.numPruned
              << " cloned=" << stats.numCloned
              << " split=" << stats.numSplit
              << " total=" << gaussianCount << std::endl;
    
    return stats;
}

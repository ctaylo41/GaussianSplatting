//
//  density_control.mm
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-28.
//

#include "density_control.hpp"
#include <iostream>
#include "gradients.hpp"

// Paper's recommended thresholds
static constexpr float GRAD_THRESHOLD = 0.0002f;          // Position gradient threshold
static constexpr float OPACITY_PRUNE_THRESHOLD = 0.005f;  // Prune if opacity < this
static constexpr float SCALE_SPLIT_THRESHOLD = 0.01f;     // Split if scale > this (world units)
static constexpr float MAX_SCREEN_SIZE = 20.0f;           // Prune if too big on screen
static constexpr size_t MAX_GAUSSIANS = 500000;           // Hard cap
static constexpr size_t DENSIFY_STOP_ITER = 15000;        // Stop densifying after this

DensityController::DensityController(MTL::Device* device, MTL::Library* library)
    : device(device)
    , maxGaussians(MAX_GAUSSIANS)
{
    NS::Error* error = nullptr;
    
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
        
        // Skip NaN/Inf gradients
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
                                      float gradThreshold,    // Ignored - using paper's value
                                      float minOpacity,       // Ignored - using paper's value
                                      float maxScale) {       // Ignored - using paper's value
    DensityStats stats = {0, 0, 0};
    
    // Stop densifying after threshold (but still prune)
    bool canDensify = (iteration < DENSIFY_STOP_ITER) && (gaussianCount < MAX_GAUSSIANS);
    
    Gaussian* gaussians = (Gaussian*)gaussianBuffer->contents();
    float* accumGrad = (float*)gradientAccum->contents();
    uint32_t* counts = (uint32_t*)gradientCount->contents();
    uint32_t* markers = (uint32_t*)markerBuffer->contents();
    
    // First pass: mark Gaussians
    for (size_t i = 0; i < gaussianCount; i++) {
        Gaussian& g = gaussians[i];
        
        // Compute opacity after sigmoid
        float rawOp = std::clamp(g.opacity, -8.0f, 8.0f);
        float opacity = 1.0f / (1.0f + expf(-rawOp));
        
        // Compute average gradient
        float avgGrad = (counts[i] > 0) ? (accumGrad[i] / counts[i]) : 0.0f;
        
        // Compute max scale in world units
        float maxScaleVal = fmaxf(fmaxf(expf(std::clamp(g.scale.x, -3.0f, 3.0f)),
                                        expf(std::clamp(g.scale.y, -3.0f, 3.0f))),
                                        expf(std::clamp(g.scale.z, -3.0f, 3.0f)));
        
        // Decision logic
        if (opacity < OPACITY_PRUNE_THRESHOLD) {
            // Prune: opacity too low
            markers[i] = 1;
            stats.numPruned++;
        } else if (canDensify && avgGrad > GRAD_THRESHOLD) {
            // High gradient - needs densification
            if (maxScaleVal > SCALE_SPLIT_THRESHOLD) {
                // Large Gaussian: split into two smaller ones
                markers[i] = 3;
                stats.numSplit++;
            } else {
                // Small Gaussian: clone it
                markers[i] = 2;
                stats.numCloned++;
            }
        } else {
            // Keep as-is
            markers[i] = 0;
        }
    }
    
    // Check if we'd exceed max
    size_t newCount = gaussianCount - stats.numPruned + stats.numCloned + stats.numSplit;
    
    if (newCount > MAX_GAUSSIANS) {
        // Scale back cloning/splitting to fit
        size_t excess = newCount - MAX_GAUSSIANS;
        size_t reduced = 0;
        
        // Remove clones first (they're less important than splits)
        for (size_t i = 0; i < gaussianCount && reduced < excess; i++) {
            if (markers[i] == 2) {
                markers[i] = 0;  // Keep original, don't clone
                stats.numCloned--;
                reduced++;
            }
        }
        
        // If still over, remove splits
        for (size_t i = 0; i < gaussianCount && reduced < excess; i++) {
            if (markers[i] == 3) {
                markers[i] = 0;  // Keep original, don't split
                stats.numSplit--;
                reduced++;
            }
        }
        
        newCount = gaussianCount - stats.numPruned + stats.numCloned + stats.numSplit;
        std::cout << "Density control: capped at " << MAX_GAUSSIANS << ", reduced " << reduced << " operations" << std::endl;
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
            // Pruned - skip
            continue;
        }
        
        if (marker == 0) {
            // Keep as-is
            newGaussians[writeIdx] = g;
            newPositions[writeIdx] = g.position;
            writeIdx++;
        } else if (marker == 2) {
            // Clone: keep original
            newGaussians[writeIdx] = g;
            newPositions[writeIdx] = g.position;
            writeIdx++;
            
            // Add clone with tiny perturbation
            Gaussian cloned = g;
            float perturbScale = 0.001f;  // Very small perturbation
            cloned.position.x += perturbScale * ((float)rand() / RAND_MAX - 0.5f);
            cloned.position.y += perturbScale * ((float)rand() / RAND_MAX - 0.5f);
            cloned.position.z += perturbScale * ((float)rand() / RAND_MAX - 0.5f);
            newGaussians[writeIdx] = cloned;
            newPositions[writeIdx] = cloned.position;
            writeIdx++;
        } else if (marker == 3) {
            // Split: replace with two smaller Gaussians
            float scaleFactor = 1.0f / 1.6f;  // Paper uses φ = 1.6
            
            // Get current scale
            simd_float3 scale = simd_make_float3(
                expf(std::clamp(g.scale.x, -3.0f, 3.0f)),
                expf(std::clamp(g.scale.y, -3.0f, 3.0f)),
                expf(std::clamp(g.scale.z, -3.0f, 3.0f))
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
            simd_float4 q = g.rotation;
            float r = q.x, x = q.y, y = q.z, z = q.w;
            simd_float3x3 R = {{
                {1 - 2*(y*y + z*z), 2*(x*y + r*z), 2*(x*z - r*y)},
                {2*(x*y - r*z), 1 - 2*(x*x + z*z), 2*(y*z + r*x)},
                {2*(x*z + r*y), 2*(y*z - r*x), 1 - 2*(x*x + y*y)}
            }};
            offset = simd_mul(R, offset);
            
            // Child 1: offset in positive direction, smaller scale
            Gaussian child1 = g;
            child1.position = g.position + offset;
            child1.scale = simd_make_float3(
                g.scale.x + logf(scaleFactor),
                g.scale.y + logf(scaleFactor),
                g.scale.z + logf(scaleFactor)
            );
            newGaussians[writeIdx] = child1;
            newPositions[writeIdx] = child1.position;
            writeIdx++;
            
            // Child 2: offset in negative direction, same smaller scale
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
    
    // Reset accumulator for new count
    resetAccumulator(gaussianCount);
    
    std::cout << "Density control: pruned=" << stats.numPruned
              << " cloned=" << stats.numCloned
              << " split=" << stats.numSplit
              << " total=" << gaussianCount << std::endl;
    
    return stats;
}

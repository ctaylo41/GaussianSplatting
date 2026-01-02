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
static constexpr float PERCENT_DENSE = 0.001f;         // percent_dense for clone vs split
static constexpr size_t MAX_GAUSSIANS = 500000;
static constexpr size_t DENSIFY_FROM_ITER = 500;      // Start densification
static constexpr size_t DENSIFY_UNTIL_ITER = 15000;   // Stop densification
static constexpr float MAX_SCALE_LOG = 4.0f;          // Clamp scale values

// Scene extent - will be set during initialization
static float sceneExtent = 1.0f;  // Default, should be set to scene diagonal

float computeApproxScreenRadius(const Gaussian& g,
                                 float focalLength,
                                 float avgDepth,
                                 float imageWidth) {
    // Get max scale in world units
    float maxScale = fmaxf(fmaxf(
        expf(std::clamp(g.scale.x, -4.0f, 4.0f)),
        expf(std::clamp(g.scale.y, -4.0f, 4.0f))),
        expf(std::clamp(g.scale.z, -4.0f, 4.0f)));
    
    // Approximate screen radius: focal * scale / depth * multiplier
    // 3 sigma covers 99% of Gaussian, so multiply by 3
    float screenRadius = focalLength * maxScale * 3.0f / avgDepth;
    
    // Normalize to fraction of image width
    return screenRadius / imageWidth;
}


void DensityController::setSceneExtent(float extent) {
    sceneExtent = extent;
    std::cout << "Density control scene extent set to: " << extent << std::endl;
    std::cout << "  Split threshold: " << (PERCENT_DENSE * extent) << " world units" << std::endl;
    std::cout << "  Prune threshold (world): " << (0.1f * extent) << " world units" << std::endl;
}

DensityController::DensityController(MTL::Device* device, MTL::Library* library)
    : device(device)
    , maxGaussians(MAX_GAUSSIANS)
{
    gradientAccum = device->newBuffer(maxGaussians * sizeof(float), MTL::ResourceStorageModeShared);
    gradientCount = device->newBuffer(maxGaussians * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    markerBuffer = device->newBuffer(maxGaussians * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    // NEW: Store position gradients for gradient-directed cloning
    positionGradAccum = device->newBuffer(maxGaussians * sizeof(simd_float3), MTL::ResourceStorageModeShared);
    
    resetAccumulator(maxGaussians);
}

DensityController::~DensityController() {
    if (gradientAccum) gradientAccum->release();
    if (gradientCount) gradientCount->release();
    if (markerBuffer) markerBuffer->release();
    if (positionGradAccum) positionGradAccum->release();
}

void DensityController::resetAccumulator(size_t gaussianCount) {
    memset(gradientAccum->contents(), 0, gaussianCount * sizeof(float));
    memset(gradientCount->contents(), 0, gaussianCount * sizeof(uint32_t));
    memset(positionGradAccum->contents(), 0, gaussianCount * sizeof(simd_float3));
}

void DensityController::accumulateGradients(MTL::CommandQueue* queue,
                                            MTL::Buffer* gradients,
                                            size_t gaussianCount) {
    // Accumulate gradient magnitudes for density control decisions
    // This runs on CPU for simplicity - could be GPU compute shader
    
    // Must match gradients.hpp and tiled_shaders.metal
    struct GaussianGradients {
        float position_x, position_y, position_z;
        float opacity;
        float scale_x, scale_y, scale_z;
        float _pad1;
        simd_float4 rotation;
        float sh[12];
        float viewspace_grad_x;  // Screen-space gradient for density control
        float viewspace_grad_y;
        float _pad2, _pad3;
    };
    
    GaussianGradients* grads = (GaussianGradients*)gradients->contents();
    float* accumGrad = (float*)gradientAccum->contents();
    uint32_t* counts = (uint32_t*)gradientCount->contents();
    simd_float3* posGradAccum = (simd_float3*)positionGradAccum->contents();
    
    for (size_t i = 0; i < gaussianCount; i++) {
        // Use VIEWSPACE (screen-space) gradients for density control
        // This matches official 3DGS which uses viewspace_point_tensor.grad[:, :2]
        float gradMag = sqrtf(grads[i].viewspace_grad_x * grads[i].viewspace_grad_x +
                              grads[i].viewspace_grad_y * grads[i].viewspace_grad_y);
        
        if (!std::isnan(gradMag) && !std::isinf(gradMag)) {
            accumGrad[i] += gradMag;
            counts[i]++;
            
            // Accumulate position gradients for gradient-directed cloning
            posGradAccum[i].x += grads[i].position_x;
            posGradAccum[i].y += grads[i].position_y;
            posGradAccum[i].z += grads[i].position_z;
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
                                      float maxScale,
                                      float focalLength,
                                      float imageWidth,
                                      float avgDepth) {

    DensityStats stats = {0, 0, 0};
    
    // Check if we should densify at this iteration
    bool canDensify = (iteration >= DENSIFY_FROM_ITER && iteration < DENSIFY_UNTIL_ITER);
    
    if (!canDensify && iteration >= DENSIFY_UNTIL_ITER) {
        std::cout << "Densification stopped at iteration " << iteration << std::endl;
        resetAccumulator(gaussianCount);
        return stats;
    }
    
    Gaussian* gaussians = (Gaussian*)gaussianBuffer->contents();
    uint32_t* markers = (uint32_t*)markerBuffer->contents();
    float* accumGrad = (float*)gradientAccum->contents();
    uint32_t* counts = (uint32_t*)gradientCount->contents();
    simd_float3* posGradAccum = (simd_float3*)positionGradAccum->contents();
    
    // Viewspace pruning threshold (20% of image width is too big)
    const float maxScreenFraction = 0.2f;
    
    // First pass: decide what to do with each Gaussian
    for (size_t i = 0; i < gaussianCount; i++) {
        Gaussian& g = gaussians[i];
        
        // Compute sigmoid opacity
        float opacity = 1.0f / (1.0f + expf(-g.opacity));
        
        // Get average gradient
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
        
        // Viewspace-based pruning: prune Gaussians with large screen-space footprints
        float screenFraction = computeApproxScreenRadius(g, focalLength, avgDepth, imageWidth);
        if (screenFraction > maxScreenFraction) {
            shouldPrune = true;
        }
        
        if (shouldPrune) {
            markers[i] = 1;  // Prune
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
        
        // Reduce clones first
        for (size_t i = 0; i < gaussianCount && excess > 0; i++) {
            if (markers[i] == 2) {
                markers[i] = 0;
                stats.numCloned--;
                excess--;
            }
        }
        
        // Then reduce splits
        for (size_t i = 0; i < gaussianCount && excess > 0; i++) {
            if (markers[i] == 3) {
                markers[i] = 0;
                stats.numSplit--;
                excess--;
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
            // Clone: Keep original
            newGaussians[writeIdx] = g;
            newPositions[writeIdx] = g.position;
            writeIdx++;
            
            // Create clone moved in gradient direction (official behavior)
            Gaussian cloned = g;
            
            // Get accumulated position gradient
            simd_float3 posGrad = posGradAccum[i];
            float gradNorm = sqrtf(posGrad.x * posGrad.x + posGrad.y * posGrad.y + posGrad.z * posGrad.z);
            
            if (gradNorm > 1e-8f) {
                // Normalize and scale by actual Gaussian size
                float actualScale = expf(std::clamp(g.scale.x, -MAX_SCALE_LOG, MAX_SCALE_LOG));
                float moveDistance = actualScale * 2.0f;  // Move by 2x the Gaussian's scale
                
                cloned.position.x += (posGrad.x / gradNorm) * moveDistance;
                cloned.position.y += (posGrad.y / gradNorm) * moveDistance;
                cloned.position.z += (posGrad.z / gradNorm) * moveDistance;
            } else {
                // Fallback: small random perturbation proportional to scale
                float actualScale = expf(std::clamp(g.scale.x, -MAX_SCALE_LOG, MAX_SCALE_LOG));
                cloned.position.x += actualScale * ((float)rand() / RAND_MAX - 0.5f);
                cloned.position.y += actualScale * ((float)rand() / RAND_MAX - 0.5f);
                cloned.position.z += actualScale * ((float)rand() / RAND_MAX - 0.5f);
            }
            
            newGaussians[writeIdx] = cloned;
            newPositions[writeIdx] = cloned.position;
            writeIdx++;
        } else if (marker == 3) {
            // Split: create two smaller Gaussians
            float scaleFactor = 1.0f / 1.6f;  // Paper uses φ = 1.6
            float logScaleFactor = logf(scaleFactor);  // Add this to log scale
            
            // Get actual scale (apply exp to log scale)
            simd_float3 scale = simd_make_float3(
                expf(std::clamp(g.scale.x, -MAX_SCALE_LOG, MAX_SCALE_LOG)),
                expf(std::clamp(g.scale.y, -MAX_SCALE_LOG, MAX_SCALE_LOG)),
                expf(std::clamp(g.scale.z, -MAX_SCALE_LOG, MAX_SCALE_LOG))
            );
            
            // Sample offset from Gaussian distribution (official uses PDF sampling)
            // Simplified: offset along major axis scaled by actual size
            simd_float3 offset;
            float maxS = fmaxf(fmaxf(scale.x, scale.y), scale.z);
            
            // Random direction scaled by Gaussian
            float rx = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
            float ry = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
            float rz = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
            float rNorm = sqrtf(rx*rx + ry*ry + rz*rz);
            if (rNorm > 0.001f) {
                rx /= rNorm; ry /= rNorm; rz /= rNorm;
            }
            
            offset = simd_make_float3(rx * scale.x, ry * scale.y, rz * scale.z);
            
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

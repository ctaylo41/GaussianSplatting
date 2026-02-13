//
//  density_control.hpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-28.
//

#include "density_control.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include "gradients.hpp"
// Apples GCD for parallel operations
#include <dispatch/dispatch.h>
#include <random>

// Number of threads for parallel operations
static const int NUM_THREADS = 8;

// Deterministic RNG for split offsets
static std::mt19937 splitRng(42);

// Papers recommended threshold

// densify_grad_threshold
static constexpr float GRAD_THRESHOLD = 0.0002f;
// Opacity prune threshold minimum opacity below which to prune
static constexpr float OPACITY_PRUNE_THRESHOLD = 0.01f;
// Brightness prune threshold prune if max(R,G,B) < this
static constexpr float MIN_BRIGHTNESS_THRESHOLD = 0.02f;
static constexpr float SH_C0 = 0.28209479177387814f;
// percent_dense for clone vs split
static constexpr float PERCENT_DENSE = 0.01f; 
// Buffer pre-allocation size no runtime cap on Gaussian count.
// Density control prune vs clone/split naturally regulates count.
static constexpr size_t MAX_GAUSSIANS = 2000000;
// Start densification
static constexpr size_t DENSIFY_FROM_ITER = 500;      
// Stop densification
static constexpr size_t DENSIFY_UNTIL_ITER = 15000;   
// Clamp scale values
static constexpr float MAX_SCALE_LOG = 4.0f;
// Opacity reset interval skip density control around these iterations 3000, 6000, 9000, 12000
static constexpr size_t OPACITY_RESET_INTERVAL = 3000;
// Warmup window after each opacity reset.
// During this window, disable opacity screen-size pruning so coverage can recover
// before aggressive pruning resumes.
static constexpr size_t OPACITY_RESET_WARMUP = 1000;

// Scene extent set during initialization
static float sceneExtent = 1.0f; 

// Helper check if we're within warm-up period after any opacity reset
static bool isInOpacityResetWarmup(size_t iteration) {
    if (OPACITY_RESET_WARMUP == 0) return false;
    if (iteration < OPACITY_RESET_INTERVAL) return false;
    // Check position within current opacity reset interval
    size_t itersSinceReset = iteration % OPACITY_RESET_INTERVAL;
    return (itersSinceReset > 0 && itersSinceReset < OPACITY_RESET_WARMUP);
} 

// Set scene extent for relative thresholds
void DensityController::setSceneExtent(float extent) {
    sceneExtent = extent;
    std::cout << "Density control scene extent set to: " << extent << std::endl;
    std::cout << "  Split threshold: " << (PERCENT_DENSE * extent) << " world units" << std::endl;
    std::cout << "  Prune threshold (world): " << (0.1f * extent) << " world units" << std::endl;
}

// Constructor
DensityController::DensityController(MTL::Device* device, MTL::Library* library)
    : device(device)
    , maxGaussians(MAX_GAUSSIANS)
{
    // Allocate buffers
    gradientAccum = device->newBuffer(maxGaussians * sizeof(float), MTL::ResourceStorageModeShared);
    gradientCount = device->newBuffer(maxGaussians * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    markerBuffer = device->newBuffer(maxGaussians * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    // Store position gradients for gradient-directed cloning
    positionGradAccum = device->newBuffer(maxGaussians * sizeof(simd_float3), MTL::ResourceStorageModeShared);
    
    // Track maximum screen-space radius across all views official max_radii2D
    maxRadii2D = device->newBuffer(maxGaussians * sizeof(float), MTL::ResourceStorageModeShared);
    
    // Initialize accumulators
    resetAccumulator(maxGaussians);
}

// Destructor
DensityController::~DensityController() {
    // Release buffers
    if (gradientAccum) gradientAccum->release();
    if (gradientCount) gradientCount->release();
    if (markerBuffer) markerBuffer->release();
    if (positionGradAccum) positionGradAccum->release();
    if (maxRadii2D) maxRadii2D->release();
}

// Ensure internal buffers can accommodate required Gaussian count
void DensityController::ensureCapacity(size_t requiredCount) {
    // No need to grow if current capacity suffices
    if (requiredCount <= maxGaussians) {
        return;
    }


    // Compute new capacity 1.5x growth to balance reallocations vs memory usage
    size_t oldCapacity = maxGaussians;
    size_t newCapacity = std::max(requiredCount, maxGaussians + maxGaussians / 2);

    // Allocate new buffers
    MTL::Buffer* newGradientAccum = device->newBuffer(newCapacity * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* newGradientCount = device->newBuffer(newCapacity * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* newMarkerBuffer = device->newBuffer(newCapacity * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* newPositionGradAccum = device->newBuffer(newCapacity * sizeof(simd_float3), MTL::ResourceStorageModeShared);
    MTL::Buffer* newMaxRadii2D = device->newBuffer(newCapacity * sizeof(float), MTL::ResourceStorageModeShared);

    // Check allocations
    if (!newGradientAccum || !newGradientCount || !newMarkerBuffer || !newPositionGradAccum || !newMaxRadii2D) {
        std::cerr << "DensityController: failed to grow internal buffers to " << newCapacity << std::endl;
        std::abort();
    }

    // Initialize new buffers to zero
    memset(newGradientAccum->contents(), 0, newCapacity * sizeof(float));
    memset(newGradientCount->contents(), 0, newCapacity * sizeof(uint32_t));
    memset(newMarkerBuffer->contents(), 0, newCapacity * sizeof(uint32_t));
    memset(newPositionGradAccum->contents(), 0, newCapacity * sizeof(simd_float3));
    memset(newMaxRadii2D->contents(), 0, newCapacity * sizeof(float));

    // Copy old data to new buffers
    memcpy(newGradientAccum->contents(), gradientAccum->contents(), oldCapacity * sizeof(float));
    memcpy(newGradientCount->contents(), gradientCount->contents(), oldCapacity * sizeof(uint32_t));
    memcpy(newMarkerBuffer->contents(), markerBuffer->contents(), oldCapacity * sizeof(uint32_t));
    memcpy(newPositionGradAccum->contents(), positionGradAccum->contents(), oldCapacity * sizeof(simd_float3));
    memcpy(newMaxRadii2D->contents(), maxRadii2D->contents(), oldCapacity * sizeof(float));

    // Release old buffers
    gradientAccum->release();
    gradientCount->release();
    markerBuffer->release();
    positionGradAccum->release();
    maxRadii2D->release();

    // Update pointers and capacity
    gradientAccum = newGradientAccum;
    gradientCount = newGradientCount;
    markerBuffer = newMarkerBuffer;
    positionGradAccum = newPositionGradAccum;
    maxRadii2D = newMaxRadii2D;
    maxGaussians = newCapacity;

    std::cout << "Density control buffers grown: " << oldCapacity << " -> " << newCapacity << std::endl;
}

// Reset accumulators
void DensityController::resetAccumulator(size_t gaussianCount) {
    ensureCapacity(gaussianCount);

    // Reset accumulators to zero using memset
    memset(gradientAccum->contents(), 0, gaussianCount * sizeof(float));
    memset(gradientCount->contents(), 0, gaussianCount * sizeof(uint32_t));
    memset(positionGradAccum->contents(), 0, gaussianCount * sizeof(simd_float3));
    memset(maxRadii2D->contents(), 0, gaussianCount * sizeof(float));
}

// Accumulate gradients into internal buffers
void DensityController::accumulateGradients(MTL::CommandQueue* queue,
                                            MTL::Buffer* gradients,
                                            size_t gaussianCount) {
    ensureCapacity(gaussianCount);
    
    // Must match gradients.hpp and tiled_shaders.metal
    struct GaussianGradients {
        float position_x, position_y, position_z;
        float opacity;
        float scale_x, scale_y, scale_z;
        float _pad1;
        simd_float4 rotation;
        float sh[12];
        float viewspace_grad_x;
        float viewspace_grad_y;
        float _pad2, _pad3;
    };
    
    // Access buffer contents
    GaussianGradients* grads = (GaussianGradients*)gradients->contents();
    float* accumGrad = (float*)gradientAccum->contents();
    uint32_t* counts = (uint32_t*)gradientCount->contents();
    simd_float3* posGradAccum = (simd_float3*)positionGradAccum->contents();
    
    // Parallel accumulation using GCD
    dispatch_queue_t dispatchQueue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
    size_t chunkSize = (gaussianCount + NUM_THREADS - 1) / NUM_THREADS;
    
    // Parallel loop
    dispatch_apply((size_t)NUM_THREADS, dispatchQueue, ^(size_t t) {
        // Compute chunk range
        size_t start = t * chunkSize;
        size_t end = std::min(start + chunkSize, gaussianCount);
        
        // Accumulate gradients for this chunk
        for (size_t i = start; i < end; i++) {
            // Use viewspace gradients for density control
            float gradMag = sqrtf(grads[i].viewspace_grad_x * grads[i].viewspace_grad_x +
                                  grads[i].viewspace_grad_y * grads[i].viewspace_grad_y);
            
            // No gradient magnitude clamping 
            
            // Only accumulate valid gradients
            if (!std::isnan(gradMag) && !std::isinf(gradMag) && gradMag > 0.0f) {
                accumGrad[i] += gradMag;
                counts[i]++;
                
                // Accumulate position gradients for gradient-directed cloning
                posGradAccum[i].x += grads[i].position_x;
                posGradAccum[i].y += grads[i].position_y;
                posGradAccum[i].z += grads[i].position_z;
            }
        }
    });
    
}

// Accumulate screen-space radii from rasterizer for accurate size-based pruning
void DensityController::accumulateRadii(MTL::Buffer* projectedGaussians,
                                        size_t gaussianCount) {
    ensureCapacity(gaussianCount);
    if (!projectedGaussians) {
        return;
    }

    // ProjectedGaussian structure must match tiled_rasterizer.hpp
    struct ProjectedGaussian {
        simd_float2 screenPos;
        float conic[3];
        float depth;
        float opacity;
        float color[3];
        float radius;
        uint32_t tileMinX;
        uint32_t tileMinY;
        uint32_t tileMaxX;
        uint32_t tileMaxY;
        uint8_t colorClamped[3]; 
        uint8_t _pad1;
        simd_float2 viewPos_xy;
        float cov2D[3];
        float viewDir[3];
    };

    // Clamp processing to buffer capacity to avoid out-of-bounds access
    size_t projectedCapacity = projectedGaussians->length() / sizeof(ProjectedGaussian);
    size_t countToProcess = std::min(gaussianCount, projectedCapacity);
    if (countToProcess == 0) {
        return;
    }
    
    // Access buffer contents
    ProjectedGaussian* projected = (ProjectedGaussian*)projectedGaussians->contents();
    float* maxRadii = (float*)maxRadii2D->contents();

    // If projected buffer is smaller than Gaussian count, log a warning and only process what fits
    if (countToProcess < gaussianCount) {
        memset(maxRadii + countToProcess, 0, (gaussianCount - countToProcess) * sizeof(float));
        std::cerr << "WARNING: accumulateRadii clamped to projected buffer capacity "
                  << projectedCapacity << " (requested " << gaussianCount << ")" << std::endl;
    }

    // Parallel update of max radii using GCD
    dispatch_queue_t dispatchQueue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
    size_t chunkSize = (countToProcess + NUM_THREADS - 1) / NUM_THREADS;
    dispatch_apply((size_t)NUM_THREADS, dispatchQueue, ^(size_t t) {
        size_t start = t * chunkSize;
        size_t end = std::min(start + chunkSize, countToProcess);
        for (size_t i = start; i < end; i++) {
            float currentRadius = projected[i].radius;
            if (currentRadius > maxRadii[i]) {
                maxRadii[i] = currentRadius;
            }
        }
    });

}

// Apply density control prune, clone, split Gaussians
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
                                      float avgDepth,
                                      bool pruneOnly) {

    ensureCapacity(gaussianCount);

    DensityStats stats = {0, 0, 0};
    
    const bool inOpacityResetWarmup = isInOpacityResetWarmup(iteration);
    
    // Check if we should densify at this iteration
    // prune only mode is used to recover opacity
    bool canDensify = !pruneOnly && (iteration >= DENSIFY_FROM_ITER && iteration < DENSIFY_UNTIL_ITER);
    
    // If past densify_until_iter, return early
    if (iteration >= DENSIFY_UNTIL_ITER) {
        std::cout << "Densification stopped at iteration " << iteration << std::endl;
        return stats;
    }
    
    // Access buffer contents
    Gaussian* gaussians = (Gaussian*)gaussianBuffer->contents();
    uint32_t* markers = (uint32_t*)markerBuffer->contents();
    float* accumGrad = (float*)gradientAccum->contents();
    uint32_t* counts = (uint32_t*)gradientCount->contents();
    simd_float3* posGradAccum = (simd_float3*)positionGradAccum->contents();
    
    // Access max_radii2D buffer actual screen radii from rasterizer
    float* maxRadii = (float*)maxRadii2D->contents();
    
    // Determine pruning thresholds and strategies based on iteration and warmup status
    const bool enableSizePruning = (iteration >= DENSIFY_FROM_ITER && iteration < DENSIFY_UNTIL_ITER);
    const bool enableScreenPruning = (iteration > OPACITY_RESET_INTERVAL) && !inOpacityResetWarmup;
    static constexpr float screenPruneThreshold = 40.0f;
    static constexpr float screenPruneOpacityGate = 0.35f;
    
    // Per thread counters for parallel first pass
    static uint32_t threadPruned[NUM_THREADS];
    static uint32_t threadCloned[NUM_THREADS];
    static uint32_t threadSplit[NUM_THREADS];
    static uint32_t threadCorrupted[NUM_THREADS];

    // Reset per thread counters
    memset(threadPruned, 0, sizeof(threadPruned));
    memset(threadCloned, 0, sizeof(threadCloned));
    memset(threadSplit, 0, sizeof(threadSplit));
    memset(threadCorrupted, 0, sizeof(threadCorrupted));

    // Capture locals for block
    const float splitThreshold = PERCENT_DENSE * sceneExtent;
    const float pruneThreshold = 0.1f * sceneExtent;

    // Parallel first pass to decide prune clone split
    dispatch_queue_t dispatchQueue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
    size_t chunkSize = (gaussianCount + NUM_THREADS - 1) / NUM_THREADS;

    // First pass decide what to do with each Gaussian
    dispatch_apply((size_t)NUM_THREADS, dispatchQueue, ^(size_t t) {
        // Compute chunk range
        size_t start = t * chunkSize;
        size_t end = std::min(start + chunkSize, gaussianCount);

        uint32_t localPruned = 0, localCloned = 0, localSplit = 0;

        // Process Gaussians in this chunk
        for (size_t i = start; i < end; i++) {
            Gaussian& g = gaussians[i];

            // Check for corrupted Gaussians NaN/inf always prune these
            bool isCorrupted = std::isnan(g.position.x) || std::isnan(g.position.y) || std::isnan(g.position.z) ||
                               std::isinf(g.position.x) || std::isinf(g.position.y) || std::isinf(g.position.z) ||
                               std::isnan(g.opacity) || std::isinf(g.opacity) ||
                               std::isnan(g.scale.x) || std::isnan(g.scale.y) || std::isnan(g.scale.z) ||
                               std::isinf(g.scale.x) || std::isinf(g.scale.y) || std::isinf(g.scale.z) ||
                               std::isnan(g.rotation.x) || std::isnan(g.rotation.y) ||
                               std::isnan(g.rotation.z) || std::isnan(g.rotation.w) ||
                               std::isinf(g.rotation.x) || std::isinf(g.rotation.y) ||
                               std::isinf(g.rotation.z) || std::isinf(g.rotation.w);

            // Check SH coefficients for NaN/inf
            if (!isCorrupted) {
                for (int j = 0; j < 12; j++) {
                    if (std::isnan(g.sh[j]) || std::isinf(g.sh[j])) {
                        isCorrupted = true;
                        break;
                    }
                }
            }

            if (isCorrupted) {
                markers[i] = 1;  // Prune
                localPruned++;
                threadCorrupted[t]++;
                continue;
            }

            // Compute sigmoid opacity
            float opacity = 1.0f / (1.0f + expf(-g.opacity));
            
            // Get average gradient
            float avgGrad = (counts[i] > 0) ? (accumGrad[i] / counts[i]) : 0.0f;
            
            // Compute scale values in world units applying exp to log scale
            float sx = expf(std::clamp(g.scale.x, -MAX_SCALE_LOG, MAX_SCALE_LOG));
            float sy = expf(std::clamp(g.scale.y, -MAX_SCALE_LOG, MAX_SCALE_LOG));
            float sz = expf(std::clamp(g.scale.z, -MAX_SCALE_LOG, MAX_SCALE_LOG));
            float maxScaleVal = fmaxf(fmaxf(sx, sy), sz);
            
            // prune_mask 
            bool shouldPrune = (!inOpacityResetWarmup && opacity < minOpacity);
            
            // Size-based pruning official 3DGS only enables after first opacity_reset_interval
            if (enableSizePruning) {
                if (maxScaleVal > pruneThreshold) {
                    shouldPrune = true;
                }

                // Screen-space pruning big_points_vs
                if (enableScreenPruning && maxRadii[i] > screenPruneThreshold && opacity < screenPruneOpacityGate) {
                    shouldPrune = true;
                }
            }
            
            // Mark accordingly and prune
            if (shouldPrune) {
                markers[i] = 1;
                localPruned++;
            } else if (canDensify && avgGrad > gradThreshold) {
                // Clone if small Split if large
                if (maxScaleVal > splitThreshold) {
                    // Split large Gaussians
                    markers[i] = 3;  
                    localSplit++;
                    

                } else {
                    // Clone small Gaussians
                    markers[i] = 2;  
                    localCloned++;
                }
            } else {
                // Keep
                markers[i] = 0;
            }
        }
        
        // Store local counts
        threadPruned[t] = localPruned;
        threadCloned[t] = localCloned;
        threadSplit[t] = localSplit;
    });

    // Sum up thread-local counters
    for (int t = 0; t < NUM_THREADS; t++) {
        stats.numPruned += threadPruned[t];
        stats.numCloned += threadCloned[t];
        stats.numSplit += threadSplit[t];
    }

    // Compute new count no artificial caps, density control self-regulates
    size_t newCount = gaussianCount - stats.numPruned + stats.numCloned + stats.numSplit;
    
    // Allocate new buffers
    MTL::Buffer* newGaussianBuffer = device->newBuffer(newCount * sizeof(Gaussian), MTL::ResourceStorageModeShared);
    MTL::Buffer* newPositionsBuffer = device->newBuffer(newCount * sizeof(simd_float3), MTL::ResourceStorageModeShared);
    
    // Access new buffer contents
    Gaussian* newGaussians = (Gaussian*)newGaussianBuffer->contents();
    simd_float3* newPositions = (simd_float3*)newPositionsBuffer->contents();
    
    // Second pass to build new arrays and populate index mapping for momentum remapping
    stats.indexMapping.clear();
    stats.indexMapping.reserve(gaussianCount - stats.numPruned);

    size_t writeIdx = 0;
    for (size_t i = 0; i < gaussianCount; i++) {
        Gaussian& g = gaussians[i];
        uint32_t marker = markers[i];

        if (marker == 1) {
            // Pruned no mapping momentum discarded
            continue;
        }

        if (marker == 0) {
            // Keep as is record old->new mapping for momentum preservation
            stats.indexMapping.push_back({i, writeIdx});
            newGaussians[writeIdx] = g;
            newPositions[writeIdx] = g.position;
            writeIdx++;
        } else if (marker == 2) {
            // Clone ecord mapping for original only 
            stats.indexMapping.push_back({i, writeIdx});
            newGaussians[writeIdx] = g;
            newPositions[writeIdx] = g.position;
            writeIdx++;

            Gaussian cloned = g;

            // Offset clone along position gradient direction to break symmetry and prevent self-reinforcing cloning
            simd_float3 posGrad = posGradAccum[i];
            float gradNorm = sqrtf(posGrad.x*posGrad.x + posGrad.y*posGrad.y + posGrad.z*posGrad.z);
            if (gradNorm > 1e-7f) {
                // Gaussian's world-space extent max axis
                float maxScaleVal = fmaxf(fmaxf(
                    expf(std::clamp(g.scale.x, -MAX_SCALE_LOG, MAX_SCALE_LOG)),
                    expf(std::clamp(g.scale.y, -MAX_SCALE_LOG, MAX_SCALE_LOG))),
                    expf(std::clamp(g.scale.z, -MAX_SCALE_LOG, MAX_SCALE_LOG)));
                // Offset magnitude proportional to Gaussian size to ensure clones are sufficiently separated, even for small Gaussians with high gradients
                float offsetMag = maxScaleVal * 2.0f;
                cloned.position.x += (posGrad.x / gradNorm) * offsetMag;
                cloned.position.y += (posGrad.y / gradNorm) * offsetMag;
                cloned.position.z += (posGrad.z / gradNorm) * offsetMag;
            }

            // Write clone at offset position (no mapping - fresh momentum)
            newGaussians[writeIdx] = cloned;
            newPositions[writeIdx] = cloned.position;
            writeIdx++;
        } else if (marker == 3) {
            // Split and create two smaller Gaussians
            float scaleFactor = 1.0f / 1.6f; 
            float logScaleFactor = logf(scaleFactor); 

            // Get actual scale by applying exp to log scale
            simd_float3 scale = simd_make_float3(
                expf(std::clamp(g.scale.x, -MAX_SCALE_LOG, MAX_SCALE_LOG)),
                expf(std::clamp(g.scale.y, -MAX_SCALE_LOG, MAX_SCALE_LOG)),
                expf(std::clamp(g.scale.z, -MAX_SCALE_LOG, MAX_SCALE_LOG))
            );
            
            // Generate random offset direction
            simd_float3 offset;
            float maxS = fmaxf(fmaxf(scale.x, scale.y), scale.z);
            
            // Random direction scaled by Gaussian
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            float rx = dist(splitRng);
            float ry = dist(splitRng);
            float rz = dist(splitRng);
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
            
            // Child 1 positive offset smaller scale
            Gaussian child1 = g;
            child1.position = g.position + offset;
            // Scale is in log space so add log(scaleFactor)
            child1.scale = simd_make_float3(
                g.scale.x + logScaleFactor,
                g.scale.y + logScaleFactor,
                g.scale.z + logScaleFactor
            );
            // Write child 1
            newGaussians[writeIdx] = child1;
            newPositions[writeIdx] = child1.position;
            writeIdx++;
            
            // Child 2 negative offset same smaller scale
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
    
    // Reset accumulators for next iteration
    resetAccumulator(gaussianCount);
    
    std::cout << "Density control: pruned=" << stats.numPruned
              << " cloned=" << stats.numCloned
              << " split=" << stats.numSplit
              << " total=" << gaussianCount;
    if (inOpacityResetWarmup) {
        std::cout << " [WARMUP: prune relaxed]";
    }
    if (pruneOnly) {
        std::cout << " [PRUNE-ONLY: opacity recovery]";
    }
    if (enableScreenPruning && screenPruneThreshold > 0) {
        std::cout << " screenPrune=" << screenPruneThreshold << "px"
                  << " (opacity<" << screenPruneOpacityGate << ")";
    }
    std::cout << std::endl;
    
    return stats;
}

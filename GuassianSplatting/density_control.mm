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

// Number of threads for parallel operations
static const int NUM_THREADS = 8;

// Papers recommended threshold

// densify_grad_threshold
static constexpr float GRAD_THRESHOLD = 0.0002f;
// Opacity prune threshold minimum opacity below which to prune
static constexpr float OPACITY_PRUNE_THRESHOLD = 0.005f;
// Brightness prune threshold prune if max(R,G,B) < this
static constexpr float MIN_BRIGHTNESS_THRESHOLD = 0.02f;
static constexpr float SH_C0 = 0.28209479177387814f;
// percent_dense for clone vs split
static constexpr float PERCENT_DENSE = 0.01f; 
static constexpr size_t MAX_GAUSSIANS = 1000000;
// Start densification
static constexpr size_t DENSIFY_FROM_ITER = 500;      
// Stop densification
static constexpr size_t DENSIFY_UNTIL_ITER = 15000;   
// Clamp scale values
static constexpr float MAX_SCALE_LOG = 4.0f;
// Opacity reset interval skip density control around these iterations 3000, 6000, 9000, 12000
static constexpr size_t OPACITY_RESET_INTERVAL = 3000;
// Warm-up iterations after opacity reset before resuming density control
static constexpr size_t OPACITY_RESET_WARMUP = 500;          

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

// Reset accumulators
void DensityController::resetAccumulator(size_t gaussianCount) {
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
            
            // Clamp gradient magnitude to prevent explosive accumulation
            // This is critical around opacity reset when gradients can spike
            const float MAX_GRAD_MAG = 1.0f;
            gradMag = std::min(gradMag, MAX_GRAD_MAG);
            
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
        float _pad1;
        simd_float2 viewPos_xy;
        float cov2D[3];
        float _pad2;
    };
    
    ProjectedGaussian* projected = (ProjectedGaussian*)projectedGaussians->contents();
    float* maxRadii = (float*)maxRadii2D->contents();
    

    
    // Update max radius for each Gaussian
    for (size_t i = 0; i < gaussianCount; i++) {
        float currentRadius = projected[i].radius;
        // Track maximum radius seen across all views
        if (currentRadius > maxRadii[i]) {
            maxRadii[i] = currentRadius;
        }
    }
    

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
                                      float avgDepth) {

    DensityStats stats = {0, 0, 0};
    
    // Skip density control during opacity reset warmup period
    // This prevents bad decisions with post-reset low opacities
    if (isInOpacityResetWarmup(iteration)) {
        std::cout << "Skipping density control - opacity reset warmup (iter " << iteration << ")" << std::endl;
        resetAccumulator(gaussianCount);
        return stats;
    }
    
    // Check if we should densify at this iteration
    bool canDensify = (iteration > DENSIFY_FROM_ITER && iteration < DENSIFY_UNTIL_ITER);
    
    // If past densify_until_iter just return
    if (iteration >= DENSIFY_UNTIL_ITER) {
        std::cout << "Densification stopped at iteration " << iteration << std::endl;
        resetAccumulator(gaussianCount);
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
    
    // Fixed percentile-based screen pruning threshold
    // Prevents death spiral from recalculating percentile every iteration
    static float maxScreenPixels = -1.0f;
    static bool thresholdComputed = false;
    
    // Dont enable screen pruning until well after opacity reset and warmup
    // This prevents mass-pruning of large Gaussians before they've adapted
    const bool enableScreenPruning = (iteration > OPACITY_RESET_INTERVAL + OPACITY_RESET_WARMUP + 200);
    
    // Compute threshold once at first densification after opacity reset
    if (enableScreenPruning && !thresholdComputed && gaussianCount > 100) {
        // Copy radii to vector for percentile calculation
        std::vector<float> radiiCopy(maxRadii, maxRadii + gaussianCount);
        
        // Find 95th percentile using nth_element
        size_t percentileIdx = size_t(gaussianCount * 0.95f);
        std::nth_element(radiiCopy.begin(), radiiCopy.begin() + percentileIdx, radiiCopy.end());
        maxScreenPixels = radiiCopy[percentileIdx];
        
        // Dont prune too aggressively or too loosely
        maxScreenPixels = std::clamp(maxScreenPixels, 50.0f, 300.0f);
        thresholdComputed = true;
        
        std::cout << "*** FIXED screen pruning threshold computed: " << maxScreenPixels << " pixels (will not change) ***" << std::endl;
    }
    
    // Per thread counters for parallel first pass
    static uint32_t threadPruned[NUM_THREADS];
    static uint32_t threadCloned[NUM_THREADS];
    static uint32_t threadSplit[NUM_THREADS];
    static uint32_t threadPrunedBlack[NUM_THREADS];

    // Reset per thread counters
    memset(threadPruned, 0, sizeof(threadPruned));
    memset(threadCloned, 0, sizeof(threadCloned));
    memset(threadSplit, 0, sizeof(threadSplit));
    memset(threadPrunedBlack, 0, sizeof(threadPrunedBlack));
    
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
        
        uint32_t localPruned = 0, localCloned = 0, localSplit = 0, localPrunedBlack = 0;
        
        // Process Gaussians in this chunk
        for (size_t i = start; i < end; i++) {
            Gaussian& g = gaussians[i];
            
            // Compute sigmoid opacity
            float opacity = 1.0f / (1.0f + expf(-g.opacity));
            
            // Get average gradient
            float avgGrad = (counts[i] > 0) ? (accumGrad[i] / counts[i]) : 0.0f;
            
            // Compute max scale in world units applying exp to log scale
            float maxScaleVal = fmaxf(fmaxf(
                expf(std::clamp(g.scale.x, -MAX_SCALE_LOG, MAX_SCALE_LOG)),
                expf(std::clamp(g.scale.y, -MAX_SCALE_LOG, MAX_SCALE_LOG))),
                expf(std::clamp(g.scale.z, -MAX_SCALE_LOG, MAX_SCALE_LOG)));
            
            // prune_mask
            bool shouldPrune = (opacity < OPACITY_PRUNE_THRESHOLD);

            // Additional prune if very very dark
            if (canDensify && iteration > OPACITY_RESET_INTERVAL && maxScaleVal > pruneThreshold) {
                shouldPrune = true;
            }
            
            // Optionally prune based on screen-space radius using adaptive percentile-based
            if (enableScreenPruning && maxRadii[i] > maxScreenPixels) {
                shouldPrune = true;
            }
            
            // Mark accordingly and prune
            if (shouldPrune) {
                markers[i] = 1;
                localPruned++;
            } else if (canDensify && avgGrad > GRAD_THRESHOLD) {
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
        threadPrunedBlack[t] = localPrunedBlack;
    });

    // Sum up thread-local counters
    for (int t = 0; t < NUM_THREADS; t++) {
        stats.numPruned += threadPruned[t];
        stats.numCloned += threadCloned[t];
        stats.numSplit += threadSplit[t];
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
        // Recompute new count
        newCount = gaussianCount - stats.numPruned + stats.numCloned + stats.numSplit;
    }
    
    // Allocate new buffers
    MTL::Buffer* newGaussianBuffer = device->newBuffer(newCount * sizeof(Gaussian), MTL::ResourceStorageModeShared);
    MTL::Buffer* newPositionsBuffer = device->newBuffer(newCount * sizeof(simd_float3), MTL::ResourceStorageModeShared);
    
    // Access new buffer contents
    Gaussian* newGaussians = (Gaussian*)newGaussianBuffer->contents();
    simd_float3* newPositions = (simd_float3*)newPositionsBuffer->contents();
    
    // Second pass to build new arrays
    size_t writeIdx = 0;
    for (size_t i = 0; i < gaussianCount; i++) {
        Gaussian& g = gaussians[i];
        uint32_t marker = markers[i];
        
        if (marker == 1) {
            // Pruned
            continue;  
        }
        
        if (marker == 0) {
            // Keep as is
            newGaussians[writeIdx] = g;
            newPositions[writeIdx] = g.position;
            writeIdx++;
        } else if (marker == 2) {
            // Clone and Keep original
            newGaussians[writeIdx] = g;
            newPositions[writeIdx] = g.position;
            writeIdx++;
            
            Gaussian cloned = g;
            
            // Write clone at same position
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
              << " total=" << gaussianCount << std::endl;
    
    return stats;
}

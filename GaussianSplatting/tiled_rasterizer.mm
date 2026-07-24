//
//  tile_rasterizer.hpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-28.
//

#include "tiled_rasterizer.hpp"
#include "ply_loader.hpp" 
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstddef>
#include <vector>
#include <dispatch/dispatch.h> 
#include <chrono>
#include <atomic>

// Number of threads for parallel operations
static const int NUM_THREADS = 8;

// Enable GPU sorting set to true to use GPU radix sort
static constexpr bool USE_GPU_SORT = true;

// Fast parallel radix sort using GCD
// Optimized with parallel histogram and parallel scatter
static void parallelRadixSort(std::vector<std::pair<uint64_t, uint32_t>>& pairs) {
    const size_t n = pairs.size();
    if (n < 1000) {
        std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        return;
    }
    
    // Temporary storage
    static std::vector<std::pair<uint64_t, uint32_t>> temp;
    if (temp.size() < n) temp.resize(n);
    
    // Per-thread histograms
    static uint32_t threadHist[NUM_THREADS][256];
    static uint32_t globalPrefix[256];
    static uint32_t threadOffset[NUM_THREADS][256];
    
    // Dispatch queue for parallel tasks
    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
    
    // Source and destination pointers
    auto* src = &pairs;
    auto* dst = &temp;
    
    // 8 passes of 8-bit radix sort
    for (int pass = 0; pass < 8; pass++) {
        int shift = pass * 8;
        size_t chunkSize = (n + NUM_THREADS - 1) / NUM_THREADS;
        
        // Parallel histogram each thread computes its own histogram
        dispatch_apply(NUM_THREADS, queue, ^(size_t t) {
            memset(threadHist[t], 0, 256 * sizeof(uint32_t));
            size_t start = t * chunkSize;
            size_t end = std::min(start + chunkSize, n);
            // Build histogram
            for (size_t i = start; i < end; i++) {
                uint8_t digit = ((*src)[i].first >> shift) & 0xFF;
                threadHist[t][digit]++;
            }
        });
        
        // Compute global prefix sums and per-thread offsets
        uint32_t sum = 0;
        for (int d = 0; d < 256; d++) {
            globalPrefix[d] = sum;
            // Compute per-thread offsets
            for (int t = 0; t < NUM_THREADS; t++) {
                threadOffset[t][d] = sum;
                sum += threadHist[t][d];
            }
        }
        
        // Parallel scatter each thread writes its chunk
        dispatch_apply(NUM_THREADS, queue, ^(size_t t) {
            uint32_t localOffset[256];
            memcpy(localOffset, threadOffset[t], 256 * sizeof(uint32_t));
            // Process chunk
            size_t start = t * chunkSize;
            size_t end = std::min(start + chunkSize, n);
            // Scatter elements
            for (size_t i = start; i < end; i++) {
                uint8_t digit = ((*src)[i].first >> shift) & 0xFF;
                (*dst)[localOffset[digit]++] = (*src)[i];
            }
        });
        
        // Swap source and destination
        std::swap(src, dst);
    }
    
    // Ensure result is in pairs
    if (src != &pairs) {
        pairs = *src;
    }
}

// Constructor
TiledRasterizer::TiledRasterizer(MTL::Device* device, MTL::Library* library, uint32_t maxGaussians)
    : device(device)
    , library(library)
    , maxGaussians(maxGaussians)
    , maxTiles(0)
    , maxPairs(0)
    , currentWidth(0)
    , currentHeight(0)
    , numTilesX(0)
    , numTilesY(0)
    , activeSortedKeys(nullptr)
    , activeSortedValues(nullptr)
{
    createPipelines(library);
    
    // Static assert struct sizes
    static_assert(sizeof(ProjectedGaussian) == 96, "ProjectedGaussian must be 96 bytes!");
    static_assert(sizeof(TiledUniforms) == 240, "TiledUniforms must be 240 bytes!");
    
    // Allocate projection buffer
    projectedGaussians = device->newBuffer(maxGaussians * sizeof(ProjectedGaussian),
                                           MTL::ResourceStorageModeShared);

    // Allocate intermediate render gradient buffer Stage 1 -> Stage 2
    renderGradientBuffer = device->newBuffer(maxGaussians * sizeof(RenderGradients),
                                              MTL::ResourceStorageModeShared);
    
    // Uniform buffer
    uniformBuffer = device->newBuffer(sizeof(TiledUniforms), MTL::ResourceStorageModeShared);
    totalPairsBuffer = device->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    // Start with reasonable default will grow if needed
    maxPairs = maxGaussians * AVG_TILES_PER_GAUSSIAN;
    gaussianKeys = device->newBuffer(maxPairs * sizeof(uint64_t), MTL::ResourceStorageModeShared);
    gaussianValues = device->newBuffer(maxPairs * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    // Atomic counter for GPU pair generation
    pairCounterBuffer = device->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    // Initialize GPU radix sort for fast GPU-based 64-bit sorting
    gpuRadixSort = new GPURadixSort64(device, library, maxPairs);
    
    // Other buffers allocated on demand
    tileRanges = nullptr;
    perPixelLastIdx = nullptr;
}

// Destructor
TiledRasterizer::~TiledRasterizer() {
    if (projectedGaussians) projectedGaussians->release();
    if (gaussianKeys) gaussianKeys->release();
    if (gaussianValues) gaussianValues->release();
    if (tileRanges) tileRanges->release();
    if (totalPairsBuffer) totalPairsBuffer->release();
    if (perPixelLastIdx) perPixelLastIdx->release();
    if (uniformBuffer) uniformBuffer->release();
    if (pairCounterBuffer) pairCounterBuffer->release();
    if (ssimCoeffKBuffer) ssimCoeffKBuffer->release();
    if (ssimCoeffLBuffer) ssimCoeffLBuffer->release();
    if (ssimCoeffMBuffer) ssimCoeffMBuffer->release();
    if (pixelGradientBuffer) pixelGradientBuffer->release();
    if (renderGradientBuffer) renderGradientBuffer->release();

    if (gpuRadixSort) delete gpuRadixSort;

    if (projectGaussiansPSO) projectGaussiansPSO->release();
    if (tiledForwardPSO) tiledForwardPSO->release();
    if (tiledBackwardPSO) tiledBackwardPSO->release();
    if (buildTileRangesPSO) buildTileRangesPSO->release();
    if (generatePairsPSO) generatePairsPSO->release();
    if (computeSSIMGradCoeffsPSO) computeSSIMGradCoeffsPSO->release();
    if (computePixelGradientPSO) computePixelGradientPSO->release();
    if (preprocessBackwardPSO) preprocessBackwardPSO->release();
}

// Create compute pipelines
void TiledRasterizer::createPipelines(MTL::Library* library) {
    NS::Error* error = nullptr;
    
    // Helper to create pipeline
    auto makePipeline = [&](const char* name) -> MTL::ComputePipelineState* {
        // Find function
        MTL::Function* func = library->newFunction(NS::String::string(name, NS::ASCIIStringEncoding));
        if (!func) {
            std::cerr << "Failed to find function: " << name << std::endl;
            return nullptr;
        }
        // Create pipeline
        MTL::ComputePipelineState* pso = device->newComputePipelineState(func, &error);
        func->release();
        if (!pso) {
            std::cerr << "Failed to create pipeline " << name << ": "
                      << error->localizedDescription()->utf8String() << std::endl;
        }
        return pso;
    };
    
    // Create pipelines
    projectGaussiansPSO = makePipeline("projectGaussians");
    tiledForwardPSO = makePipeline("tiledForward");
    tiledBackwardPSO = makePipeline("tiledBackward");
    buildTileRangesPSO = makePipeline("buildTileRanges");
    generatePairsPSO = makePipeline("generateTilePairs");
    computeSSIMGradCoeffsPSO = makePipeline("computeSSIMGradCoeffs");
    computePixelGradientPSO = makePipeline("computePixelGradient");
    preprocessBackwardPSO = makePipeline("preprocessBackward");
}

// Commit a command buffer, wait for it, and report any execution error.
//
// Metal does not throw on GPU faults — it marks the command buffer Error and returns
// from waitUntilCompleted normally. Every buffer here is ResourceStorageModeShared, so
// on abort the CPU reads whatever the GPU managed to write before dying: a partially
// filled gradient buffer, or stale contents from the previous iteration. Feeding that
// to Adam corrupts the model a little on every abort, which compounds into a rising
// loss long after the aborts themselves start. Hence: detect, label, and propagate.
bool TiledRasterizer::commitAndCheck(MTL::CommandBuffer* cmdBuffer, const char* label) {
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();

    if (cmdBuffer->status() == MTL::CommandBufferStatusError) {
        abortCount++;
        passFailed = true;

        NS::Error* err = cmdBuffer->error();
        std::cerr << "\n[GPU ABORT] pass=" << label
                  << " abort #" << abortCount;
        if (err) {
            std::cerr << " code=" << err->code();
            if (err->localizedDescription()) {
                std::cerr << " desc=" << err->localizedDescription()->utf8String();
            }
        }
        std::cerr << "\n            state: gaussians=" << maxGaussians
                  << " maxPairs=" << maxPairs
                  << " tiles=" << numTilesX << "x" << numTilesY
                  << " res=" << currentWidth << "x" << currentHeight
                  << std::endl;
        return false;
    }
    return true;
}

// Ensure buffers have enough capacity
void TiledRasterizer::ensureBufferCapacity(uint32_t width, uint32_t height, size_t gaussianCount) {
    // Compute required tiles
    uint32_t newNumTilesX = (width + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t newNumTilesY = (height + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t newMaxTiles = newNumTilesX * newNumTilesY;
    uint32_t numPixels = width * height;
    
    // Check if reallocation needed
    bool needsTileRealloc = (newMaxTiles > maxTiles);
    bool needsPixelRealloc = (numPixels > currentWidth * currentHeight);
    bool needsGaussianRealloc = (gaussianCount > maxGaussians);
    
    // Reallocate if needed
    if (needsTileRealloc) {
        if (tileRanges) tileRanges->release();
        maxTiles = newMaxTiles;
        tileRanges = device->newBuffer(maxTiles * sizeof(TileRange), MTL::ResourceStorageModeShared);
    }

    // Per-Gaussian buffers
    if (needsGaussianRealloc) {
        if (projectedGaussians) projectedGaussians->release();
        projectedGaussians = device->newBuffer(gaussianCount * sizeof(ProjectedGaussian),
                                               MTL::ResourceStorageModeShared);

        if (renderGradientBuffer) renderGradientBuffer->release();
        renderGradientBuffer = device->newBuffer(gaussianCount * sizeof(RenderGradients),
                                                 MTL::ResourceStorageModeShared);

        maxGaussians = (uint32_t)gaussianCount;
        std::cout << "Tiled rasterizer buffers grown to " << maxGaussians << " gaussians" << std::endl;
    }
    
    // Per-pixel last index buffer
    if (needsPixelRealloc || !perPixelLastIdx) {
        if (perPixelLastIdx) perPixelLastIdx->release();
        perPixelLastIdx = device->newBuffer(numPixels * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    }

    // SSIM gradient buffers 3 floats per pixel for each coefficient map + final gradient
    if (needsPixelRealloc || !ssimCoeffKBuffer) {
        size_t pixelFloatBufSize = numPixels * 3 * sizeof(float);
        if (ssimCoeffKBuffer) ssimCoeffKBuffer->release();
        if (ssimCoeffLBuffer) ssimCoeffLBuffer->release();
        if (ssimCoeffMBuffer) ssimCoeffMBuffer->release();
        if (pixelGradientBuffer) pixelGradientBuffer->release();
        ssimCoeffKBuffer = device->newBuffer(pixelFloatBufSize, MTL::ResourceStorageModeShared);
        ssimCoeffLBuffer = device->newBuffer(pixelFloatBufSize, MTL::ResourceStorageModeShared);
        ssimCoeffMBuffer = device->newBuffer(pixelFloatBufSize, MTL::ResourceStorageModeShared);
        pixelGradientBuffer = device->newBuffer(pixelFloatBufSize, MTL::ResourceStorageModeShared);
    }

    // Update current sizes
    currentWidth = width;
    currentHeight = height;
    numTilesX = newNumTilesX;
    numTilesY = newNumTilesY;
}

// Ensure pairs buffer has enough capacity
void TiledRasterizer::ensurePairsCapacity(uint32_t requiredPairs) {
    // Don't allocate more than 100M pairs to prevent corruption issues
    if (requiredPairs > 100000000) {
        std::cerr << "WARNING: requiredPairs too large (" << requiredPairs << "), clamping to 100M" << std::endl;
        requiredPairs = 100000000;
    }
    
    // Check if reallocation needed
    if (requiredPairs <= maxPairs) return;
    
    // Grow by 1.5x or to required size, whichever is larger
    uint32_t newMaxPairs = std::max(requiredPairs, (uint32_t)(maxPairs * 1.5));

    // Reallocate buffers
    if (gaussianKeys) gaussianKeys->release();
    if (gaussianValues) gaussianValues->release();
    
    // Update maxPairs
    maxPairs = newMaxPairs;
    gaussianKeys = device->newBuffer(maxPairs * sizeof(uint64_t), MTL::ResourceStorageModeShared);
    gaussianValues = device->newBuffer(maxPairs * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    // Also ensure GPU radix sort has enough capacity
    if (gpuRadixSort) {
        delete gpuRadixSort;
        gpuRadixSort = new GPURadixSort64(device, library, maxPairs);
    }
}

// Forward pass
void TiledRasterizer::forward(MTL::CommandQueue* queue,
                               MTL::Buffer* gaussianBuffer,
                               size_t gaussianCount,
                               const TiledUniforms& uniforms,
                               MTL::Texture* outputTexture) {
    
    // Start timing the forward pass
    auto forwardStart = std::chrono::high_resolution_clock::now();

    // Reset per-step failure flag; forward is the first pass of a training step, so this
    // scopes the flag to forward+backward of a single iteration.
    passFailed = false;

    // Get output texture size
    uint32_t width = (uint32_t)outputTexture->width();
    uint32_t height = (uint32_t)outputTexture->height();
    
    // ensure buffers have enough capacity
    ensureBufferCapacity(width, height, gaussianCount);
    
    // Print input Gaussian info 
    static bool inputDebugPrinted = false;
    // Print only once
    if (!inputDebugPrinted) {
        Gaussian* gPtr = (Gaussian*)gaussianBuffer->contents();
        std::cout << "=== Input Gaussian Debug ===" << std::endl;
        for (int i = 0; i < std::min(3, (int)gaussianCount); i++) {
            Gaussian& g = gPtr[i];
            std::cout << "Gaussian " << i << ": pos=(" << g.position.x << "," << g.position.y << "," << g.position.z
                      << ") scale(log)=(" << g.scale.x << "," << g.scale.y << "," << g.scale.z
                      << ") rot=(" << g.rotation.x << "," << g.rotation.y << "," << g.rotation.z << "," << g.rotation.w
                      << ") opacity=" << g.opacity << std::endl;
        }
        inputDebugPrinted = true;
    }
    
    // Copy uniforms
    TiledUniforms u = uniforms;
    u.numTilesX = numTilesX;
    u.numTilesY = numTilesY;
    u.numGaussians = (uint32_t)gaussianCount;
    memcpy(uniformBuffer->contents(), &u, sizeof(TiledUniforms));
    
    // Initialize per-pixel data
    uint32_t numPixels = width * height;
    uint32_t* lastIdxPtr = (uint32_t*)perPixelLastIdx->contents();
    memset(lastIdxPtr, 0xFF, numPixels * sizeof(uint32_t));
    
    auto t1 = std::chrono::high_resolution_clock::now();
    
    // Reset atomic counter for GPU pair generation
    *((uint32_t*)pairCounterBuffer->contents()) = 0;
    
    // Ensure buffer capacity estimate based on gaussian count
    uint32_t estimatedPairs = (uint32_t)(gaussianCount * AVG_TILES_PER_GAUSSIAN);
    ensurePairsCapacity(estimatedPairs);
    
    // Project Pair Generation in single command buffer
    MTL::CommandBuffer* cmdBuffer = queue->commandBuffer();
    
    // Project all Gaussians
    {
        // Project Gaussians to screen space
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(projectGaussiansPSO);
        enc->setBuffer(gaussianBuffer, 0, 0);
        enc->setBuffer(projectedGaussians, 0, 1);
        enc->setBuffer(uniformBuffer, 0, 2);
        
        // Set number of Gaussians
        NS::UInteger threadGroupSize = projectGaussiansPSO->maxTotalThreadsPerThreadgroup();
        if (threadGroupSize > 256) threadGroupSize = 256;
        
        // Dispatch threads
        MTL::Size grid = MTL::Size(gaussianCount, 1, 1);
        MTL::Size threadgroup = MTL::Size(threadGroupSize, 1, 1);
        enc->dispatchThreads(grid, threadgroup);
        enc->endEncoding();
    }
    
    // GPU Pair Generation same command buffer so no sync needed
    {
        // Generate pairs on GPU
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(generatePairsPSO);
        enc->setBuffer(projectedGaussians, 0, 0);
        enc->setBuffer(gaussianKeys, 0, 1);
        enc->setBuffer(gaussianValues, 0, 2);
        enc->setBuffer(pairCounterBuffer, 0, 3);
        
        // Set parameters
        uint32_t numG = (uint32_t)gaussianCount;
        enc->setBytes(&numG, sizeof(uint32_t), 4);
        enc->setBytes(&numTilesX, sizeof(uint32_t), 5);
        enc->setBytes(&maxPairs, sizeof(uint32_t), 6);
        
        // Dispatch threads
        NS::UInteger threadGroupSize = generatePairsPSO->maxTotalThreadsPerThreadgroup();
        if (threadGroupSize > 256) threadGroupSize = 256;
        
        // Dispatch threads
        MTL::Size grid = MTL::Size(gaussianCount, 1, 1);
        MTL::Size threadgroup = MTL::Size(threadGroupSize, 1, 1);
        enc->dispatchThreads(grid, threadgroup);
        enc->endEncoding();
    }
    
    // Submit and wait need results for CPU sorting
    // A failure here means projectedGaussians / the pair counter are garbage, so the
    // rest of the frame would sort and render nonsense. Bail out immediately.
    if (!commitAndCheck(cmdBuffer, "forward:project+pairgen")) {
        memset(tileRanges->contents(), 0, maxTiles * sizeof(TileRange));
        return;
    }

    // Timing after projection pair generation
    auto t2 = std::chrono::high_resolution_clock::now();
    
    // CPU-side tile binning and sorting
    ProjectedGaussian* projPtr = (ProjectedGaussian*)projectedGaussians->contents();
    
    // DEBUG Print some projected Gaussian info
    static bool debugPrinted = false;
    if (!debugPrinted) {
        // Print radius and tile info for first 100 Gaussians
        Gaussian* gPtr = (Gaussian*)gaussianBuffer->contents();
        int validCount = 0;
        float avgRadius = 0, minRadius = 1e10, maxRadius = 0;
        float minDepth = 1e10, maxDepth = 0;
        
        std::cout << "\n=== Radius & Tile Debug ===" << std::endl;
        
        for (uint32_t i = 0; i < std::min((uint32_t)100, (uint32_t)gaussianCount); i++) {
            const ProjectedGaussian& p = projPtr[i];
            if (p.radius > 0) {
                avgRadius += p.radius;
                minRadius = std::min(minRadius, p.radius);
                maxRadius = std::max(maxRadius, p.radius);
                if (p.depth > 0) {
                    minDepth = std::min(minDepth, p.depth);
                    maxDepth = std::max(maxDepth, p.depth);
                }
                validCount++;
                
                // Detailed debug for first 5 valid Gaussians
                if (validCount <= 5) {
                    const Gaussian& g = gPtr[i];
                    float worldScaleMax = std::max({expf(g.scale.x), expf(g.scale.y), expf(g.scale.z)});
                    
                    // Expected tile bounds from radius
                    int expectedMinTileX = std::max(0, int(p.screenPos.x - p.radius) / 16);
                    int expectedMaxTileX = std::min(int(numTilesX - 1), int(p.screenPos.x + p.radius) / 16);
                    int expectedMinTileY = std::max(0, int(p.screenPos.y - p.radius) / 16);
                    int expectedMaxTileY = std::min(int(numTilesY - 1), int(p.screenPos.y + p.radius) / 16);
                    uint32_t expectedTiles = (expectedMaxTileX - expectedMinTileX + 1) * (expectedMaxTileY - expectedMinTileY + 1);
                    uint32_t actualTiles = (p.tileMaxX - p.tileMinX + 1) * (p.tileMaxY - p.tileMinY + 1);
                    
                    std::cout << "Gaussian " << i << ":" << std::endl;
                    std::cout << "  World scale (exp): (" << expf(g.scale.x) << "," << expf(g.scale.y) << "," << expf(g.scale.z) << ") max=" << worldScaleMax << std::endl;
                    std::cout << "  Screen pos: (" << p.screenPos.x << "," << p.screenPos.y << ") depth=" << p.depth << std::endl;
                    std::cout << "  Screen radius (pixels): " << p.radius << " (should be ~3*sqrt(eigenvalue) of 2D cov)" << std::endl;
                    std::cout << "  Cov2D: (" << p.cov2D[0] << "," << p.cov2D[1] << "," << p.cov2D[2] << ")" << std::endl;
                    std::cout << "  Tile bounds: (" << p.tileMinX << "," << p.tileMinY << ") to (" << p.tileMaxX << "," << p.tileMaxY << ") = " << actualTiles << " tiles" << std::endl;
                    std::cout << "  Expected tiles: (" << expectedMinTileX << "," << expectedMinTileY << ") to (" << expectedMaxTileX << "," << expectedMaxTileY << ") = " << expectedTiles << " tiles" << std::endl;
                    
                    // Radius should be related to depth
                    float roughScreenScale = worldScaleMax / p.depth;
                    std::cout << "  Rough screen scale (worldMax/depth): " << roughScreenScale << std::endl;
                }
            }
        }
        
        // Summary
        if (validCount > 0) {
            std::cout << "\nSummary (first 100 Gaussians):" << std::endl;
            std::cout << "  Valid: " << validCount << std::endl;
            std::cout << "  Radius range: " << minRadius << " to " << maxRadius << " pixels (avg=" << (avgRadius/validCount) << ")" << std::endl;
            std::cout << "  Depth range: " << minDepth << " to " << maxDepth << std::endl;
        }
        debugPrinted = true;
        std::cout << "=== End Radius Debug ===\n" << std::endl;
    }
    
    // Timing after projection
    auto t3 = std::chrono::high_resolution_clock::now();
    
    // Read total pairs from atomic counter GPU pair gen already ran in batched command buffer
    uint32_t totalPairs = *((uint32_t*)pairCounterBuffer->contents());

    // If pair buffer overflowed, grow and re-run pair generation projection is still valid
    if (totalPairs > maxPairs) {
        ensurePairsCapacity(totalPairs);

        // Reset counter and re-run pair generation with larger buffer
        *((uint32_t*)pairCounterBuffer->contents()) = 0;
        MTL::CommandBuffer* retryCmdBuffer = queue->commandBuffer();
        {
            MTL::ComputeCommandEncoder* enc = retryCmdBuffer->computeCommandEncoder();
            enc->setComputePipelineState(generatePairsPSO);
            enc->setBuffer(projectedGaussians, 0, 0);
            enc->setBuffer(gaussianKeys, 0, 1);
            enc->setBuffer(gaussianValues, 0, 2);
            enc->setBuffer(pairCounterBuffer, 0, 3);
            uint32_t numG = (uint32_t)gaussianCount;
            enc->setBytes(&numG, sizeof(uint32_t), 4);
            enc->setBytes(&numTilesX, sizeof(uint32_t), 5);
            enc->setBytes(&maxPairs, sizeof(uint32_t), 6);
            NS::UInteger tgSize = generatePairsPSO->maxTotalThreadsPerThreadgroup();
            if (tgSize > 256) tgSize = 256;
            enc->dispatchThreads(MTL::Size(gaussianCount, 1, 1), MTL::Size(tgSize, 1, 1));
            enc->endEncoding();
        }
        if (!commitAndCheck(retryCmdBuffer, "forward:pairgen-retry")) {
            memset(tileRanges->contents(), 0, maxTiles * sizeof(TileRange));
            return;
        }
        totalPairs = *((uint32_t*)pairCounterBuffer->contents());
        if (totalPairs > maxPairs) totalPairs = maxPairs;
    }

    if (totalPairs == 0) {
        // Clear tile ranges
        memset(tileRanges->contents(), 0, maxTiles * sizeof(TileRange));
        return;
    }
    
    uint32_t pairCount = totalPairs;
    
    // Choose which buffers to use based on sorting method
    MTL::Buffer* sortedKeysBuffer = gaussianKeys;
    MTL::Buffer* sortedValuesBuffer = gaussianValues;
    
    if (USE_GPU_SORT) {
        // GPU sort - entirely on GPU
        gpuRadixSort->sort(queue, gaussianKeys, gaussianValues, totalPairs);
        // Use the GPU's internal sorted buffers directly
        sortedKeysBuffer = gpuRadixSort->getSortedKeys();
        sortedValuesBuffer = gpuRadixSort->getSortedValues();

        // Store active sorted buffers for backward pass GPU sort case
        activeSortedKeys = sortedKeysBuffer;
        activeSortedValues = sortedValuesBuffer;
    } else {
        // CPU sort slower, requires GPU->CPU->GPU copies
        // Get pointers to buffers
        uint64_t* gpuKeys = (uint64_t*)gaussianKeys->contents();
        uint32_t* gpuValues = (uint32_t*)gaussianValues->contents();
        
        // Copy GPU-generated pairs to CPU vector for sorting
        static std::vector<std::pair<uint64_t, uint32_t>> pairs;
        pairs.resize(totalPairs);
        
        // Populate pairs vector
        for (uint32_t i = 0; i < totalPairs; i++) {
            pairs[i] = {gpuKeys[i], gpuValues[i]};
        }
        
        // Sort by key using fast CPU radix sort
        parallelRadixSort(pairs);
        
        // Copy sorted data back to GPU buffers
        for (uint32_t i = 0; i < pairCount; i++) {
            gpuKeys[i] = pairs[i].first;
            gpuValues[i] = pairs[i].second;
        }
        
        // Use the original buffers as sorted buffers
        activeSortedKeys = gaussianKeys;
        activeSortedValues = gaussianValues;
    }
    
    // Get pointers for debug printing use the active sorted buffers
    uint64_t* gpuKeys = (uint64_t*)activeSortedKeys->contents();
    uint32_t* gpuValues = (uint32_t*)activeSortedValues->contents();
    
    auto t6 = std::chrono::high_resolution_clock::now();
    
    // Build tile ranges and render in single command buffer
    cmdBuffer = queue->commandBuffer();
    
    // Build tile ranges on GPU using parallel binary search
    {
        // Setup compute encoder and buffers use sorted buffers
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(buildTileRangesPSO);
        enc->setBuffer(activeSortedKeys, 0, 0);
        enc->setBuffer(tileRanges, 0, 1);
        enc->setBytes(&pairCount, sizeof(uint32_t), 2);
        enc->setBytes(&maxTiles, sizeof(uint32_t), 3);
        
        // Dispatch threads
        NS::UInteger threadGroupSize = buildTileRangesPSO->maxTotalThreadsPerThreadgroup();
        if (threadGroupSize > 256) threadGroupSize = 256;
        
        // Dispatch threads
        MTL::Size grid = MTL::Size(maxTiles, 1, 1);
        MTL::Size threadgroup = MTL::Size(threadGroupSize, 1, 1);
        enc->dispatchThreads(grid, threadgroup);
        enc->endEncoding();
    }
    
    // GPU forward rendering same command buffer no sync needed
    {
        // Setup compute encoder and buffers
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(tiledForwardPSO);
        enc->setBuffer(gaussianBuffer, 0, 0);
        enc->setBuffer(projectedGaussians, 0, 1);
        // sortedIndices use active sorted values buffer correct for both GPU and CPU sort
        enc->setBuffer(activeSortedValues, 0, 2);  
        enc->setBuffer(tileRanges, 0, 3);
        enc->setBuffer(uniformBuffer, 0, 4);
        enc->setBuffer(perPixelLastIdx, 0, 5);
        enc->setTexture(outputTexture, 0);
        
        // Dispatch threads
        MTL::Size grid = MTL::Size(width, height, 1);
        MTL::Size threadgroup = MTL::Size(TILE_SIZE, TILE_SIZE, 1);
        enc->dispatchThreads(grid, threadgroup);
        enc->endEncoding();
    }
    
    // Submit and wait for render to complete
    commitAndCheck(cmdBuffer, "forward:tileranges+render");

    auto t7 = std::chrono::high_resolution_clock::now();
    
    // Check tile ranges for gaps/overlaps after render completes
    TileRange* ranges = (TileRange*)tileRanges->contents();
    static bool tileRangeDebugPrinted = false;
    // Print only once
    if (!tileRangeDebugPrinted) {
        std::cout << "\n=== Tile Range Debug ===" << std::endl;
        std::cout << "Total pairs: " << pairCount << ", Total tiles: " << maxTiles << std::endl;
        
        uint32_t totalCoverage = 0;
        uint32_t tilesWithData = 0;
        uint32_t maxCount = 0;
        uint32_t lastEnd = 0;
        bool foundGap = false;
        bool foundOverlap = false;
        
        // Iterate over tiles and print ranges
        for (uint32_t tile = 0; tile < maxTiles && tile < 10000; tile++) {
            if (ranges[tile].count > 0) {
                tilesWithData++;
                totalCoverage += ranges[tile].count;
                maxCount = std::max(maxCount, ranges[tile].count);
                
                // Check for gaps non-contiguous ranges
                if (tilesWithData > 1 && ranges[tile].start != lastEnd) {
                    if (!foundGap) {
                        std::cout << "WARNING: Gap/overlap at tile " << tile 
                                  << " (expected start=" << lastEnd << ", actual start=" << ranges[tile].start << ")" << std::endl;
                    }
                    foundGap = true;
                }
                lastEnd = ranges[tile].start + ranges[tile].count;
                
                // Print first few tiles with data
                if (tilesWithData <= 5) {
                    std::cout << "Tile " << tile << ": start=" << ranges[tile].start 
                              << " count=" << ranges[tile].count << std::endl;
                }
            }
        }
        
        // Print some sample pairs to verify sorting
        std::cout << "\nFirst 10 pairs (tileIdx, gaussianIdx):" << std::endl;
        for (uint32_t i = 0; i < std::min(10u, pairCount); i++) {
            uint32_t tileId = (uint32_t)(gpuKeys[i] >> 32);
            uint32_t depthBits = (uint32_t)(gpuKeys[i] & 0xFFFFFFFF);
            std::cout << "  [" << i << "] tile=" << tileId << " gaussian=" << gpuValues[i] 
                      << " depthKey=0x" << std::hex << depthBits << std::dec << std::endl;
        }
        
        // Summary
        std::cout << "\nSummary:" << std::endl;
        std::cout << "  Tiles with Gaussians: " << tilesWithData << "/" << maxTiles << std::endl;
        std::cout << "  Total coverage: " << totalCoverage << " (should equal pairCount=" << pairCount << ")" << std::endl;
        std::cout << "  Max Gaussians per tile: " << maxCount << std::endl;
        if (totalCoverage != pairCount) {
            std::cout << "  ERROR: Coverage mismatch! Lost " << (pairCount - totalCoverage) << " pairs!" << std::endl;
        }
        std::cout << "=== End Tile Range Debug ===\n" << std::endl;
        
        tileRangeDebugPrinted = true;
    }
    
    // Print timing stats periodically
    static int frameCount = 0;
    static double totalProjectPairGenTime = 0, totalSortTime = 0;
    static double totalRangeRenderTime = 0, totalForwardTime = 0;
    
    // Project and PairGen batched
    double projectPairGenMs = std::chrono::duration<double, std::milli>(t2 - t1).count(); 
    // Sort and copy
    double sortMs = std::chrono::duration<double, std::milli>(t6 - t3).count();  
    // Range and Render batched
    double rangeRenderMs = std::chrono::duration<double, std::milli>(t7 - t6).count();  
    // Total forward
    double forwardMs = std::chrono::duration<double, std::milli>(t7 - forwardStart).count();
    
    // Accumulate
    totalProjectPairGenTime += projectPairGenMs;
    totalSortTime += sortMs;
    totalRangeRenderTime += rangeRenderMs;
    totalForwardTime += forwardMs;
    frameCount++;
    
    // Print every 100 frames
    if (frameCount == 100) {
        const char* sortMethod = USE_GPU_SORT ? "GPU" : "CPU";
        printf("\n=== Forward Pass Timing (avg over %d frames) ===\n", frameCount);
        printf("  Project+PairGen (GPU):  %6.2f ms\n", totalProjectPairGenTime / frameCount);
        printf("  Sort (%s):              %6.2f ms\n", sortMethod, totalSortTime / frameCount);
        printf("  Range+Render (GPU):     %6.2f ms\n", totalRangeRenderTime / frameCount);
        printf("  TOTAL:                  %6.2f ms\n", totalForwardTime / frameCount);
        printf("  Pairs/frame:            ~%u\n", totalPairs);
        
        frameCount = 0;
        totalProjectPairGenTime = totalSortTime = totalRangeRenderTime = totalForwardTime = 0;
    }
}

// Backward pass combined L1 + D-SSIM gradient
void TiledRasterizer::backward(MTL::CommandQueue* queue,
                                MTL::Buffer* gaussianBuffer,
                                MTL::Buffer* gradientBuffer,
                                size_t gaussianCount,
                                const TiledUniforms& uniforms,
                                MTL::Texture* renderedTexture,
                                MTL::Texture* groundTruthTexture) {

    uint32_t width = (uint32_t)renderedTexture->width();
    uint32_t height = (uint32_t)renderedTexture->height();

    // Ensure SSIM gradient buffers are allocated
    ensureBufferCapacity(width, height, gaussianCount);

    // Copy uniforms small, CPU is fine
    TiledUniforms u = uniforms;
    u.numTilesX = numTilesX;
    u.numTilesY = numTilesY;
    u.numGaussians = (uint32_t)gaussianCount;
    memcpy(uniformBuffer->contents(), &u, sizeof(TiledUniforms));

    // Ensure renderGradientBuffer is large enough
    if (!renderGradientBuffer || renderGradientBuffer->length() < gaussianCount * sizeof(RenderGradients)) {
        if (renderGradientBuffer) renderGradientBuffer->release();
        renderGradientBuffer = device->newBuffer(gaussianCount * sizeof(RenderGradients),
                                                  MTL::ResourceStorageModeShared);
    }

    MTL::CommandBuffer* cmdBuffer = queue->commandBuffer();

    // 1. Clear intermediate render gradient buffer AND final gradient buffer on GPU
    {
        MTL::BlitCommandEncoder* blit = cmdBuffer->blitCommandEncoder();
        blit->fillBuffer(renderGradientBuffer, NS::Range(0, gaussianCount * sizeof(RenderGradients)), 0);
        blit->fillBuffer(gradientBuffer, NS::Range(0, gaussianCount * sizeof(GaussianGradients)), 0);
        blit->endEncoding();
    }

    MTL::Size pixelGrid = MTL::Size(width, height, 1);
    MTL::Size pixelThreadgroup = MTL::Size(16, 16, 1);

    // 2. Compute SSIM gradient coefficient maps (K, L, M per pixel)
    {
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(computeSSIMGradCoeffsPSO);
        enc->setTexture(renderedTexture, 0);
        enc->setTexture(groundTruthTexture, 1);
        enc->setBuffer(ssimCoeffKBuffer, 0, 0);
        enc->setBuffer(ssimCoeffLBuffer, 0, 1);
        enc->setBuffer(ssimCoeffMBuffer, 0, 2);
        enc->dispatchThreads(pixelGrid, pixelThreadgroup);
        enc->endEncoding();
    }

    // 3. Convolve coefficients and combine with L1 to get per-pixel gradient
    {
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(computePixelGradientPSO);
        enc->setTexture(renderedTexture, 0);
        enc->setTexture(groundTruthTexture, 1);
        enc->setBuffer(ssimCoeffKBuffer, 0, 0);
        enc->setBuffer(ssimCoeffLBuffer, 0, 1);
        enc->setBuffer(ssimCoeffMBuffer, 0, 2);
        enc->setBuffer(pixelGradientBuffer, 0, 3);
        enc->dispatchThreads(pixelGrid, pixelThreadgroup);
        enc->endEncoding();
    }

    // 4. Stage 1: Tiled backward — accumulates 9 intermediate gradients per Gaussian
    //    (dL_dColor[3], dL_dConic[3], dL_dOpacity[1], dL_dMean2D[2])
    {
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(tiledBackwardPSO);
        enc->setBuffer(projectedGaussians, 0, 0);
        enc->setBuffer(renderGradientBuffer, 0, 1);
        enc->setBuffer(activeSortedValues, 0, 2);
        enc->setBuffer(tileRanges, 0, 3);
        enc->setBuffer(uniformBuffer, 0, 4);
        enc->setBuffer(perPixelLastIdx, 0, 5);
        enc->setBuffer(pixelGradientBuffer, 0, 6);

        MTL::Size grid = MTL::Size(width, height, 1);
        MTL::Size threadgroup = MTL::Size(TILE_SIZE, TILE_SIZE, 1);
        enc->dispatchThreads(grid, threadgroup);
        enc->endEncoding();
    }

    // 5. Stage 2: Per-Gaussian preprocess — computes final gradients (no atomics)
    //    SH, scale, rotation, position, viewspace gradients
    {
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(preprocessBackwardPSO);
        enc->setBuffer(gaussianBuffer, 0, 0);
        enc->setBuffer(projectedGaussians, 0, 1);
        enc->setBuffer(renderGradientBuffer, 0, 2);
        enc->setBuffer(gradientBuffer, 0, 3);
        enc->setBuffer(uniformBuffer, 0, 4);

        NS::UInteger threadGroupSize = preprocessBackwardPSO->maxTotalThreadsPerThreadgroup();
        if (threadGroupSize > 256) threadGroupSize = 256;
        MTL::Size grid = MTL::Size(gaussianCount, 1, 1);
        MTL::Size threadgroup = MTL::Size(threadGroupSize, 1, 1);
        enc->dispatchThreads(grid, threadgroup);
        enc->endEncoding();
    }

    commitAndCheck(cmdBuffer, "backward:ssim+tiled+preprocess");
}

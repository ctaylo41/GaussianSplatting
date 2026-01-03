//
//  tiled_rasterizer.mm
//  GaussianSplatting
//
//  GPU-accelerated tiled rasterizer with GPU sorting
//

#include "tiled_rasterizer.hpp"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <vector>

// ============================================================================
// Helper to create compute pipeline
// ============================================================================

static MTL::ComputePipelineState* createPipeline(MTL::Device* device,
                                                  MTL::Library* library,
                                                  const char* functionName) {
    NS::Error* error = nullptr;
    
    auto funcName = NS::String::string(functionName, NS::ASCIIStringEncoding);
    MTL::Function* func = library->newFunction(funcName);
    
    if (!func) {
        std::cerr << "Warning: Function not found: " << functionName << std::endl;
        return nullptr;
    }
    
    MTL::ComputePipelineState* pso = device->newComputePipelineState(func, &error);
    func->release();
    
    if (!pso) {
        std::cerr << "Failed to create pipeline for " << functionName;
        if (error) {
            std::cerr << ": " << error->localizedDescription()->utf8String();
        }
        std::cerr << std::endl;
        return nullptr;
    }
    
    return pso;
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

TiledRasterizer::TiledRasterizer(MTL::Device* device, MTL::Library* library, uint32_t maxGaussians)
    : device(device)
    , maxGaussians(maxGaussians)
    , maxTiles(0)
    , maxPairs(0)
    , currentWidth(0)
    , currentHeight(0)
    , numTilesX(0)
    , numTilesY(0)
    , lastPairCount(0)
    , lastSortTimeMs(0)
{
    createPipelines(library);
    
    // Allocate projection buffer
    projectedGaussians = device->newBuffer(maxGaussians * sizeof(ProjectedGaussian),
                                           MTL::ResourceStorageModeShared);
    
    // Uniform buffer
    uniformBuffer = device->newBuffer(sizeof(TiledUniforms), MTL::ResourceStorageModeShared);
    
    // Total pairs counter
    totalPairsBuffer = device->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    // Histogram for radix sort (256 buckets)
    histogramBuffer = device->newBuffer(256 * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    // Initial pair allocation
    maxPairs = maxGaussians * AVG_TILES_PER_GAUSSIAN;
    
    keysBuffer[0] = device->newBuffer(maxPairs * sizeof(uint64_t), MTL::ResourceStorageModeShared);
    keysBuffer[1] = device->newBuffer(maxPairs * sizeof(uint64_t), MTL::ResourceStorageModeShared);
    valuesBuffer[0] = device->newBuffer(maxPairs * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    valuesBuffer[1] = device->newBuffer(maxPairs * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    // Other buffers allocated on demand
    tileCountsBuffer = nullptr;
    tileOffsetsBuffer = nullptr;
    tileWriteIdxBuffer = nullptr;
    tileRanges = nullptr;
    perPixelTransmittance = nullptr;
    perPixelLastIdx = nullptr;
}

TiledRasterizer::~TiledRasterizer() {
    if (projectedGaussians) projectedGaussians->release();
    if (tileCountsBuffer) tileCountsBuffer->release();
    if (tileOffsetsBuffer) tileOffsetsBuffer->release();
    if (tileWriteIdxBuffer) tileWriteIdxBuffer->release();
    if (totalPairsBuffer) totalPairsBuffer->release();
    if (keysBuffer[0]) keysBuffer[0]->release();
    if (keysBuffer[1]) keysBuffer[1]->release();
    if (valuesBuffer[0]) valuesBuffer[0]->release();
    if (valuesBuffer[1]) valuesBuffer[1]->release();
    if (histogramBuffer) histogramBuffer->release();
    if (tileRanges) tileRanges->release();
    if (perPixelTransmittance) perPixelTransmittance->release();
    if (perPixelLastIdx) perPixelLastIdx->release();
    if (uniformBuffer) uniformBuffer->release();
    
    if (projectGaussiansPSO) projectGaussiansPSO->release();
    if (tiledForwardPSO) tiledForwardPSO->release();
    if (tiledBackwardPSO) tiledBackwardPSO->release();
    if (countTilePairsPSO) countTilePairsPSO->release();
    if (generateTileKeysPSO) generateTileKeysPSO->release();
    if (histogram64PSO) histogram64PSO->release();
    if (prefixSum256PSO) prefixSum256PSO->release();
    if (scatter64SimplePSO) scatter64SimplePSO->release();
    if (buildTileRangesPSO) buildTileRangesPSO->release();
    if (clearHistogramPSO) clearHistogramPSO->release();
}

void TiledRasterizer::createPipelines(MTL::Library* library) {
    // Main rasterization pipelines
    projectGaussiansPSO = createPipeline(device, library, "projectGaussians");
    tiledForwardPSO = createPipeline(device, library, "tiledForward");
    tiledBackwardPSO = createPipeline(device, library, "tiledBackward");
    
    // GPU sorting pipelines (from gpu_sort.metal)
    countTilePairsPSO = createPipeline(device, library, "countTilePairs");
    generateTileKeysPSO = createPipeline(device, library, "generateTileKeys");
    histogram64PSO = createPipeline(device, library, "histogram64");
    prefixSum256PSO = createPipeline(device, library, "prefixSum256");
    scatter64SimplePSO = createPipeline(device, library, "scatter64Simple");
    buildTileRangesPSO = createPipeline(device, library, "buildTileRanges");
    clearHistogramPSO = createPipeline(device, library, "clearHistogram");
    
    // Check if GPU sort is available
    if (!countTilePairsPSO || !generateTileKeysPSO || !histogram64PSO ||
        !scatter64SimplePSO || !buildTileRangesPSO) {
        std::cout << "GPU sorting pipelines not available, falling back to CPU sort" << std::endl;
        useGPUSort = false;
    }
}

void TiledRasterizer::ensureBufferCapacity(uint32_t width, uint32_t height) {
    uint32_t tilesX = (width + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t tilesY = (height + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t numTiles = tilesX * tilesY;
    
    if (numTiles <= maxTiles && width <= currentWidth && height <= currentHeight) {
        numTilesX = tilesX;
        numTilesY = tilesY;
        return;
    }
    
    currentWidth = width;
    currentHeight = height;
    numTilesX = tilesX;
    numTilesY = tilesY;
    maxTiles = numTiles;
    
    std::cout << "Resizing tile buffers: " << tilesX << "x" << tilesY
              << " = " << numTiles << " tiles" << std::endl;
    
    // Tile counting buffers
    if (tileCountsBuffer) tileCountsBuffer->release();
    if (tileOffsetsBuffer) tileOffsetsBuffer->release();
    if (tileWriteIdxBuffer) tileWriteIdxBuffer->release();
    if (tileRanges) tileRanges->release();
    
    tileCountsBuffer = device->newBuffer(numTiles * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    tileOffsetsBuffer = device->newBuffer(numTiles * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    tileWriteIdxBuffer = device->newBuffer(numTiles * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    tileRanges = device->newBuffer(numTiles * sizeof(TileRange), MTL::ResourceStorageModeShared);
    
    // Per-pixel buffers
    uint32_t numPixels = width * height;
    if (perPixelTransmittance) perPixelTransmittance->release();
    if (perPixelLastIdx) perPixelLastIdx->release();
    
    perPixelTransmittance = device->newBuffer(numPixels * sizeof(float), MTL::ResourceStorageModeShared);
    perPixelLastIdx = device->newBuffer(numPixels * sizeof(uint32_t), MTL::ResourceStorageModeShared);
}

void TiledRasterizer::ensurePairsCapacity(uint32_t requiredPairs) {
    if (requiredPairs <= maxPairs) return;
    
    uint32_t newMaxPairs = std::max(requiredPairs, (uint32_t)(maxPairs * 1.5));
    
    std::cout << "Growing pairs buffer: " << maxPairs << " -> " << newMaxPairs << std::endl;
    
    if (keysBuffer[0]) keysBuffer[0]->release();
    if (keysBuffer[1]) keysBuffer[1]->release();
    if (valuesBuffer[0]) valuesBuffer[0]->release();
    if (valuesBuffer[1]) valuesBuffer[1]->release();
    
    maxPairs = newMaxPairs;
    keysBuffer[0] = device->newBuffer(maxPairs * sizeof(uint64_t), MTL::ResourceStorageModeShared);
    keysBuffer[1] = device->newBuffer(maxPairs * sizeof(uint64_t), MTL::ResourceStorageModeShared);
    valuesBuffer[0] = device->newBuffer(maxPairs * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    valuesBuffer[1] = device->newBuffer(maxPairs * sizeof(uint32_t), MTL::ResourceStorageModeShared);
}

// ============================================================================
// GPU Sorting Implementation
// ============================================================================

void TiledRasterizer::gpuSort(MTL::CommandQueue* queue, size_t gaussianCount) {
    static bool debugPrinted = false;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    uint32_t numGaussiansU32 = (uint32_t)gaussianCount;
    uint32_t maxTilesPerGaussian = MAX_TILES_PER_GAUSSIAN;
    uint32_t numTiles = numTilesX * numTilesY;
    
    // Debug: Check projected Gaussians
    if (!debugPrinted) {
        ProjectedGaussian* proj = (ProjectedGaussian*)projectedGaussians->contents();
        int validCount = 0;
        for (size_t i = 0; i < std::min(gaussianCount, (size_t)100); i++) {
            if (proj[i].radius > 0) validCount++;
        }
        printf("DEBUG gpuSort: %d/%d projected Gaussians have radius > 0\n", validCount, (int)std::min(gaussianCount, (size_t)100));
        if (validCount > 0) {
            for (size_t i = 0; i < gaussianCount && i < 5; i++) {
                if (proj[i].radius > 0) {
                    printf("  Gaussian %zu: screenPos=(%.1f,%.1f) radius=%.1f tiles=(%u,%u)-(%u,%u)\n",
                           i, proj[i].screenPos.x, proj[i].screenPos.y, proj[i].radius,
                           proj[i].tileMinX, proj[i].tileMinY, proj[i].tileMaxX, proj[i].tileMaxY);
                }
            }
        }
    }
    
    // Clear tile counts
    memset(tileCountsBuffer->contents(), 0, numTiles * sizeof(uint32_t));
    memset(tileWriteIdxBuffer->contents(), 0, numTiles * sizeof(uint32_t));
    *(uint32_t*)totalPairsBuffer->contents() = 0;
    
    MTL::CommandBuffer* cmdBuffer = queue->commandBuffer();
    
    // Step 1: Count tile pairs
    {
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(countTilePairsPSO);
        enc->setBuffer(projectedGaussians, 0, 0);
        enc->setBuffer(tileCountsBuffer, 0, 1);
        enc->setBuffer(totalPairsBuffer, 0, 2);
        enc->setBytes(&numGaussiansU32, sizeof(uint32_t), 3);
        enc->setBytes(&numTilesX, sizeof(uint32_t), 4);
        enc->setBytes(&maxTilesPerGaussian, sizeof(uint32_t), 5);
        
        MTL::Size grid = MTL::Size(gaussianCount, 1, 1);
        MTL::Size tg = MTL::Size(THREADGROUP_SIZE, 1, 1);
        enc->dispatchThreads(grid, tg);
        enc->endEncoding();
    }
    
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();
    
    // Get total pairs
    uint32_t totalPairs = *(uint32_t*)totalPairsBuffer->contents();
    lastPairCount = totalPairs;
    
    if (!debugPrinted) {
        printf("DEBUG gpuSort: totalPairs = %u\n", totalPairs);
        debugPrinted = true;
    }
    
    if (totalPairs == 0) {
        TileRange* ranges = (TileRange*)tileRanges->contents();
        memset(ranges, 0, numTiles * sizeof(TileRange));
        lastSortTimeMs = 0;
        return;
    }
    
    // Ensure capacity
    ensurePairsCapacity(totalPairs);
    
    // Step 2: CPU prefix sum on tile counts
    {
        uint32_t* counts = (uint32_t*)tileCountsBuffer->contents();
        uint32_t* offsets = (uint32_t*)tileOffsetsBuffer->contents();
        uint32_t sum = 0;
        for (uint32_t i = 0; i < numTiles; i++) {
            offsets[i] = sum;
            sum += counts[i];
        }
    }
    
    // Step 3: Generate keys
    cmdBuffer = queue->commandBuffer();
    {
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(generateTileKeysPSO);
        enc->setBuffer(projectedGaussians, 0, 0);
        enc->setBuffer(tileOffsetsBuffer, 0, 1);
        enc->setBuffer(tileWriteIdxBuffer, 0, 2);
        enc->setBuffer(keysBuffer[0], 0, 3);
        enc->setBuffer(valuesBuffer[0], 0, 4);
        enc->setBytes(&numGaussiansU32, sizeof(uint32_t), 5);
        enc->setBytes(&numTilesX, sizeof(uint32_t), 6);
        enc->setBytes(&maxTilesPerGaussian, sizeof(uint32_t), 7);
        
        MTL::Size grid = MTL::Size(gaussianCount, 1, 1);
        MTL::Size tg = MTL::Size(THREADGROUP_SIZE, 1, 1);
        enc->dispatchThreads(grid, tg);
        enc->endEncoding();
    }
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();
    
    // Step 4: 64-bit radix sort (8 passes)
    int srcIdx = 0;
    for (uint32_t pass = 0; pass < 8; pass++) {
        uint32_t bitOffset = pass * 8;
        int dstIdx = 1 - srcIdx;
        
        // Clear histogram
        memset(histogramBuffer->contents(), 0, 256 * sizeof(uint32_t));
        
        cmdBuffer = queue->commandBuffer();
        
        // Histogram
        {
            MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
            enc->setComputePipelineState(histogram64PSO);
            enc->setBuffer(keysBuffer[srcIdx], 0, 0);
            enc->setBuffer(histogramBuffer, 0, 1);
            enc->setBytes(&bitOffset, sizeof(uint32_t), 2);
            enc->setBytes(&totalPairs, sizeof(uint32_t), 3);
            
            MTL::Size grid = MTL::Size(totalPairs, 1, 1);
            MTL::Size tg = MTL::Size(THREADGROUP_SIZE, 1, 1);
            enc->dispatchThreads(grid, tg);
            enc->endEncoding();
        }
        
        cmdBuffer->commit();
        cmdBuffer->waitUntilCompleted();
        
        // CPU prefix sum on histogram
        {
            uint32_t* hist = (uint32_t*)histogramBuffer->contents();
            uint32_t sum = 0;
            for (int i = 0; i < 256; i++) {
                uint32_t count = hist[i];
                hist[i] = sum;
                sum += count;
            }
        }
        
        // Scatter
        cmdBuffer = queue->commandBuffer();
        {
            MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
            enc->setComputePipelineState(scatter64SimplePSO);
            enc->setBuffer(keysBuffer[srcIdx], 0, 0);
            enc->setBuffer(valuesBuffer[srcIdx], 0, 1);
            enc->setBuffer(keysBuffer[dstIdx], 0, 2);
            enc->setBuffer(valuesBuffer[dstIdx], 0, 3);
            enc->setBuffer(histogramBuffer, 0, 4);
            enc->setBytes(&bitOffset, sizeof(uint32_t), 5);
            enc->setBytes(&totalPairs, sizeof(uint32_t), 6);
            
            MTL::Size grid = MTL::Size(totalPairs, 1, 1);
            MTL::Size tg = MTL::Size(THREADGROUP_SIZE, 1, 1);
            enc->dispatchThreads(grid, tg);
            enc->endEncoding();
        }
        cmdBuffer->commit();
        cmdBuffer->waitUntilCompleted();
        
        srcIdx = dstIdx;
    }
    
    // Step 5: Build tile ranges
    cmdBuffer = queue->commandBuffer();
    {
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(buildTileRangesPSO);
        enc->setBuffer(keysBuffer[srcIdx], 0, 0);
        enc->setBuffer(tileRanges, 0, 1);
        enc->setBytes(&totalPairs, sizeof(uint32_t), 2);
        enc->setBytes(&numTiles, sizeof(uint32_t), 3);
        
        MTL::Size grid = MTL::Size(numTiles, 1, 1);
        MTL::Size tg = MTL::Size(std::min(numTiles, THREADGROUP_SIZE), 1, 1);
        enc->dispatchThreads(grid, tg);
        enc->endEncoding();
    }
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();
    
    // Copy sorted values to buffer 0 if needed
    if (srcIdx != 0) {
        memcpy(valuesBuffer[0]->contents(), valuesBuffer[srcIdx]->contents(),
               totalPairs * sizeof(uint32_t));
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    lastSortTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
}

// ============================================================================
// CPU Sorting (fallback)
// ============================================================================

void TiledRasterizer::cpuSort(size_t gaussianCount) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    ProjectedGaussian* projPtr = (ProjectedGaussian*)projectedGaussians->contents();
    
    // Build pairs
    std::vector<std::pair<uint64_t, uint32_t>> pairs;
    pairs.reserve(gaussianCount * AVG_TILES_PER_GAUSSIAN);
    
    for (uint32_t gIdx = 0; gIdx < gaussianCount; gIdx++) {
        const ProjectedGaussian& p = projPtr[gIdx];
        if (p.radius <= 0 || p.tileMinX > p.tileMaxX || p.tileMinY > p.tileMaxY) continue;
        
        uint32_t tilesX = p.tileMaxX - p.tileMinX + 1;
        uint32_t tilesY = p.tileMaxY - p.tileMinY + 1;
        if (tilesX * tilesY > MAX_TILES_PER_GAUSSIAN) continue;
        
        // Convert depth to sortable bits
        uint32_t depthBits = *reinterpret_cast<const uint32_t*>(&p.depth);
        depthBits = (depthBits & 0x80000000) ? ~depthBits : (depthBits | 0x80000000);
        
        for (uint32_t ty = p.tileMinY; ty <= p.tileMaxY; ty++) {
            for (uint32_t tx = p.tileMinX; tx <= p.tileMaxX; tx++) {
                uint32_t tileIdx = ty * numTilesX + tx;
                uint64_t key = (uint64_t(tileIdx) << 32) | uint64_t(depthBits);
                pairs.emplace_back(key, gIdx);
            }
        }
    }
    
    lastPairCount = (uint32_t)pairs.size();
    
    if (pairs.empty()) {
        TileRange* ranges = (TileRange*)tileRanges->contents();
        memset(ranges, 0, maxTiles * sizeof(TileRange));
        lastSortTimeMs = 0;
        return;
    }
    
    ensurePairsCapacity((uint32_t)pairs.size());
    
    // Sort
    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
    
    // Copy to buffers
    uint64_t* keys = (uint64_t*)keysBuffer[0]->contents();
    uint32_t* values = (uint32_t*)valuesBuffer[0]->contents();
    TileRange* ranges = (TileRange*)tileRanges->contents();
    
    memset(ranges, 0, maxTiles * sizeof(TileRange));
    
    uint32_t pairCount = (uint32_t)pairs.size();
    for (uint32_t i = 0; i < pairCount; i++) {
        keys[i] = pairs[i].first;
        values[i] = pairs[i].second;
    }
    
    // Build tile ranges
    if (pairCount > 0) {
        uint32_t currentTile = (uint32_t)(pairs[0].first >> 32);
        uint32_t rangeStart = 0;
        
        for (uint32_t i = 1; i <= pairCount; i++) {
            uint32_t tile = (i < pairCount) ? (uint32_t)(pairs[i].first >> 32) : UINT32_MAX;
            if (tile != currentTile) {
                if (currentTile < maxTiles) {
                    ranges[currentTile].start = rangeStart;
                    ranges[currentTile].count = i - rangeStart;
                }
                currentTile = tile;
                rangeStart = i;
            }
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    lastSortTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
}

// ============================================================================
// Forward Pass
// ============================================================================

void TiledRasterizer::forward(MTL::CommandQueue* queue,
                               MTL::Buffer* gaussianBuffer,
                               size_t gaussianCount,
                               const TiledUniforms& uniforms,
                               MTL::Texture* outputTexture) {
    uint32_t width = (uint32_t)outputTexture->width();
    uint32_t height = (uint32_t)outputTexture->height();
    
    ensureBufferCapacity(width, height);
    
    // Update uniforms
    TiledUniforms u = uniforms;
    u.numTilesX = numTilesX;
    u.numTilesY = numTilesY;
    u.numGaussians = (uint32_t)gaussianCount;
    memcpy(uniformBuffer->contents(), &u, sizeof(TiledUniforms));
    
    // Step 1: Project Gaussians
    MTL::CommandBuffer* cmdBuffer = queue->commandBuffer();
    {
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(projectGaussiansPSO);
        enc->setBuffer(gaussianBuffer, 0, 0);
        enc->setBuffer(projectedGaussians, 0, 1);
        enc->setBuffer(uniformBuffer, 0, 2);
        
        MTL::Size grid = MTL::Size(gaussianCount, 1, 1);
        MTL::Size tg = MTL::Size(THREADGROUP_SIZE, 1, 1);
        enc->dispatchThreads(grid, tg);
        enc->endEncoding();
    }
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();
    
    // Step 2: Sort (GPU or CPU)
    if (useGPUSort && countTilePairsPSO && generateTileKeysPSO && histogram64PSO &&
        scatter64SimplePSO && buildTileRangesPSO) {
        gpuSort(queue, gaussianCount);
    } else {
        cpuSort(gaussianCount);
    }
    
    // Step 3: Render
    cmdBuffer = queue->commandBuffer();
    {
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(tiledForwardPSO);
        enc->setBuffer(gaussianBuffer, 0, 0);
        enc->setBuffer(projectedGaussians, 0, 1);
        enc->setBuffer(valuesBuffer[0], 0, 2);  // Sorted indices
        enc->setBuffer(tileRanges, 0, 3);
        enc->setBuffer(uniformBuffer, 0, 4);
        enc->setBuffer(perPixelLastIdx, 0, 5);  // lastContribIdx in shader
        enc->setTexture(outputTexture, 0);
        
        MTL::Size grid = MTL::Size(width, height, 1);
        MTL::Size threadgroup = MTL::Size(TILE_SIZE, TILE_SIZE, 1);
        enc->dispatchThreads(grid, threadgroup);
        enc->endEncoding();
    }
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();
}

// ============================================================================
// Backward Pass
// ============================================================================

void TiledRasterizer::backward(MTL::CommandQueue* queue,
                                MTL::Buffer* gaussianBuffer,
                                MTL::Buffer* gradientBuffer,
                                size_t gaussianCount,
                                const TiledUniforms& uniforms,
                                MTL::Texture* renderedTexture,
                                MTL::Texture* groundTruthTexture) {
    uint32_t width = (uint32_t)renderedTexture->width();
    uint32_t height = (uint32_t)renderedTexture->height();
    
    MTL::CommandBuffer* cmdBuffer = queue->commandBuffer();
    {
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(tiledBackwardPSO);
        enc->setBuffer(gaussianBuffer, 0, 0);
        enc->setBuffer(gradientBuffer, 0, 1);
        enc->setBuffer(projectedGaussians, 0, 2);
        enc->setBuffer(valuesBuffer[0], 0, 3);  // Sorted indices
        enc->setBuffer(tileRanges, 0, 4);
        enc->setBuffer(uniformBuffer, 0, 5);
        enc->setBuffer(perPixelLastIdx, 0, 6);
        enc->setTexture(renderedTexture, 0);
        enc->setTexture(groundTruthTexture, 1);
        
        MTL::Size grid = MTL::Size(width, height, 1);
        MTL::Size threadgroup = MTL::Size(TILE_SIZE, TILE_SIZE, 1);
        enc->dispatchThreads(grid, threadgroup);
        enc->endEncoding();
    }
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();
}

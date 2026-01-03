//
//  tiled_rasterizer.mm
//  GaussianSplatting
//
//  FIXED: Proper buffer initialization and GPU sort
//  Key fixes:
//  1. sortedIndicesBuffer is always initialized with identity permutation
//  2. Buffers are properly allocated before first use
//  3. After radix sort, copy result to sortedIndicesBuffer
//  4. Use consistent buffer for rendering
//

#include "tiled_rasterizer.hpp"
#include "ply_loader.hpp"  // For Gaussian struct
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstddef>
#include <vector>
#include <chrono>

// Helper to create compute pipeline
static MTL::ComputePipelineState* createPipeline(MTL::Device* device,
                                                  MTL::Library* library,
                                                  const char* functionName) {
    NS::Error* error = nullptr;
    auto funcName = NS::String::string(functionName, NS::ASCIIStringEncoding);
    MTL::Function* func = library->newFunction(funcName);
    
    if (!func) {
        std::cerr << "Failed to find function: " << functionName << std::endl;
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
    
    // Verify struct alignment - should be 88 bytes
    printf("sizeof(ProjectedGaussian) = %zu bytes (expected 88)\n", sizeof(ProjectedGaussian));
    printf("  offsetof(screenPos) = %zu (expected 0)\n", offsetof(ProjectedGaussian, screenPos));
    printf("  offsetof(conic) = %zu (expected 8)\n", offsetof(ProjectedGaussian, conic));
    printf("  offsetof(depth) = %zu (expected 20)\n", offsetof(ProjectedGaussian, depth));
    printf("  offsetof(opacity) = %zu (expected 24)\n", offsetof(ProjectedGaussian, opacity));
    printf("  offsetof(color) = %zu (expected 28)\n", offsetof(ProjectedGaussian, color));
    printf("  offsetof(radius) = %zu (expected 40)\n", offsetof(ProjectedGaussian, radius));
    printf("  offsetof(tileMinX) = %zu (expected 44)\n", offsetof(ProjectedGaussian, tileMinX));
    printf("  offsetof(tileMinY) = %zu (expected 48)\n", offsetof(ProjectedGaussian, tileMinY));
    printf("  offsetof(tileMaxX) = %zu (expected 52)\n", offsetof(ProjectedGaussian, tileMaxX));
    printf("  offsetof(tileMaxY) = %zu (expected 56)\n", offsetof(ProjectedGaussian, tileMaxY));
    printf("  offsetof(viewPos_xy) = %zu (expected 64)\n", offsetof(ProjectedGaussian, viewPos_xy));
    printf("  offsetof(cov2D) = %zu (expected 72)\n", offsetof(ProjectedGaussian, cov2D));
    
    // Allocate projection buffer
    projectedGaussians = device->newBuffer(maxGaussians * sizeof(ProjectedGaussian),
                                           MTL::ResourceStorageModeShared);
    
    // Uniform buffer
    uniformBuffer = device->newBuffer(sizeof(TiledUniforms), MTL::ResourceStorageModeShared);
    
    // Histogram buffer for radix sort (256 buckets for 8-bit radix)
    histogramBuffer = device->newBuffer(256 * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    // CRITICAL: Initialize pairs buffers with reasonable default size
    // This prevents nullptr access on first frame
    maxPairs = maxGaussians * AVG_TILES_PER_GAUSSIAN;
    
    keysBuffer[0] = device->newBuffer(maxPairs * sizeof(uint64_t), MTL::ResourceStorageModeShared);
    keysBuffer[1] = device->newBuffer(maxPairs * sizeof(uint64_t), MTL::ResourceStorageModeShared);
    valuesBuffer[0] = device->newBuffer(maxPairs * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    valuesBuffer[1] = device->newBuffer(maxPairs * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    // CRITICAL: Initialize value buffers with identity permutation
    // This ensures any unwritten positions still contain valid indices
    uint32_t* vals0 = (uint32_t*)valuesBuffer[0]->contents();
    uint32_t* vals1 = (uint32_t*)valuesBuffer[1]->contents();
    for (uint32_t i = 0; i < maxPairs; i++) {
        vals0[i] = i % maxGaussians;  // Valid index even if not overwritten
        vals1[i] = i % maxGaussians;
    }
    
    // CRITICAL: sortedIndicesBuffer is the single source of truth for rendering
    // Initialize with identity permutation so iteration 0 works even if no pairs are generated
    sortedIndicesBuffer = device->newBuffer(maxPairs * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    uint32_t* initIndices = (uint32_t*)sortedIndicesBuffer->contents();
    for (uint32_t i = 0; i < maxPairs; i++) {
        initIndices[i] = i % maxGaussians;
    }
    
    std::cout << "TiledRasterizer initialized: maxGaussians=" << maxGaussians 
              << ", initial maxPairs=" << maxPairs << std::endl;
}

TiledRasterizer::~TiledRasterizer() {
    if (projectedGaussians) projectedGaussians->release();
    if (keysBuffer[0]) keysBuffer[0]->release();
    if (keysBuffer[1]) keysBuffer[1]->release();
    if (valuesBuffer[0]) valuesBuffer[0]->release();
    if (valuesBuffer[1]) valuesBuffer[1]->release();
    if (sortedIndicesBuffer) sortedIndicesBuffer->release();
    if (histogramBuffer) histogramBuffer->release();
    if (tileRanges) tileRanges->release();
    if (perPixelLastIdx) perPixelLastIdx->release();
    if (uniformBuffer) uniformBuffer->release();
    
    if (projectGaussiansPSO) projectGaussiansPSO->release();
    if (tiledForwardPSO) tiledForwardPSO->release();
    if (tiledBackwardPSO) tiledBackwardPSO->release();
    if (histogram64PSO) histogram64PSO->release();
    if (scatter64SimplePSO) scatter64SimplePSO->release();
    if (buildTileRangesPSO) buildTileRangesPSO->release();
}

void TiledRasterizer::createPipelines(MTL::Library* library) {
    projectGaussiansPSO = createPipeline(device, library, "projectGaussians");
    tiledForwardPSO = createPipeline(device, library, "tiledForward");
    tiledBackwardPSO = createPipeline(device, library, "tiledBackward");
    
    // GPU sort pipelines (optional - will fall back to CPU if these fail)
    histogram64PSO = createPipeline(device, library, "histogram64");
    scatter64SimplePSO = createPipeline(device, library, "scatter64Simple");
    buildTileRangesPSO = createPipeline(device, library, "buildTileRanges");
    
    if (!projectGaussiansPSO) {
        std::cerr << "CRITICAL: projectGaussiansPSO is null!" << std::endl;
    }
    if (!tiledForwardPSO) {
        std::cerr << "CRITICAL: tiledForwardPSO is null!" << std::endl;
    }
    if (!tiledBackwardPSO) {
        std::cerr << "CRITICAL: tiledBackwardPSO is null!" << std::endl;
    }
    
    // Report GPU sort status
    bool gpuSortAvailable = (histogram64PSO && scatter64SimplePSO && buildTileRangesPSO);
    std::cout << "GPU sort available: " << (gpuSortAvailable ? "YES" : "NO (using CPU fallback)") << std::endl;
}

void TiledRasterizer::ensureBufferCapacity(uint32_t width, uint32_t height) {
    uint32_t newNumTilesX = (width + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t newNumTilesY = (height + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t newMaxTiles = newNumTilesX * newNumTilesY;
    uint32_t numPixels = width * height;
    
    bool needsTileRealloc = (newMaxTiles > maxTiles);
    bool needsPixelRealloc = (numPixels > currentWidth * currentHeight) || !perPixelLastIdx;
    
    if (needsTileRealloc) {
        if (tileRanges) tileRanges->release();
        maxTiles = newMaxTiles;
        tileRanges = device->newBuffer(maxTiles * sizeof(TileRange), MTL::ResourceStorageModeShared);
        
        // Initialize tile ranges to zero
        memset(tileRanges->contents(), 0, maxTiles * sizeof(TileRange));
        
        std::cout << "Resized tile buffers: " << newNumTilesX << "x" << newNumTilesY
                  << " = " << newMaxTiles << " tiles" << std::endl;
    }
    
    if (needsPixelRealloc) {
        if (perPixelLastIdx) perPixelLastIdx->release();
        perPixelLastIdx = device->newBuffer(numPixels * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    }
    
    currentWidth = width;
    currentHeight = height;
    numTilesX = newNumTilesX;
    numTilesY = newNumTilesY;
}

void TiledRasterizer::ensurePairsCapacity(uint32_t requiredPairs) {
    // Sanity check - don't allocate more than 100M pairs
    if (requiredPairs > 100000000) {
        std::cerr << "WARNING: requiredPairs too large (" << requiredPairs << "), clamping to 100M" << std::endl;
        requiredPairs = 100000000;
    }
    
    if (requiredPairs <= maxPairs) return;
    
    // Grow by 1.5x or to required size, whichever is larger
    uint32_t newMaxPairs = std::max(requiredPairs, (uint32_t)(maxPairs * 1.5));
    
    std::cout << "Growing pairs buffer: " << maxPairs << " -> " << newMaxPairs << std::endl;
    
    if (keysBuffer[0]) keysBuffer[0]->release();
    if (keysBuffer[1]) keysBuffer[1]->release();
    if (valuesBuffer[0]) valuesBuffer[0]->release();
    if (valuesBuffer[1]) valuesBuffer[1]->release();
    if (sortedIndicesBuffer) sortedIndicesBuffer->release();
    
    maxPairs = newMaxPairs;
    keysBuffer[0] = device->newBuffer(maxPairs * sizeof(uint64_t), MTL::ResourceStorageModeShared);
    keysBuffer[1] = device->newBuffer(maxPairs * sizeof(uint64_t), MTL::ResourceStorageModeShared);
    valuesBuffer[0] = device->newBuffer(maxPairs * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    valuesBuffer[1] = device->newBuffer(maxPairs * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    sortedIndicesBuffer = device->newBuffer(maxPairs * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    // Initialize value buffers with valid indices
    uint32_t* vals0 = (uint32_t*)valuesBuffer[0]->contents();
    uint32_t* vals1 = (uint32_t*)valuesBuffer[1]->contents();
    uint32_t* initIndices = (uint32_t*)sortedIndicesBuffer->contents();
    for (uint32_t i = 0; i < maxPairs; i++) {
        vals0[i] = i % maxGaussians;
        vals1[i] = i % maxGaussians;
        initIndices[i] = i % maxGaussians;
    }
}

// ============================================================================
// GPU Sorting Implementation
// ============================================================================

void TiledRasterizer::gpuSort(MTL::CommandQueue* queue, size_t gaussianCount) {
    static bool debugPrinted = false;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    uint32_t numTiles = numTilesX * numTilesY;
    
    // Debug: Check projected Gaussians
    ProjectedGaussian* projPtr = (ProjectedGaussian*)projectedGaussians->contents();
    
    if (!debugPrinted) {
        int validCount = 0;
        int sampleSize = (int)std::min(gaussianCount, (size_t)100);
        for (int i = 0; i < sampleSize; i++) {
            if (projPtr[i].radius > 0) validCount++;
        }
        printf("DEBUG gpuSort: %d/%d projected Gaussians have radius > 0\n", validCount, sampleSize);
        if (validCount > 0) {
            for (size_t i = 0; i < std::min(gaussianCount, (size_t)3); i++) {
                if (projPtr[i].radius > 0) {
                    printf("  Gaussian %zu: screenPos=(%.1f,%.1f) radius=%.1f tiles=(%u,%u)-(%u,%u) depth=%.3f\n",
                           i, projPtr[i].screenPos.x, projPtr[i].screenPos.y, projPtr[i].radius,
                           projPtr[i].tileMinX, projPtr[i].tileMinY, projPtr[i].tileMaxX, projPtr[i].tileMaxY, projPtr[i].depth);
                }
            }
        }
    }
    
    // Step 1: Count tile-Gaussian pairs
    uint32_t totalPairs = 0;
    
    for (uint32_t i = 0; i < gaussianCount; i++) {
        const ProjectedGaussian& p = projPtr[i];
        if (p.radius <= 0 || p.tileMinX > p.tileMaxX || p.tileMinY > p.tileMaxY) continue;
        
        uint32_t tilesX = p.tileMaxX - p.tileMinX + 1;
        uint32_t tilesY = p.tileMaxY - p.tileMinY + 1;
        uint32_t pairCount = tilesX * tilesY;
        
        if (pairCount > MAX_TILES_PER_GAUSSIAN) continue;
        totalPairs += pairCount;
    }
    
    lastPairCount = totalPairs;
    
    if (!debugPrinted) {
        printf("DEBUG gpuSort: totalPairs = %u\n", totalPairs);
    }
    
    // CRITICAL FIX: Even if totalPairs == 0, we need valid buffers for rendering
    // Clear tile ranges and ensure sortedIndicesBuffer has identity permutation
    if (totalPairs == 0) {
        memset(tileRanges->contents(), 0, numTiles * sizeof(TileRange));
        // sortedIndicesBuffer already has identity from initialization
        debugPrinted = true;
        
        auto endTime = std::chrono::high_resolution_clock::now();
        lastSortTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        return;
    }
    
    ensurePairsCapacity(totalPairs);
    
    // Step 2: Generate key-value pairs on CPU
    uint64_t* keys = (uint64_t*)keysBuffer[0]->contents();
    uint32_t* values = (uint32_t*)valuesBuffer[0]->contents();
    uint32_t pairIdx = 0;
    
    for (uint32_t gIdx = 0; gIdx < gaussianCount; gIdx++) {
        const ProjectedGaussian& p = projPtr[gIdx];
        if (p.radius <= 0 || p.tileMinX > p.tileMaxX || p.tileMinY > p.tileMaxY) continue;
        
        uint32_t tilesX = p.tileMaxX - p.tileMinX + 1;
        uint32_t tilesY = p.tileMaxY - p.tileMinY + 1;
        if (tilesX * tilesY > MAX_TILES_PER_GAUSSIAN) continue;
        
        // Convert depth to sortable bits (front-to-back order)
        uint32_t depthBits = *reinterpret_cast<const uint32_t*>(&p.depth);
        // For positive floats, flip sign bit for sorting
        depthBits = (depthBits & 0x80000000) ? ~depthBits : (depthBits | 0x80000000);
        
        for (uint32_t ty = p.tileMinY; ty <= p.tileMaxY; ty++) {
            for (uint32_t tx = p.tileMinX; tx <= p.tileMaxX; tx++) {
                uint32_t tileIdx = ty * numTilesX + tx;
                // Key = tile index (high 32 bits) + depth (low 32 bits)
                keys[pairIdx] = ((uint64_t)tileIdx << 32) | depthBits;
                values[pairIdx] = gIdx;
                pairIdx++;
            }
        }
    }
    
    if (!debugPrinted) {
        printf("DEBUG: Generated %u pairs\n", pairIdx);
        if (pairIdx > 0) {
            printf("  First 3 pairs:\n");
            for (uint32_t i = 0; i < std::min(pairIdx, 3u); i++) {
                uint32_t tile = (uint32_t)(keys[i] >> 32);
                uint32_t depth = (uint32_t)(keys[i] & 0xFFFFFFFF);
                printf("    [%u]: tile=%u, depth=0x%08X, gaussianIdx=%u\n", i, tile, depth, values[i]);
            }
        }
    }
    
    // Note: For StorageModeShared buffers, CPU writes are immediately visible to GPU
    // No synchronization needed on Apple Silicon unified memory
    
    // Step 3: 64-bit radix sort (8 passes of 8-bit sort)
    int srcIdx = 0;
    
    for (uint32_t pass = 0; pass < 8; pass++) {
        uint32_t bitOffset = pass * 8;
        int dstIdx = 1 - srcIdx;
        
        // Clear histogram using CPU (guaranteed correct)
        memset(histogramBuffer->contents(), 0, 256 * sizeof(uint32_t));
        
        MTL::CommandBuffer* cmdBuffer = queue->commandBuffer();
        
        // Build histogram
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
        
        // CPU prefix sum (simple and correct)
        {
            uint32_t* hist = (uint32_t*)histogramBuffer->contents();
            
            // Debug: check histogram on first pass
            if (!debugPrinted && pass == 0) {
                uint32_t totalCount = 0;
                for (int i = 0; i < 256; i++) {
                    totalCount += hist[i];
                }
                printf("DEBUG: Pass 0 histogram total = %u (expected %u)\n", totalCount, totalPairs);
                if (totalCount != totalPairs) {
                    printf("  ERROR: Histogram count mismatch!\n");
                }
            }
            
            uint32_t sum = 0;
            for (int i = 0; i < 256; i++) {
                uint32_t count = hist[i];
                hist[i] = sum;
                sum += count;
            }
        }
        
        // Note: For StorageModeShared, CPU writes are immediately visible to GPU
        
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
        
        // Debug: After each pass, verify the values are still valid indices
        if (!debugPrinted && pass == 7) {
            uint32_t* dstVals = (uint32_t*)valuesBuffer[dstIdx]->contents();
            bool allValid = true;
            int invalidCount = 0;
            for (uint32_t i = 0; i < totalPairs && invalidCount < 5; i++) {
                if (dstVals[i] >= gaussianCount) {
                    printf("  Pass %u: Invalid value[%u] = %u (gaussianCount=%zu)\n",
                           pass, i, dstVals[i], gaussianCount);
                    invalidCount++;
                    allValid = false;
                }
            }
            if (!allValid) {
                printf("  ERROR: Values corrupted after pass %u\n", pass);
            }
        }
        
        srcIdx = dstIdx;
    }
    
    // After 8 passes, srcIdx points to the buffer with sorted data
    // CRITICAL FIX: Copy sorted values to sortedIndicesBuffer for rendering
    uint32_t* sortedVals = (uint32_t*)valuesBuffer[srcIdx]->contents();
    uint32_t* outputIndices = (uint32_t*)sortedIndicesBuffer->contents();
    memcpy(outputIndices, sortedVals, totalPairs * sizeof(uint32_t));
    
    // Also get sorted keys for building tile ranges
    uint64_t* sortedKeys = (uint64_t*)keysBuffer[srcIdx]->contents();
    
    if (!debugPrinted) {
        printf("DEBUG: After sorting (final in buffer %d):\n", srcIdx);
        printf("  First 3 sorted pairs:\n");
        for (uint32_t i = 0; i < std::min(totalPairs, 3u); i++) {
            uint32_t tile = (uint32_t)(sortedKeys[i] >> 32);
            uint32_t depth = (uint32_t)(sortedKeys[i] & 0xFFFFFFFF);
            printf("    [%u]: tile=%u, depth=0x%08X, gaussianIdx=%u\n", i, tile, depth, outputIndices[i]);
        }
        
        // Verify keys are sorted
        bool keysSorted = true;
        for (uint32_t i = 1; i < totalPairs; i++) {
            if (sortedKeys[i] < sortedKeys[i-1]) {
                printf("  ERROR: Keys not sorted at index %u: key[%u]=0x%llX > key[%u]=0x%llX\n",
                       i, i-1, sortedKeys[i-1], i, sortedKeys[i]);
                keysSorted = false;
                break;
            }
        }
        if (keysSorted && totalPairs > 0) {
            printf("  Keys are correctly sorted\n");
        }
        
        // Verify Gaussian indices are valid
        bool allValid = true;
        for (uint32_t i = 0; i < totalPairs; i++) {
            if (outputIndices[i] >= gaussianCount) {
                printf("  ERROR: Invalid gaussianIdx=%u at sorted position %u\n", outputIndices[i], i);
                allValid = false;
                break;
            }
        }
        if (allValid) {
            printf("  All sorted Gaussian indices are valid\n");
        }
    }
    
    // Step 4: Build tile ranges from sorted keys
    // Use keysBuffer[srcIdx] which has the sorted keys
    MTL::CommandBuffer* cmdBuffer = queue->commandBuffer();
    {
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(buildTileRangesPSO);
        enc->setBuffer(keysBuffer[srcIdx], 0, 0);  // FIXED: Use correct sorted buffer
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
    
    // Debug: Check tile ranges
    if (!debugPrinted) {
        TileRange* ranges = (TileRange*)tileRanges->contents();
        uint32_t tilesWithData = 0;
        uint32_t totalCoverage = 0;
        
        for (uint32_t i = 0; i < numTiles; i++) {
            if (ranges[i].count > 0) {
                tilesWithData++;
                totalCoverage += ranges[i].count;
            }
        }
        
        printf("DEBUG: Tile ranges - %u tiles have data, totalCoverage=%u (expected %u)\n",
               tilesWithData, totalCoverage, totalPairs);
        
        if (tilesWithData > 0) {
            printf("  First tile with data:\n");
            for (uint32_t i = 0; i < numTiles; i++) {
                if (ranges[i].count > 0) {
                    printf("    tile %u: start=%u, count=%u\n", i, ranges[i].start, ranges[i].count);
                    break;
                }
            }
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    lastSortTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    
    debugPrinted = true;
}

// ============================================================================
// CPU Sorting (fallback)
// ============================================================================

void TiledRasterizer::cpuSort(size_t gaussianCount) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    ProjectedGaussian* projPtr = (ProjectedGaussian*)projectedGaussians->contents();
    uint32_t numTiles = numTilesX * numTilesY;
    
    // Build pairs
    std::vector<std::pair<uint64_t, uint32_t>> pairs;
    pairs.reserve(gaussianCount * AVG_TILES_PER_GAUSSIAN);
    
    for (uint32_t gIdx = 0; gIdx < gaussianCount; gIdx++) {
        const ProjectedGaussian& p = projPtr[gIdx];
        if (p.radius <= 0 || p.tileMinX > p.tileMaxX || p.tileMinY > p.tileMaxY) continue;
        
        uint32_t tilesX = p.tileMaxX - p.tileMinX + 1;
        uint32_t tilesY = p.tileMaxY - p.tileMinY + 1;
        if (tilesX * tilesY > MAX_TILES_PER_GAUSSIAN) continue;
        
        uint32_t depthBits = *reinterpret_cast<const uint32_t*>(&p.depth);
        depthBits = (depthBits & 0x80000000) ? ~depthBits : (depthBits | 0x80000000);
        
        for (uint32_t ty = p.tileMinY; ty <= p.tileMaxY; ty++) {
            for (uint32_t tx = p.tileMinX; tx <= p.tileMaxX; tx++) {
                uint32_t tileIdx = ty * numTilesX + tx;
                uint64_t key = (uint64_t(tileIdx) << 32) | depthBits;
                pairs.emplace_back(key, gIdx);
            }
        }
    }
    
    lastPairCount = (uint32_t)pairs.size();
    
    // Handle zero pairs case
    if (pairs.empty()) {
        memset(tileRanges->contents(), 0, maxTiles * sizeof(TileRange));
        auto endTime = std::chrono::high_resolution_clock::now();
        lastSortTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        return;
    }
    
    // Sort by key (tile + depth)
    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
    
    // Ensure buffer capacity
    ensurePairsCapacity((uint32_t)pairs.size());
    
    // Write sorted indices to sortedIndicesBuffer
    uint32_t* sortedIndices = (uint32_t*)sortedIndicesBuffer->contents();
    TileRange* ranges = (TileRange*)tileRanges->contents();
    memset(ranges, 0, maxTiles * sizeof(TileRange));
    
    uint32_t pairCount = (uint32_t)pairs.size();
    
    for (uint32_t i = 0; i < pairCount; i++) {
        sortedIndices[i] = pairs[i].second;
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
    
    // Copy uniforms
    TiledUniforms u = uniforms;
    u.numTilesX = numTilesX;
    u.numTilesY = numTilesY;
    u.numGaussians = (uint32_t)gaussianCount;
    u.screenSize = simd_make_float2((float)width, (float)height);
    memcpy(uniformBuffer->contents(), &u, sizeof(TiledUniforms));
    
    // Initialize per-pixel data
    uint32_t numPixels = width * height;
    uint32_t* lastIdxPtr = (uint32_t*)perPixelLastIdx->contents();
    memset(lastIdxPtr, 0xFF, numPixels * sizeof(uint32_t));
    
    MTL::CommandBuffer* cmdBuffer = queue->commandBuffer();
    
    // Step 1: Project all Gaussians
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
    bool gpuSortAvailable = (histogram64PSO && scatter64SimplePSO && buildTileRangesPSO);
    
    static bool sortDebugPrinted = false;
    if (!sortDebugPrinted) {
        printf("DEBUG forward: gpuSortAvailable=%d, useGPUSort=%d\n", gpuSortAvailable, useGPUSort);
        printf("  histogram64PSO=%p, scatter64SimplePSO=%p, buildTileRangesPSO=%p\n",
               (void*)histogram64PSO, (void*)scatter64SimplePSO, (void*)buildTileRangesPSO);
        sortDebugPrinted = true;
    }
    
    if (useGPUSort && gpuSortAvailable) {
        gpuSort(queue, gaussianCount);
    } else {
        cpuSort(gaussianCount);
    }
    
    // Step 3: Render
    // CRITICAL: Use sortedIndicesBuffer which is always populated correctly
    cmdBuffer = queue->commandBuffer();
    {
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(tiledForwardPSO);
        enc->setBuffer(gaussianBuffer, 0, 0);
        enc->setBuffer(projectedGaussians, 0, 1);
        enc->setBuffer(sortedIndicesBuffer, 0, 2);  // FIXED: Use dedicated sortedIndicesBuffer
        enc->setBuffer(tileRanges, 0, 3);
        enc->setBuffer(uniformBuffer, 0, 4);
        enc->setBuffer(perPixelLastIdx, 0, 5);
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
    
    // Clear gradients
    memset(gradientBuffer->contents(), 0, gaussianCount * sizeof(GaussianGradients));
    
    // Copy uniforms
    TiledUniforms u = uniforms;
    u.numTilesX = numTilesX;
    u.numTilesY = numTilesY;
    u.numGaussians = (uint32_t)gaussianCount;
    memcpy(uniformBuffer->contents(), &u, sizeof(TiledUniforms));
    
    MTL::CommandBuffer* cmdBuffer = queue->commandBuffer();
    
    {
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(tiledBackwardPSO);
        enc->setBuffer(gaussianBuffer, 0, 0);
        enc->setBuffer(gradientBuffer, 0, 1);
        enc->setBuffer(projectedGaussians, 0, 2);
        enc->setBuffer(sortedIndicesBuffer, 0, 3);  // FIXED: Use dedicated sortedIndicesBuffer
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
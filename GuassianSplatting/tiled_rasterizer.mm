//
//  tiled_rasterizer.mm
//  GuassianSplatting
//

#include "tiled_rasterizer.hpp"
#include "ply_loader.hpp"  // For Gaussian struct
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstddef>
#include <vector>
#include <dispatch/dispatch.h>  // Apple's GCD for parallel operations
#include <chrono>
#include <atomic>

// Number of threads for parallel operations
static const int NUM_THREADS = 8;

// FAST parallel radix sort using GCD
// Optimized with parallel histogram and parallel scatter
static void parallelRadixSort(std::vector<std::pair<uint64_t, uint32_t>>& pairs) {
    const size_t n = pairs.size();
    if (n < 1000) {
        std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        return;
    }
    
    static std::vector<std::pair<uint64_t, uint32_t>> temp;
    if (temp.size() < n) temp.resize(n);
    
    // Per-thread histograms
    static uint32_t threadHist[NUM_THREADS][256];
    static uint32_t globalPrefix[256];
    static uint32_t threadOffset[NUM_THREADS][256];
    
    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
    
    auto* src = &pairs;
    auto* dst = &temp;
    
    // 8 passes of 8-bit radix sort
    for (int pass = 0; pass < 8; pass++) {
        int shift = pass * 8;
        size_t chunkSize = (n + NUM_THREADS - 1) / NUM_THREADS;
        
        // Step 1: Parallel histogram (each thread counts its chunk)
        dispatch_apply(NUM_THREADS, queue, ^(size_t t) {
            memset(threadHist[t], 0, 256 * sizeof(uint32_t));
            size_t start = t * chunkSize;
            size_t end = std::min(start + chunkSize, n);
            for (size_t i = start; i < end; i++) {
                uint8_t digit = ((*src)[i].first >> shift) & 0xFF;
                threadHist[t][digit]++;
            }
        });
        
        // Step 2: Compute global prefix sums and per-thread offsets
        uint32_t sum = 0;
        for (int d = 0; d < 256; d++) {
            globalPrefix[d] = sum;
            for (int t = 0; t < NUM_THREADS; t++) {
                threadOffset[t][d] = sum;
                sum += threadHist[t][d];
            }
        }
        
        // Step 3: Parallel scatter (each thread writes its chunk)
        dispatch_apply(NUM_THREADS, queue, ^(size_t t) {
            uint32_t localOffset[256];
            memcpy(localOffset, threadOffset[t], 256 * sizeof(uint32_t));
            
            size_t start = t * chunkSize;
            size_t end = std::min(start + chunkSize, n);
            for (size_t i = start; i < end; i++) {
                uint8_t digit = ((*src)[i].first >> shift) & 0xFF;
                (*dst)[localOffset[digit]++] = (*src)[i];
            }
        });
        
        std::swap(src, dst);
    }
    
    // Ensure result is in pairs
    if (src != &pairs) {
        pairs = *src;
    }
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
    totalPairsBuffer = device->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    // Start with reasonable default - will grow if needed
    maxPairs = maxGaussians * AVG_TILES_PER_GAUSSIAN;
    gaussianKeys = device->newBuffer(maxPairs * sizeof(uint64_t), MTL::ResourceStorageModeShared);
    gaussianValues = device->newBuffer(maxPairs * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    // Other buffers allocated on demand
    tileRanges = nullptr;
    perPixelLastIdx = nullptr;
}

TiledRasterizer::~TiledRasterizer() {
    if (projectedGaussians) projectedGaussians->release();
    if (gaussianKeys) gaussianKeys->release();
    if (gaussianValues) gaussianValues->release();
    if (tileRanges) tileRanges->release();
    if (totalPairsBuffer) totalPairsBuffer->release();
    if (perPixelLastIdx) perPixelLastIdx->release();
    if (uniformBuffer) uniformBuffer->release();
    
    if (projectGaussiansPSO) projectGaussiansPSO->release();
    if (tiledForwardPSO) tiledForwardPSO->release();
    if (tiledBackwardPSO) tiledBackwardPSO->release();
    if (buildTileRangesPSO) buildTileRangesPSO->release();
}

void TiledRasterizer::createPipelines(MTL::Library* library) {
    NS::Error* error = nullptr;
    
    auto makePipeline = [&](const char* name) -> MTL::ComputePipelineState* {
        MTL::Function* func = library->newFunction(NS::String::string(name, NS::ASCIIStringEncoding));
        if (!func) {
            std::cerr << "Failed to find function: " << name << std::endl;
            return nullptr;
        }
        MTL::ComputePipelineState* pso = device->newComputePipelineState(func, &error);
        func->release();
        if (!pso) {
            std::cerr << "Failed to create pipeline " << name << ": "
                      << error->localizedDescription()->utf8String() << std::endl;
        }
        return pso;
    };
    
    projectGaussiansPSO = makePipeline("projectGaussians");
    tiledForwardPSO = makePipeline("tiledForward");
    tiledBackwardPSO = makePipeline("tiledBackward");
    buildTileRangesPSO = makePipeline("buildTileRanges");
}

void TiledRasterizer::ensureBufferCapacity(uint32_t width, uint32_t height, size_t gaussianCount) {
    uint32_t newNumTilesX = (width + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t newNumTilesY = (height + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t newMaxTiles = newNumTilesX * newNumTilesY;
    uint32_t numPixels = width * height;
    
    bool needsTileRealloc = (newMaxTiles > maxTiles);
    bool needsPixelRealloc = (numPixels > currentWidth * currentHeight);
    
    if (needsTileRealloc) {
        if (tileRanges) tileRanges->release();
        maxTiles = newMaxTiles;
        tileRanges = device->newBuffer(maxTiles * sizeof(TileRange), MTL::ResourceStorageModeShared);
    }
    
    if (needsPixelRealloc || !perPixelLastIdx) {
        if (perPixelLastIdx) perPixelLastIdx->release();
        perPixelLastIdx = device->newBuffer(numPixels * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    }
    
    currentWidth = width;
    currentHeight = height;
    numTilesX = newNumTilesX;
    numTilesY = newNumTilesY;
}

void TiledRasterizer::ensurePairsCapacity(uint32_t requiredPairs) {
    // Sanity check - don't allocate more than 100M pairs (prevents corruption issues)
    if (requiredPairs > 100000000) {
        std::cerr << "WARNING: requiredPairs too large (" << requiredPairs << "), clamping to 100M" << std::endl;
        requiredPairs = 100000000;
    }
    
    if (requiredPairs <= maxPairs) return;
    
    // Grow by 1.5x or to required size, whichever is larger
    uint32_t newMaxPairs = std::max(requiredPairs, (uint32_t)(maxPairs * 1.5));
    
    std::cout << "Growing pairs buffer: " << maxPairs << " -> " << newMaxPairs << std::endl;
    
    if (gaussianKeys) gaussianKeys->release();
    if (gaussianValues) gaussianValues->release();
    
    maxPairs = newMaxPairs;
    gaussianKeys = device->newBuffer(maxPairs * sizeof(uint64_t), MTL::ResourceStorageModeShared);
    gaussianValues = device->newBuffer(maxPairs * sizeof(uint32_t), MTL::ResourceStorageModeShared);
}

void TiledRasterizer::forward(MTL::CommandQueue* queue,
                               MTL::Buffer* gaussianBuffer,
                               size_t gaussianCount,
                               const TiledUniforms& uniforms,
                               MTL::Texture* outputTexture) {
    
    auto forwardStart = std::chrono::high_resolution_clock::now();
    
    uint32_t width = (uint32_t)outputTexture->width();
    uint32_t height = (uint32_t)outputTexture->height();
    
    ensureBufferCapacity(width, height, gaussianCount);
    
    // DEBUG: Print a few input Gaussians (once)
    static bool inputDebugPrinted = false;
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
    
    MTL::CommandBuffer* cmdBuffer = queue->commandBuffer();
    
    // Step 1: Project all Gaussians
    {
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(projectGaussiansPSO);
        enc->setBuffer(gaussianBuffer, 0, 0);
        enc->setBuffer(projectedGaussians, 0, 1);
        enc->setBuffer(uniformBuffer, 0, 2);
        
        NS::UInteger threadGroupSize = projectGaussiansPSO->maxTotalThreadsPerThreadgroup();
        if (threadGroupSize > 256) threadGroupSize = 256;
        
        MTL::Size grid = MTL::Size(gaussianCount, 1, 1);
        MTL::Size threadgroup = MTL::Size(threadGroupSize, 1, 1);
        enc->dispatchThreads(grid, threadgroup);
        enc->endEncoding();
    }
    
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();
    
    auto t2 = std::chrono::high_resolution_clock::now();
    
    // Step 2: CPU-side tile binning and sorting
    ProjectedGaussian* projPtr = (ProjectedGaussian*)projectedGaussians->contents();
    
    // DEBUG: Print some projected Gaussian info (once)
    static bool debugPrinted = false;
    if (!debugPrinted) {
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
                    
                    // Sanity check: radius should be related to depth (closer = bigger)
                    float roughScreenScale = worldScaleMax / p.depth;  // Very rough approximation
                    std::cout << "  Rough screen scale (worldMax/depth): " << roughScreenScale << std::endl;
                }
            }
        }
        
        if (validCount > 0) {
            std::cout << "\nSummary (first 100 Gaussians):" << std::endl;
            std::cout << "  Valid: " << validCount << std::endl;
            std::cout << "  Radius range: " << minRadius << " to " << maxRadius << " pixels (avg=" << (avgRadius/validCount) << ")" << std::endl;
            std::cout << "  Depth range: " << minDepth << " to " << maxDepth << std::endl;
        }
        debugPrinted = true;
        std::cout << "=== End Radius Debug ===\n" << std::endl;
    }
    
    auto t3 = std::chrono::high_resolution_clock::now();
    
    // PARALLEL pair generation using GCD
    // Each thread builds its own vector, then merge
    const float MIN_OPACITY = 0.005f;
    const uint32_t numTilesXLocal = numTilesX;  // Capture for block
    
    dispatch_queue_t dispatchQueue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
    
    // Per-thread pair vectors
    static std::vector<std::pair<uint64_t, uint32_t>> threadPairs[NUM_THREADS];
    
    size_t chunkSize = (gaussianCount + NUM_THREADS - 1) / NUM_THREADS;
    
    dispatch_apply((size_t)NUM_THREADS, dispatchQueue, ^(size_t t) {
        threadPairs[t].clear();
        threadPairs[t].reserve(chunkSize * AVG_TILES_PER_GAUSSIAN);
        
        size_t start = t * chunkSize;
        size_t end = std::min(start + chunkSize, gaussianCount);
        
        for (size_t gIdx = start; gIdx < end; gIdx++) {
            const ProjectedGaussian& p = projPtr[gIdx];
            if (p.radius <= 0 || p.tileMinX > p.tileMaxX || p.tileMinY > p.tileMaxY) continue;
            if (p.opacity < MIN_OPACITY) continue;
            if (p.tileMinX > 10000 || p.tileMaxX > 10000 || p.tileMinY > 10000 || p.tileMaxY > 10000) continue;
            
            uint32_t tilesX = p.tileMaxX - p.tileMinX + 1;
            uint32_t tilesY = p.tileMaxY - p.tileMinY + 1;
            if (tilesX * tilesY > 256) continue;
            
            uint32_t depthKey = *reinterpret_cast<const uint32_t*>(&p.depth);
            depthKey = (depthKey & 0x80000000) ? ~depthKey : (depthKey | 0x80000000);
            
            for (uint32_t ty = p.tileMinY; ty <= p.tileMaxY; ty++) {
                for (uint32_t tx = p.tileMinX; tx <= p.tileMaxX; tx++) {
                    uint32_t tileIdx = ty * numTilesXLocal + tx;
                    uint64_t key = (uint64_t(tileIdx) << 32) | depthKey;
                    threadPairs[t].emplace_back(key, (uint32_t)gIdx);
                }
            }
        }
    });
    
    // Merge thread results
    static std::vector<std::pair<uint64_t, uint32_t>> pairs;
    pairs.clear();
    size_t totalPairsEst = 0;
    for (int t = 0; t < NUM_THREADS; t++) {
        totalPairsEst += threadPairs[t].size();
    }
    pairs.reserve(totalPairsEst);
    
    for (int t = 0; t < NUM_THREADS; t++) {
        pairs.insert(pairs.end(), threadPairs[t].begin(), threadPairs[t].end());
    }
    
    uint32_t totalPairs = (uint32_t)pairs.size();
    
    auto t4 = std::chrono::high_resolution_clock::now();
    
    if (totalPairs == 0) {
        // Clear tile ranges
        memset(tileRanges->contents(), 0, maxTiles * sizeof(TileRange));
        return;
    }
    
    // Clamp totalPairs to prevent memory explosion
    if (totalPairs > 50000000) {
        std::cerr << "WARNING: totalPairs clamped from " << totalPairs << " to 50M" << std::endl;
        pairs.resize(50000000);
        totalPairs = 50000000;
    }
    
    // Ensure buffer capacity
    ensurePairsCapacity(totalPairs);
    
    // Sort by key (tile + depth) - use fast radix sort
    auto sortStart = std::chrono::high_resolution_clock::now();
    parallelRadixSort(pairs);
    auto t5 = std::chrono::high_resolution_clock::now();
    
    // Copy sorted data to GPU buffers
    uint64_t* keyPtr = (uint64_t*)gaussianKeys->contents();
    uint32_t* sortedIndices = (uint32_t*)gaussianValues->contents();
    uint32_t pairCount = (uint32_t)pairs.size();
    
    for (uint32_t i = 0; i < pairCount; i++) {
        keyPtr[i] = pairs[i].first;
        sortedIndices[i] = pairs[i].second;
    }
    
    // Build tile ranges on GPU using parallel binary search
    {
        MTL::CommandBuffer* rangeCmdBuffer = queue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = rangeCmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(buildTileRangesPSO);
        enc->setBuffer(gaussianKeys, 0, 0);
        enc->setBuffer(tileRanges, 0, 1);
        enc->setBytes(&pairCount, sizeof(uint32_t), 2);
        enc->setBytes(&maxTiles, sizeof(uint32_t), 3);
        
        NS::UInteger threadGroupSize = buildTileRangesPSO->maxTotalThreadsPerThreadgroup();
        if (threadGroupSize > 256) threadGroupSize = 256;
        
        MTL::Size grid = MTL::Size(maxTiles, 1, 1);
        MTL::Size threadgroup = MTL::Size(threadGroupSize, 1, 1);
        enc->dispatchThreads(grid, threadgroup);
        enc->endEncoding();
        
        rangeCmdBuffer->commit();
        rangeCmdBuffer->waitUntilCompleted();
    }
    
    TileRange* ranges = (TileRange*)tileRanges->contents();
    
    auto t6 = std::chrono::high_resolution_clock::now();
    
    // DEBUG: Check tile ranges for gaps/overlaps
    static bool tileRangeDebugPrinted = false;
    if (!tileRangeDebugPrinted) {
        std::cout << "\n=== Tile Range Debug ===" << std::endl;
        std::cout << "Total pairs: " << pairCount << ", Total tiles: " << maxTiles << std::endl;
        
        uint32_t totalCoverage = 0;
        uint32_t tilesWithData = 0;
        uint32_t maxCount = 0;
        uint32_t lastEnd = 0;
        bool foundGap = false;
        bool foundOverlap = false;
        
        for (uint32_t tile = 0; tile < maxTiles && tile < 10000; tile++) {
            if (ranges[tile].count > 0) {
                tilesWithData++;
                totalCoverage += ranges[tile].count;
                maxCount = std::max(maxCount, ranges[tile].count);
                
                // Check for gaps (non-contiguous ranges)
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
            uint32_t tileId = (uint32_t)(pairs[i].first >> 32);
            uint32_t depthBits = (uint32_t)(pairs[i].first & 0xFFFFFFFF);
            std::cout << "  [" << i << "] tile=" << tileId << " gaussian=" << pairs[i].second 
                      << " depthKey=0x" << std::hex << depthBits << std::dec << std::endl;
        }
        
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
    
    // Step 3: GPU forward rendering
    cmdBuffer = queue->commandBuffer();
    
    {
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(tiledForwardPSO);
        enc->setBuffer(gaussianBuffer, 0, 0);
        enc->setBuffer(projectedGaussians, 0, 1);
        enc->setBuffer(gaussianValues, 0, 2);  // sortedIndices
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
    
    auto t7 = std::chrono::high_resolution_clock::now();
    
    // Print timing stats periodically
    static int frameCount = 0;
    static double totalProjectTime = 0, totalPairGenTime = 0, totalSortTime = 0;
    static double totalRangeTime = 0, totalRenderTime = 0, totalForwardTime = 0;
    
    double projectMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double pairGenMs = std::chrono::duration<double, std::milli>(t4 - t3).count();
    double sortMs = std::chrono::duration<double, std::milli>(t5 - t4).count();
    double rangeMs = std::chrono::duration<double, std::milli>(t6 - t5).count();
    double renderMs = std::chrono::duration<double, std::milli>(t7 - t6).count();
    double forwardMs = std::chrono::duration<double, std::milli>(t7 - forwardStart).count();
    
    totalProjectTime += projectMs;
    totalPairGenTime += pairGenMs;
    totalSortTime += sortMs;
    totalRangeTime += rangeMs;
    totalRenderTime += renderMs;
    totalForwardTime += forwardMs;
    frameCount++;
    
    if (frameCount == 100) {
        printf("\n=== Forward Pass Timing (avg over %d frames) ===\n", frameCount);
        printf("  Project (GPU):  %6.2f ms\n", totalProjectTime / frameCount);
        printf("  Pair Gen:       %6.2f ms\n", totalPairGenTime / frameCount);
        printf("  Sort:           %6.2f ms\n", totalSortTime / frameCount);
        printf("  Range Build:    %6.2f ms\n", totalRangeTime / frameCount);
        printf("  Render (GPU):   %6.2f ms\n", totalRenderTime / frameCount);
        printf("  TOTAL:          %6.2f ms\n", totalForwardTime / frameCount);
        printf("  Pairs/frame:    ~%u\n", totalPairs);
        
        frameCount = 0;
        totalProjectTime = totalPairGenTime = totalSortTime = 0;
        totalRangeTime = totalRenderTime = totalForwardTime = 0;
    }
}

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
        enc->setBuffer(gaussianValues, 0, 3);  // sortedIndices
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

//
//  tiled_rasterizer.mm
//  GuassianSplatting
//

#include "tiled_rasterizer.hpp"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <vector>

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
    
    uint32_t width = (uint32_t)outputTexture->width();
    uint32_t height = (uint32_t)outputTexture->height();
    
    ensureBufferCapacity(width, height, gaussianCount);
    
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
    
    // Step 2: CPU-side tile binning and sorting
    ProjectedGaussian* projPtr = (ProjectedGaussian*)projectedGaussians->contents();
    
    // First pass: count pairs
    uint32_t totalPairs = 0;
    for (uint32_t gIdx = 0; gIdx < gaussianCount; gIdx++) {
        const ProjectedGaussian& p = projPtr[gIdx];
        if (p.radius <= 0 || p.tileMinX > p.tileMaxX || p.tileMinY > p.tileMaxY) continue;
        
        uint32_t tilesX = p.tileMaxX - p.tileMinX + 1;
        uint32_t tilesY = p.tileMaxY - p.tileMinY + 1;
        totalPairs += tilesX * tilesY;
    }
    
    if (totalPairs == 0) {
        // Clear tile ranges
        memset(tileRanges->contents(), 0, maxTiles * sizeof(TileRange));
        return;
    }
    
    // Ensure buffer capacity
    ensurePairsCapacity(totalPairs);
    
    // Second pass: build pairs
    std::vector<std::pair<uint64_t, uint32_t>> pairs;
    pairs.reserve(totalPairs);
    
    for (uint32_t gIdx = 0; gIdx < gaussianCount; gIdx++) {
        const ProjectedGaussian& p = projPtr[gIdx];
        if (p.radius <= 0 || p.tileMinX > p.tileMaxX || p.tileMinY > p.tileMaxY) continue;
        
        // Convert depth to sortable key
        uint32_t depthKey = *reinterpret_cast<const uint32_t*>(&p.depth);
        depthKey = (depthKey & 0x80000000) ? ~depthKey : (depthKey | 0x80000000);
        
        for (uint32_t ty = p.tileMinY; ty <= p.tileMaxY; ty++) {
            for (uint32_t tx = p.tileMinX; tx <= p.tileMaxX; tx++) {
                uint32_t tileIdx = ty * numTilesX + tx;
                uint64_t key = (uint64_t(tileIdx) << 32) | depthKey;
                pairs.emplace_back(key, gIdx);
            }
        }
    }
    
    // Sort by key (tile + depth)
    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
    
    // Build tile ranges and sorted indices
    uint32_t* sortedIndices = (uint32_t*)gaussianValues->contents();
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

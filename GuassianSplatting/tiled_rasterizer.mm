//
//  sort.metal
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-28.
//

#include "tiled_rasterizer.hpp"
#include <iostream>
#include <algorithm>

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
    
    // Other buffers allocated on demand
    tileCounts = nullptr;
    tileOffsets = nullptr;
    tileWriteOffsets = nullptr;
    gaussianKeys = nullptr;
    gaussianValues = nullptr;
    tileRanges = nullptr;
    sortedGaussianIndices = nullptr;
    totalPairsBuffer = nullptr;
    perPixelTransmittance = nullptr;
    perPixelLastIdx = nullptr;
}

TiledRasterizer::~TiledRasterizer() {
    if (projectedGaussians) projectedGaussians->release();
    if (tileCounts) tileCounts->release();
    if (tileOffsets) tileOffsets->release();
    if (tileWriteOffsets) tileWriteOffsets->release();
    if (gaussianKeys) gaussianKeys->release();
    if (gaussianValues) gaussianValues->release();
    if (tileRanges) tileRanges->release();
    if (sortedGaussianIndices) sortedGaussianIndices->release();
    if (totalPairsBuffer) totalPairsBuffer->release();
    if (perPixelTransmittance) perPixelTransmittance->release();
    if (perPixelLastIdx) perPixelLastIdx->release();
    if (uniformBuffer) uniformBuffer->release();
    
    if (projectGaussiansPSO) projectGaussiansPSO->release();
    if (countTilesPSO) countTilesPSO->release();
    if (writeGaussianKeysPSO) writeGaussianKeysPSO->release();
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
    countTilesPSO = makePipeline("countTilesPerGaussian");
    writeGaussianKeysPSO = makePipeline("writeGaussianKeys");
    tiledForwardPSO = makePipeline("tiledForward");
    tiledBackwardPSO = makePipeline("tiledBackward");
}

void TiledRasterizer::ensureBufferCapacity(uint32_t width, uint32_t height) {
    uint32_t newNumTilesX = (width + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t newNumTilesY = (height + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t newMaxTiles = newNumTilesX * newNumTilesY;
    uint32_t newMaxPairs = maxGaussians * MAX_PAIRS_PER_GAUSSIAN;
    uint32_t numPixels = width * height;
    
    bool needsRealloc = (newMaxTiles > maxTiles) ||
                        (newMaxPairs > maxPairs) ||
                        (width != currentWidth) ||
                        (height != currentHeight);
    
    if (!needsRealloc) return;
    
    // Release old buffers
    if (tileCounts) tileCounts->release();
    if (tileOffsets) tileOffsets->release();
    if (tileWriteOffsets) tileWriteOffsets->release();
    if (gaussianKeys) gaussianKeys->release();
    if (gaussianValues) gaussianValues->release();
    if (tileRanges) tileRanges->release();
    if (sortedGaussianIndices) sortedGaussianIndices->release();
    if (totalPairsBuffer) totalPairsBuffer->release();
    if (perPixelTransmittance) perPixelTransmittance->release();
    if (perPixelLastIdx) perPixelLastIdx->release();
    
    maxTiles = newMaxTiles;
    maxPairs = newMaxPairs;
    currentWidth = width;
    currentHeight = height;
    numTilesX = newNumTilesX;
    numTilesY = newNumTilesY;
    
    // Allocate new buffers
    tileCounts = device->newBuffer(maxTiles * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    tileOffsets = device->newBuffer(maxTiles * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    tileWriteOffsets = device->newBuffer(maxTiles * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    gaussianKeys = device->newBuffer(maxPairs * sizeof(uint64_t), MTL::ResourceStorageModeShared);
    gaussianValues = device->newBuffer(maxPairs * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    tileRanges = device->newBuffer(maxTiles * sizeof(TileRange), MTL::ResourceStorageModeShared);
    sortedGaussianIndices = device->newBuffer(maxPairs * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    totalPairsBuffer = device->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    perPixelTransmittance = device->newBuffer(numPixels * sizeof(float), MTL::ResourceStorageModeShared);
    perPixelLastIdx = device->newBuffer(numPixels * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    std::cout << "Allocated tiled rasterizer buffers: "
              << numTilesX << "x" << numTilesY << " tiles ("
              << maxTiles << " total), " << maxPairs << " max pairs" << std::endl;
}

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
    memcpy(uniformBuffer->contents(), &u, sizeof(TiledUniforms));
    
    // Clear tile counts and total pairs
    memset(tileCounts->contents(), 0, maxTiles * sizeof(uint32_t));
    *(uint32_t*)totalPairsBuffer->contents() = 0;
    
    MTL::CommandBuffer* cmdBuffer = queue->commandBuffer();
    
    // Step 1: Project all Gaussians to screen space
    {
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(projectGaussiansPSO);
        enc->setBuffer(gaussianBuffer, 0, 0);
        enc->setBuffer(projectedGaussians, 0, 1);
        enc->setBuffer(uniformBuffer, 0, 2);
        
        MTL::Size grid = MTL::Size(gaussianCount, 1, 1);
        MTL::Size threadgroup = MTL::Size(std::min((size_t)256, gaussianCount), 1, 1);
        enc->dispatchThreads(grid, threadgroup);
        enc->endEncoding();
    }
    
    // Step 2: Count tiles per Gaussian
    {
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(countTilesPSO);
        enc->setBuffer(projectedGaussians, 0, 0);
        enc->setBuffer(tileCounts, 0, 1);
        enc->setBuffer(totalPairsBuffer, 0, 2);
        enc->setBuffer(uniformBuffer, 0, 3);
        
        MTL::Size grid = MTL::Size(gaussianCount, 1, 1);
        MTL::Size threadgroup = MTL::Size(std::min((size_t)256, gaussianCount), 1, 1);
        enc->dispatchThreads(grid, threadgroup);
        enc->endEncoding();
    }
    
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();
    
    // Get total pairs
    uint32_t totalPairs = *(uint32_t*)totalPairsBuffer->contents();
    if (totalPairs == 0) {
        std::cout << "Warning: No Gaussian-tile pairs generated" << std::endl;
        return;
    }
    
    // Compute prefix sum (CPU)
    uint32_t* counts = (uint32_t*)tileCounts->contents();
    uint32_t* offsets = (uint32_t*)tileOffsets->contents();
    uint32_t* writeOffsets = (uint32_t*)tileWriteOffsets->contents();
    
    uint32_t runningSum = 0;
    for (uint32_t i = 0; i < maxTiles; i++) {
        offsets[i] = runningSum;
        writeOffsets[i] = runningSum;
        runningSum += counts[i];
    }
    
    cmdBuffer = queue->commandBuffer();
    
    // Step 3: Write Gaussian keys
    {
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(writeGaussianKeysPSO);
        enc->setBuffer(projectedGaussians, 0, 0);
        enc->setBuffer(gaussianKeys, 0, 1);
        enc->setBuffer(gaussianValues, 0, 2);
        enc->setBuffer(tileWriteOffsets, 0, 3);
        enc->setBuffer(uniformBuffer, 0, 4);
        
        MTL::Size grid = MTL::Size(gaussianCount, 1, 1);
        MTL::Size threadgroup = MTL::Size(std::min((size_t)256, gaussianCount), 1, 1);
        enc->dispatchThreads(grid, threadgroup);
        enc->endEncoding();
    }
    
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();
    
    // Step 4: Sort (CPU for simplicity - can be GPU later)
    {
        struct SortPair {
            uint64_t key;
            uint32_t value;
        };
        
        std::vector<SortPair> pairs(totalPairs);
        uint64_t* keyPtr = (uint64_t*)gaussianKeys->contents();
        uint32_t* valPtr = (uint32_t*)gaussianValues->contents();
        
        for (uint32_t i = 0; i < totalPairs; i++) {
            pairs[i].key = keyPtr[i];
            pairs[i].value = valPtr[i];
        }
        
        std::sort(pairs.begin(), pairs.end(), [](const SortPair& a, const SortPair& b) {
            return a.key < b.key;
        });
        
        // Write back sorted indices
        uint32_t* sortedIndices = (uint32_t*)sortedGaussianIndices->contents();
        for (uint32_t i = 0; i < totalPairs; i++) {
            sortedIndices[i] = pairs[i].value;
        }
        
        // Compute tile ranges
        TileRange* ranges = (TileRange*)tileRanges->contents();
        memset(ranges, 0, maxTiles * sizeof(TileRange));
        
        if (totalPairs > 0) {
            uint32_t currentTile = (uint32_t)(pairs[0].key >> 32);
            uint32_t rangeStart = 0;
            
            for (uint32_t i = 1; i <= totalPairs; i++) {
                uint32_t tile = (i < totalPairs) ? (uint32_t)(pairs[i].key >> 32) : UINT32_MAX;
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
    }
    
    cmdBuffer = queue->commandBuffer();
    
    // Step 5: Tiled forward rendering
    {
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(tiledForwardPSO);
        enc->setBuffer(gaussianBuffer, 0, 0);
        enc->setBuffer(projectedGaussians, 0, 1);
        enc->setBuffer(sortedGaussianIndices, 0, 2);
        enc->setBuffer(tileRanges, 0, 3);
        enc->setBuffer(uniformBuffer, 0, 4);
        enc->setBuffer(perPixelTransmittance, 0, 5);
        enc->setBuffer(perPixelLastIdx, 0, 6);
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
    
    // Tiled backward pass
    {
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(tiledBackwardPSO);
        enc->setBuffer(gaussianBuffer, 0, 0);
        enc->setBuffer(gradientBuffer, 0, 1);
        enc->setBuffer(projectedGaussians, 0, 2);
        enc->setBuffer(sortedGaussianIndices, 0, 3);
        enc->setBuffer(tileRanges, 0, 4);
        enc->setBuffer(uniformBuffer, 0, 5);
        enc->setBuffer(perPixelTransmittance, 0, 6);
        enc->setBuffer(perPixelLastIdx, 0, 7);
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

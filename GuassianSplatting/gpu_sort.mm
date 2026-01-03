//
//  gpu_sort.mm
//  GaussianSplatting
//
//  Fixed GPU Radix Sort - Implementation
//

#include "gpu_sort.hpp"
#include <iostream>
#include <algorithm>
#include <cstring>

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

// ============================================================================
// GPURadixSort32 Implementation
// ============================================================================

GPURadixSort32::GPURadixSort32(MTL::Device* device, MTL::Library* library, size_t maxElements)
    : device(device)
    , maxElements(maxElements)
{
    createPipelines(library);
    
    // Allocate double buffers
    keysBuffers[0] = device->newBuffer(maxElements * sizeof(uint32_t),
                                       MTL::ResourceStorageModeShared);
    keysBuffers[1] = device->newBuffer(maxElements * sizeof(uint32_t),
                                       MTL::ResourceStorageModeShared);
    valuesBuffers[0] = device->newBuffer(maxElements * sizeof(uint32_t),
                                         MTL::ResourceStorageModeShared);
    valuesBuffers[1] = device->newBuffer(maxElements * sizeof(uint32_t),
                                         MTL::ResourceStorageModeShared);
    
    // Histogram buffer
    histogramBuffer = device->newBuffer(RADIX_SIZE * sizeof(uint32_t),
                                        MTL::ResourceStorageModeShared);
    digitCountersBuffer = device->newBuffer(RADIX_SIZE * sizeof(uint32_t),
                                            MTL::ResourceStorageModeShared);
    
    // Camera position buffer
    cameraPosBuffer = device->newBuffer(sizeof(simd_float3),
                                        MTL::ResourceStorageModeShared);
}

GPURadixSort32::~GPURadixSort32() {
    if (keysBuffers[0]) keysBuffers[0]->release();
    if (keysBuffers[1]) keysBuffers[1]->release();
    if (valuesBuffers[0]) valuesBuffers[0]->release();
    if (valuesBuffers[1]) valuesBuffers[1]->release();
    if (histogramBuffer) histogramBuffer->release();
    if (digitCountersBuffer) digitCountersBuffer->release();
    if (cameraPosBuffer) cameraPosBuffer->release();
    
    if (computeDepthsPSO) computeDepthsPSO->release();
    if (histogram32PSO) histogram32PSO->release();
    if (prefixSum256PSO) prefixSum256PSO->release();
    if (scatter32SimplePSO) scatter32SimplePSO->release();
    if (scatter32OptimizedPSO) scatter32OptimizedPSO->release();
    if (clearHistogramPSO) clearHistogramPSO->release();
}

void GPURadixSort32::createPipelines(MTL::Library* library) {
    computeDepthsPSO = createPipeline(device, library, "computeDepths");
    histogram32PSO = createPipeline(device, library, "histogram32");
    prefixSum256PSO = createPipeline(device, library, "prefixSum256");
    scatter32SimplePSO = createPipeline(device, library, "scatter32Simple");
    scatter32OptimizedPSO = createPipeline(device, library, "scatterOptimized32");
    clearHistogramPSO = createPipeline(device, library, "clearHistogram");
}

void GPURadixSort32::ensureCapacity(size_t numElements) {
    if (numElements <= maxElements) return;
    
    maxElements = std::max(numElements, maxElements * 2);
    
    // Reallocate buffers
    if (keysBuffers[0]) keysBuffers[0]->release();
    if (keysBuffers[1]) keysBuffers[1]->release();
    if (valuesBuffers[0]) valuesBuffers[0]->release();
    if (valuesBuffers[1]) valuesBuffers[1]->release();
    
    keysBuffers[0] = device->newBuffer(maxElements * sizeof(uint32_t),
                                       MTL::ResourceStorageModeShared);
    keysBuffers[1] = device->newBuffer(maxElements * sizeof(uint32_t),
                                       MTL::ResourceStorageModeShared);
    valuesBuffers[0] = device->newBuffer(maxElements * sizeof(uint32_t),
                                         MTL::ResourceStorageModeShared);
    valuesBuffers[1] = device->newBuffer(maxElements * sizeof(uint32_t),
                                         MTL::ResourceStorageModeShared);
}

MTL::Buffer* GPURadixSort32::sort(MTL::CommandQueue* queue,
                                   MTL::Buffer* positionBuffer,
                                   simd_float3 cameraPos,
                                   size_t numElements) {
    if (numElements == 0) return valuesBuffers[0];
    
    ensureCapacity(numElements);
    
    uint32_t numElementsU32 = (uint32_t)numElements;
    
    // Copy camera position
    memcpy(cameraPosBuffer->contents(), &cameraPos, sizeof(simd_float3));
    
    MTL::CommandBuffer* cmdBuffer = queue->commandBuffer();
    
    // Step 1: Compute depths and initialize keys/values
    {
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(computeDepthsPSO);
        enc->setBuffer(positionBuffer, 0, 0);
        enc->setBuffer(cameraPosBuffer, 0, 1);
        enc->setBuffer(keysBuffers[0], 0, 2);
        enc->setBuffer(valuesBuffers[0], 0, 3);
        enc->setBytes(&numElementsU32, sizeof(uint32_t), 4);
        
        MTL::Size grid = MTL::Size(numElements, 1, 1);
        MTL::Size tg = MTL::Size(THREADGROUP_SIZE, 1, 1);
        enc->dispatchThreads(grid, tg);
        enc->endEncoding();
    }
    
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();
    
    // Step 2: 4 passes of radix sort (8 bits per pass)
    int srcIdx = 0;
    
    for (uint32_t pass = 0; pass < NUM_PASSES; pass++) {
        uint32_t bitOffset = pass * 8;
        int dstIdx = 1 - srcIdx;
        
        cmdBuffer = queue->commandBuffer();
        
        // Clear histogram
        {
            MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
            enc->setComputePipelineState(clearHistogramPSO);
            enc->setBuffer(histogramBuffer, 0, 0);
            enc->dispatchThreads(MTL::Size(RADIX_SIZE, 1, 1), MTL::Size(RADIX_SIZE, 1, 1));
            enc->endEncoding();
        }
        
        // Build histogram
        {
            MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
            enc->setComputePipelineState(histogram32PSO);
            enc->setBuffer(keysBuffers[srcIdx], 0, 0);
            enc->setBuffer(histogramBuffer, 0, 1);
            enc->setBytes(&bitOffset, sizeof(uint32_t), 2);
            enc->setBytes(&numElementsU32, sizeof(uint32_t), 3);
            
            MTL::Size grid = MTL::Size(numElements, 1, 1);
            MTL::Size tg = MTL::Size(THREADGROUP_SIZE, 1, 1);
            enc->dispatchThreads(grid, tg);
            enc->endEncoding();
        }
        
        cmdBuffer->commit();
        cmdBuffer->waitUntilCompleted();
        
        // CPU prefix sum (simple and correct)
        {
            uint32_t* hist = (uint32_t*)histogramBuffer->contents();
            uint32_t sum = 0;
            for (int i = 0; i < RADIX_SIZE; i++) {
                uint32_t count = hist[i];
                hist[i] = sum;
                sum += count;
            }
        }
        
        // Scatter using simple O(n²) method (correct but slow)
        // TODO: Replace with optimized scatter once verified correct
        cmdBuffer = queue->commandBuffer();
        {
            MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
            enc->setComputePipelineState(scatter32SimplePSO);
            enc->setBuffer(keysBuffers[srcIdx], 0, 0);
            enc->setBuffer(valuesBuffers[srcIdx], 0, 1);
            enc->setBuffer(keysBuffers[dstIdx], 0, 2);
            enc->setBuffer(valuesBuffers[dstIdx], 0, 3);
            enc->setBuffer(histogramBuffer, 0, 4);
            enc->setBytes(&bitOffset, sizeof(uint32_t), 5);
            enc->setBytes(&numElementsU32, sizeof(uint32_t), 6);
            
            MTL::Size grid = MTL::Size(numElements, 1, 1);
            MTL::Size tg = MTL::Size(THREADGROUP_SIZE, 1, 1);
            enc->dispatchThreads(grid, tg);
            enc->endEncoding();
        }
        
        cmdBuffer->commit();
        cmdBuffer->waitUntilCompleted();
        
        srcIdx = dstIdx;
    }
    
    currentBuffer = srcIdx;
    return valuesBuffers[srcIdx];
}

// ============================================================================
// GPURadixSort64 Implementation
// ============================================================================

GPURadixSort64::GPURadixSort64(MTL::Device* device, MTL::Library* library, size_t maxElements)
    : device(device)
    , maxElements(maxElements)
{
    createPipelines(library);
    
    // Allocate double buffers for 64-bit keys
    keysBuffers[0] = device->newBuffer(maxElements * sizeof(uint64_t),
                                       MTL::ResourceStorageModeShared);
    keysBuffers[1] = device->newBuffer(maxElements * sizeof(uint64_t),
                                       MTL::ResourceStorageModeShared);
    valuesBuffers[0] = device->newBuffer(maxElements * sizeof(uint32_t),
                                         MTL::ResourceStorageModeShared);
    valuesBuffers[1] = device->newBuffer(maxElements * sizeof(uint32_t),
                                         MTL::ResourceStorageModeShared);
    
    histogramBuffer = device->newBuffer(RADIX_SIZE * sizeof(uint32_t),
                                        MTL::ResourceStorageModeShared);
    digitCountersBuffer = device->newBuffer(RADIX_SIZE * sizeof(uint32_t),
                                            MTL::ResourceStorageModeShared);
}

GPURadixSort64::~GPURadixSort64() {
    if (keysBuffers[0]) keysBuffers[0]->release();
    if (keysBuffers[1]) keysBuffers[1]->release();
    if (valuesBuffers[0]) valuesBuffers[0]->release();
    if (valuesBuffers[1]) valuesBuffers[1]->release();
    if (histogramBuffer) histogramBuffer->release();
    if (digitCountersBuffer) digitCountersBuffer->release();
    
    if (histogram64PSO) histogram64PSO->release();
    if (prefixSum256PSO) prefixSum256PSO->release();
    if (scatter64SimplePSO) scatter64SimplePSO->release();
    if (scatter64OptimizedPSO) scatter64OptimizedPSO->release();
    if (clearHistogramPSO) clearHistogramPSO->release();
}

void GPURadixSort64::createPipelines(MTL::Library* library) {
    histogram64PSO = createPipeline(device, library, "histogram64");
    prefixSum256PSO = createPipeline(device, library, "prefixSum256");
    scatter64SimplePSO = createPipeline(device, library, "scatter64Simple");
    scatter64OptimizedPSO = createPipeline(device, library, "scatter64Optimized");
    clearHistogramPSO = createPipeline(device, library, "clearHistogram");
}

void GPURadixSort64::ensureCapacity(size_t numElements) {
    if (numElements <= maxElements) return;
    
    maxElements = std::max(numElements, maxElements * 2);
    
    if (keysBuffers[0]) keysBuffers[0]->release();
    if (keysBuffers[1]) keysBuffers[1]->release();
    if (valuesBuffers[0]) valuesBuffers[0]->release();
    if (valuesBuffers[1]) valuesBuffers[1]->release();
    
    keysBuffers[0] = device->newBuffer(maxElements * sizeof(uint64_t),
                                       MTL::ResourceStorageModeShared);
    keysBuffers[1] = device->newBuffer(maxElements * sizeof(uint64_t),
                                       MTL::ResourceStorageModeShared);
    valuesBuffers[0] = device->newBuffer(maxElements * sizeof(uint32_t),
                                         MTL::ResourceStorageModeShared);
    valuesBuffers[1] = device->newBuffer(maxElements * sizeof(uint32_t),
                                         MTL::ResourceStorageModeShared);
}

void GPURadixSort64::sort(MTL::CommandQueue* queue,
                           MTL::Buffer* keysIn,
                           MTL::Buffer* valuesIn,
                           size_t numElements) {
    if (numElements == 0) return;
    
    ensureCapacity(numElements);
    
    uint32_t numElementsU32 = (uint32_t)numElements;
    
    // Copy input to our buffers
    memcpy(keysBuffers[0]->contents(), keysIn->contents(), numElements * sizeof(uint64_t));
    memcpy(valuesBuffers[0]->contents(), valuesIn->contents(), numElements * sizeof(uint32_t));
    
    int srcIdx = 0;
    
    // 8 passes for 64-bit keys
    for (uint32_t pass = 0; pass < NUM_PASSES; pass++) {
        uint32_t bitOffset = pass * 8;
        int dstIdx = 1 - srcIdx;
        
        MTL::CommandBuffer* cmdBuffer = queue->commandBuffer();
        
        // Clear histogram
        {
            MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
            enc->setComputePipelineState(clearHistogramPSO);
            enc->setBuffer(histogramBuffer, 0, 0);
            enc->dispatchThreads(MTL::Size(RADIX_SIZE, 1, 1), MTL::Size(RADIX_SIZE, 1, 1));
            enc->endEncoding();
        }
        
        // Build histogram
        {
            MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
            enc->setComputePipelineState(histogram64PSO);
            enc->setBuffer(keysBuffers[srcIdx], 0, 0);
            enc->setBuffer(histogramBuffer, 0, 1);
            enc->setBytes(&bitOffset, sizeof(uint32_t), 2);
            enc->setBytes(&numElementsU32, sizeof(uint32_t), 3);
            
            MTL::Size grid = MTL::Size(numElements, 1, 1);
            MTL::Size tg = MTL::Size(THREADGROUP_SIZE, 1, 1);
            enc->dispatchThreads(grid, tg);
            enc->endEncoding();
        }
        
        cmdBuffer->commit();
        cmdBuffer->waitUntilCompleted();
        
        // CPU prefix sum
        {
            uint32_t* hist = (uint32_t*)histogramBuffer->contents();
            uint32_t sum = 0;
            for (int i = 0; i < RADIX_SIZE; i++) {
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
            enc->setBuffer(keysBuffers[srcIdx], 0, 0);
            enc->setBuffer(valuesBuffers[srcIdx], 0, 1);
            enc->setBuffer(keysBuffers[dstIdx], 0, 2);
            enc->setBuffer(valuesBuffers[dstIdx], 0, 3);
            enc->setBuffer(histogramBuffer, 0, 4);
            enc->setBytes(&bitOffset, sizeof(uint32_t), 5);
            enc->setBytes(&numElementsU32, sizeof(uint32_t), 6);
            
            MTL::Size grid = MTL::Size(numElements, 1, 1);
            MTL::Size tg = MTL::Size(THREADGROUP_SIZE, 1, 1);
            enc->dispatchThreads(grid, tg);
            enc->endEncoding();
        }
        
        cmdBuffer->commit();
        cmdBuffer->waitUntilCompleted();
        
        srcIdx = dstIdx;
    }
    
    currentBuffer = srcIdx;
}

// ============================================================================
// GPUTileSorter Implementation
// ============================================================================

GPUTileSorter::GPUTileSorter(MTL::Device* device, MTL::Library* library,
                             size_t maxGaussians, size_t maxPairs)
    : device(device)
    , maxGaussians(maxGaussians)
    , maxPairs(maxPairs)
{
    createPipelines(library);
    
    // Create sub-sorter
    radixSort = new GPURadixSort64(device, library, maxPairs);
    
    // Allocate initial buffers
    tileCountsBuffer = nullptr;
    tileOffsetsBuffer = nullptr;
    tileWriteIdxBuffer = nullptr;
    keysBuffer = nullptr;
    valuesBuffer = nullptr;
    tileRangesBuffer = nullptr;
    
    totalPairsBuffer = device->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
}

GPUTileSorter::~GPUTileSorter() {
    delete radixSort;
    
    if (tileCountsBuffer) tileCountsBuffer->release();
    if (tileOffsetsBuffer) tileOffsetsBuffer->release();
    if (tileWriteIdxBuffer) tileWriteIdxBuffer->release();
    if (keysBuffer) keysBuffer->release();
    if (valuesBuffer) valuesBuffer->release();
    if (tileRangesBuffer) tileRangesBuffer->release();
    if (totalPairsBuffer) totalPairsBuffer->release();
    
    if (countTilePairsPSO) countTilePairsPSO->release();
    if (prefixSumPSO) prefixSumPSO->release();
    if (generateTileKeysPSO) generateTileKeysPSO->release();
    if (buildTileRangesPSO) buildTileRangesPSO->release();
    if (clearHistogramPSO) clearHistogramPSO->release();
}

void GPUTileSorter::createPipelines(MTL::Library* library) {
    countTilePairsPSO = createPipeline(device, library, "countTilePairs");
    generateTileKeysPSO = createPipeline(device, library, "generateTileKeys");
    buildTileRangesPSO = createPipeline(device, library, "buildTileRanges");
    clearHistogramPSO = createPipeline(device, library, "clearHistogram");
}

void GPUTileSorter::ensureCapacity(uint32_t numTiles, uint32_t numPairs) {
    // Tile buffers
    if (!tileCountsBuffer || tileCountsBuffer->length() < numTiles * sizeof(uint32_t)) {
        if (tileCountsBuffer) tileCountsBuffer->release();
        if (tileOffsetsBuffer) tileOffsetsBuffer->release();
        if (tileWriteIdxBuffer) tileWriteIdxBuffer->release();
        if (tileRangesBuffer) tileRangesBuffer->release();
        
        tileCountsBuffer = device->newBuffer(numTiles * sizeof(uint32_t),
                                             MTL::ResourceStorageModeShared);
        tileOffsetsBuffer = device->newBuffer(numTiles * sizeof(uint32_t),
                                              MTL::ResourceStorageModeShared);
        tileWriteIdxBuffer = device->newBuffer(numTiles * sizeof(uint32_t),
                                               MTL::ResourceStorageModeShared);
        tileRangesBuffer = device->newBuffer(numTiles * sizeof(simd_uint2),
                                             MTL::ResourceStorageModeShared);
    }
    
    // Pair buffers
    if (!keysBuffer || keysBuffer->length() < numPairs * sizeof(uint64_t)) {
        if (keysBuffer) keysBuffer->release();
        if (valuesBuffer) valuesBuffer->release();
        
        keysBuffer = device->newBuffer(numPairs * sizeof(uint64_t),
                                       MTL::ResourceStorageModeShared);
        valuesBuffer = device->newBuffer(numPairs * sizeof(uint32_t),
                                         MTL::ResourceStorageModeShared);
    }
}

void GPUTileSorter::prefixSumCPU(MTL::Buffer* input, MTL::Buffer* output, uint32_t count) {
    uint32_t* in = (uint32_t*)input->contents();
    uint32_t* out = (uint32_t*)output->contents();
    
    uint32_t sum = 0;
    for (uint32_t i = 0; i < count; i++) {
        out[i] = sum;
        sum += in[i];
    }
}

uint32_t GPUTileSorter::sortGaussiansToTiles(
    MTL::CommandQueue* queue,
    MTL::Buffer* projectedGaussians,
    size_t numGaussians,
    uint32_t screenWidth,
    uint32_t screenHeight,
    uint32_t tileSize)
{
    numTilesX = (screenWidth + tileSize - 1) / tileSize;
    numTilesY = (screenHeight + tileSize - 1) / tileSize;
    uint32_t numTiles = numTilesX * numTilesY;
    
    // Estimate max pairs (generous overestimate)
    uint32_t estimatedPairs = (uint32_t)(numGaussians * 8);  // Assume avg 8 tiles per Gaussian
    ensureCapacity(numTiles, estimatedPairs);
    
    // Clear tile counts and total pairs
    memset(tileCountsBuffer->contents(), 0, numTiles * sizeof(uint32_t));
    memset(tileWriteIdxBuffer->contents(), 0, numTiles * sizeof(uint32_t));
    *(uint32_t*)totalPairsBuffer->contents() = 0;
    
    uint32_t numGaussiansU32 = (uint32_t)numGaussians;
    uint32_t maxTilesPerGaussian = MAX_TILES_PER_GAUSSIAN;
    
    MTL::CommandBuffer* cmdBuffer = queue->commandBuffer();
    
    // Step 1: Count tile pairs per Gaussian
    {
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(countTilePairsPSO);
        enc->setBuffer(projectedGaussians, 0, 0);
        enc->setBuffer(tileCountsBuffer, 0, 1);
        enc->setBuffer(totalPairsBuffer, 0, 2);
        enc->setBytes(&numGaussiansU32, sizeof(uint32_t), 3);
        enc->setBytes(&numTilesX, sizeof(uint32_t), 4);
        enc->setBytes(&maxTilesPerGaussian, sizeof(uint32_t), 5);
        
        MTL::Size grid = MTL::Size(numGaussians, 1, 1);
        MTL::Size tg = MTL::Size(THREADGROUP_SIZE, 1, 1);
        enc->dispatchThreads(grid, tg);
        enc->endEncoding();
    }
    
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();
    
    // Get total pairs
    uint32_t totalPairs = *(uint32_t*)totalPairsBuffer->contents();
    lastPairCount = totalPairs;
    
    if (totalPairs == 0) {
        // Initialize empty tile ranges
        simd_uint2* ranges = (simd_uint2*)tileRangesBuffer->contents();
        for (uint32_t i = 0; i < numTiles; i++) {
            ranges[i] = simd_make_uint2(0, 0);
        }
        return 0;
    }
    
    // Ensure we have enough space
    if (totalPairs > maxPairs) {
        maxPairs = totalPairs * 2;
        if (keysBuffer) keysBuffer->release();
        if (valuesBuffer) valuesBuffer->release();
        keysBuffer = device->newBuffer(maxPairs * sizeof(uint64_t),
                                       MTL::ResourceStorageModeShared);
        valuesBuffer = device->newBuffer(maxPairs * sizeof(uint32_t),
                                         MTL::ResourceStorageModeShared);
    }
    
    // Step 2: Prefix sum on tile counts to get offsets
    prefixSumCPU(tileCountsBuffer, tileOffsetsBuffer, numTiles);
    
    // Step 3: Generate tile-depth keys
    cmdBuffer = queue->commandBuffer();
    {
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(generateTileKeysPSO);
        enc->setBuffer(projectedGaussians, 0, 0);
        enc->setBuffer(tileOffsetsBuffer, 0, 1);
        enc->setBuffer(tileWriteIdxBuffer, 0, 2);
        enc->setBuffer(keysBuffer, 0, 3);
        enc->setBuffer(valuesBuffer, 0, 4);
        enc->setBytes(&numGaussiansU32, sizeof(uint32_t), 5);
        enc->setBytes(&numTilesX, sizeof(uint32_t), 6);
        enc->setBytes(&maxTilesPerGaussian, sizeof(uint32_t), 7);
        
        MTL::Size grid = MTL::Size(numGaussians, 1, 1);
        MTL::Size tg = MTL::Size(THREADGROUP_SIZE, 1, 1);
        enc->dispatchThreads(grid, tg);
        enc->endEncoding();
    }
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();
    
    // Step 4: Sort by 64-bit keys (tile + depth)
    radixSort->sort(queue, keysBuffer, valuesBuffer, totalPairs);
    
    // Copy sorted values back (radix sort uses internal buffers)
    // The sorted values are in radixSort->getSortedValues()
    
    // Step 5: Build tile ranges from sorted keys
    cmdBuffer = queue->commandBuffer();
    {
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(buildTileRangesPSO);
        enc->setBuffer(radixSort->getSortedKeys(), 0, 0);
        enc->setBuffer(tileRangesBuffer, 0, 1);
        enc->setBytes(&totalPairs, sizeof(uint32_t), 2);
        enc->setBytes(&numTiles, sizeof(uint32_t), 3);
        
        MTL::Size grid = MTL::Size(numTiles, 1, 1);
        MTL::Size tg = MTL::Size(THREADGROUP_SIZE, 1, 1);
        enc->dispatchThreads(grid, tg);
        enc->endEncoding();
    }
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();
    
    return totalPairs;
}

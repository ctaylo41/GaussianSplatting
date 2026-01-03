//
//  gpu_sort.hpp
//  GaussianSplatting
//
//  Fixed GPU Radix Sort - Header
//

#pragma once

#include <Metal/Metal.hpp>
#include <simd/simd.h>

// ============================================================================
// 32-bit GPU Radix Sort (for viewer depth sorting)
// ============================================================================

class GPURadixSort32 {
public:
    GPURadixSort32(MTL::Device* device, MTL::Library* library, size_t maxElements);
    ~GPURadixSort32();
    
    // Sort Gaussians by depth, returns buffer of sorted indices
    // Uses 4 passes of 8-bit radix sort
    MTL::Buffer* sort(MTL::CommandQueue* queue,
                      MTL::Buffer* positionBuffer,  // float3 positions
                      simd_float3 cameraPos,
                      size_t numElements);
    
    MTL::Buffer* getSortedIndices() { return valuesBuffers[currentBuffer]; }
    
private:
    static constexpr size_t RADIX_SIZE = 256;
    static constexpr size_t THREADGROUP_SIZE = 256;
    static constexpr size_t NUM_PASSES = 4;  // 32 bits / 8 bits per pass
    
    MTL::Device* device;
    size_t maxElements;
    int currentBuffer = 0;
    
    // Compute pipelines
    MTL::ComputePipelineState* computeDepthsPSO;
    MTL::ComputePipelineState* histogram32PSO;
    MTL::ComputePipelineState* prefixSum256PSO;
    MTL::ComputePipelineState* scatter32SimplePSO;
    MTL::ComputePipelineState* scatter32OptimizedPSO;
    MTL::ComputePipelineState* clearHistogramPSO;
    
    // Double-buffered key/value arrays
    MTL::Buffer* keysBuffers[2];
    MTL::Buffer* valuesBuffers[2];
    
    // Histogram and prefix sums
    MTL::Buffer* histogramBuffer;
    MTL::Buffer* digitCountersBuffer;
    
    // Uniform buffers
    MTL::Buffer* cameraPosBuffer;
    
    void createPipelines(MTL::Library* library);
    void ensureCapacity(size_t numElements);
};

// ============================================================================
// 64-bit GPU Radix Sort (for tile + depth compound keys)
// ============================================================================

class GPURadixSort64 {
public:
    GPURadixSort64(MTL::Device* device, MTL::Library* library, size_t maxElements);
    ~GPURadixSort64();
    
    // Sort 64-bit keys with 32-bit values
    // Uses 8 passes of 8-bit radix sort
    void sort(MTL::CommandQueue* queue,
              MTL::Buffer* keysIn,
              MTL::Buffer* valuesIn,
              size_t numElements);
    
    MTL::Buffer* getSortedKeys() { return keysBuffers[currentBuffer]; }
    MTL::Buffer* getSortedValues() { return valuesBuffers[currentBuffer]; }
    
private:
    static constexpr size_t RADIX_SIZE = 256;
    static constexpr size_t THREADGROUP_SIZE = 256;
    static constexpr size_t NUM_PASSES = 8;  // 64 bits / 8 bits per pass
    
    MTL::Device* device;
    size_t maxElements;
    int currentBuffer = 0;
    
    // Compute pipelines
    MTL::ComputePipelineState* histogram64PSO;
    MTL::ComputePipelineState* prefixSum256PSO;
    MTL::ComputePipelineState* scatter64SimplePSO;
    MTL::ComputePipelineState* scatter64OptimizedPSO;
    MTL::ComputePipelineState* clearHistogramPSO;
    
    // Double-buffered arrays
    MTL::Buffer* keysBuffers[2];
    MTL::Buffer* valuesBuffers[2];
    
    // Histogram
    MTL::Buffer* histogramBuffer;
    MTL::Buffer* digitCountersBuffer;
    
    void createPipelines(MTL::Library* library);
    void ensureCapacity(size_t numElements);
};

// ============================================================================
// GPU Tile Sorter (combines projection counting, key generation, and sorting)
// ============================================================================

class GPUTileSorter {
public:
    GPUTileSorter(MTL::Device* device, MTL::Library* library,
                  size_t maxGaussians, size_t maxPairs);
    ~GPUTileSorter();
    
    // Main entry point: sort Gaussians into tiles
    // Returns number of tile-Gaussian pairs generated
    uint32_t sortGaussiansToTiles(
        MTL::CommandQueue* queue,
        MTL::Buffer* projectedGaussians,  // ProjectedGaussian array from projection pass
        size_t numGaussians,
        uint32_t screenWidth,
        uint32_t screenHeight,
        uint32_t tileSize = 16);
    
    // Access results
    MTL::Buffer* getSortedValues() { return radixSort->getSortedValues(); }
    MTL::Buffer* getTileRanges() { return tileRangesBuffer; }
    
    // Get tile counts for debugging
    uint32_t getNumTilesX() const { return numTilesX; }
    uint32_t getNumTilesY() const { return numTilesY; }
    uint32_t getNumTiles() const { return numTilesX * numTilesY; }
    uint32_t getLastPairCount() const { return lastPairCount; }
    
private:
    static constexpr uint32_t MAX_TILES_PER_GAUSSIAN = 256;
    static constexpr uint32_t THREADGROUP_SIZE = 256;
    
    MTL::Device* device;
    size_t maxGaussians;
    size_t maxPairs;
    uint32_t numTilesX = 0;
    uint32_t numTilesY = 0;
    uint32_t lastPairCount = 0;
    
    // Sub-sorter
    GPURadixSort64* radixSort;
    
    // Compute pipelines
    MTL::ComputePipelineState* countTilePairsPSO;
    MTL::ComputePipelineState* prefixSumPSO;
    MTL::ComputePipelineState* generateTileKeysPSO;
    MTL::ComputePipelineState* buildTileRangesPSO;
    MTL::ComputePipelineState* clearHistogramPSO;
    
    // Buffers
    MTL::Buffer* tileCountsBuffer;
    MTL::Buffer* tileOffsetsBuffer;
    MTL::Buffer* tileWriteIdxBuffer;
    MTL::Buffer* totalPairsBuffer;
    MTL::Buffer* keysBuffer;
    MTL::Buffer* valuesBuffer;
    MTL::Buffer* tileRangesBuffer;
    
    void createPipelines(MTL::Library* library);
    void ensureCapacity(uint32_t numTiles, uint32_t numPairs);
    void prefixSumCPU(MTL::Buffer* input, MTL::Buffer* output, uint32_t count);
};

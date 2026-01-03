//
//  tiled_rasterizer.hpp
//  GaussianSplatting
//
//  Updated with GPU sorting for fast forward pass
//

#pragma once

#include <Metal/Metal.hpp>
#include <simd/simd.h>

// Forward declaration
class GPUTileSorter;
class GPURadixSort64;

// Must match shader definition EXACTLY
struct TileRange {
    uint32_t start;
    uint32_t count;
};

// Projected Gaussian data - must match shader struct
struct ProjectedGaussian {
    simd_float2 screenPos;
    simd_float3 conic;       // Inverse 2D covariance (a, b, c)
    float depth;
    float opacity;           // After sigmoid
    simd_float3 color;
    float radius;
    uint32_t tileMinX;
    uint32_t tileMinY;
    uint32_t tileMaxX;
    uint32_t tileMaxY;
};

struct TiledUniforms {
    simd_float4x4 viewMatrix;
    simd_float4x4 projectionMatrix;
    simd_float4x4 viewProjectionMatrix;
    simd_float2 screenSize;
    simd_float2 focalLength;
    simd_float3 cameraPos;
    float _pad1;
    uint32_t numTilesX;
    uint32_t numTilesY;
    uint32_t numGaussians;
    uint32_t _pad2;
};

class TiledRasterizer {
public:
    TiledRasterizer(MTL::Device* device, MTL::Library* library, uint32_t maxGaussians);
    ~TiledRasterizer();
    
    // Enable/disable GPU sorting (default: enabled)
    void setUseGPUSort(bool enable) { useGPUSort = enable; }
    bool getUseGPUSort() const { return useGPUSort; }
    
    void forward(MTL::CommandQueue* queue,
                 MTL::Buffer* gaussianBuffer,
                 size_t gaussianCount,
                 const TiledUniforms& uniforms,
                 MTL::Texture* outputTexture);
    
    void backward(MTL::CommandQueue* queue,
                  MTL::Buffer* gaussianBuffer,
                  MTL::Buffer* gradientBuffer,
                  size_t gaussianCount,
                  const TiledUniforms& uniforms,
                  MTL::Texture* renderedTexture,
                  MTL::Texture* groundTruthTexture);
    
    // Accessors for debugging
    MTL::Buffer* getSortedIndices() { return valuesBuffer[0]; }
    MTL::Buffer* getTileRanges() { return tileRanges; }
    MTL::Buffer* getProjectedGaussians() { return projectedGaussians; }
    
    // Performance stats
    uint32_t getLastPairCount() const { return lastPairCount; }
    double getLastSortTimeMs() const { return lastSortTimeMs; }
    
private:
    static constexpr uint32_t TILE_SIZE = 16;
    static constexpr uint32_t AVG_TILES_PER_GAUSSIAN = 8;
    static constexpr uint32_t MAX_TILES_PER_GAUSSIAN = 256;
    static constexpr uint32_t THREADGROUP_SIZE = 256;
    
    MTL::Device* device;
    bool useGPUSort = true;
    
    // Compute pipelines
    MTL::ComputePipelineState* projectGaussiansPSO;
    MTL::ComputePipelineState* tiledForwardPSO;
    MTL::ComputePipelineState* tiledBackwardPSO;
    
    // GPU sorting pipelines (from gpu_sort.metal)
    MTL::ComputePipelineState* countTilePairsPSO;
    MTL::ComputePipelineState* generateTileKeysPSO;
    MTL::ComputePipelineState* histogram64PSO;
    MTL::ComputePipelineState* prefixSum256PSO;
    MTL::ComputePipelineState* scatter64SimplePSO;
    MTL::ComputePipelineState* buildTileRangesPSO;
    MTL::ComputePipelineState* clearHistogramPSO;
    
    // Projection buffer
    MTL::Buffer* projectedGaussians;
    
    // Tile sorting buffers
    MTL::Buffer* tileCountsBuffer;
    MTL::Buffer* tileOffsetsBuffer;
    MTL::Buffer* tileWriteIdxBuffer;
    MTL::Buffer* totalPairsBuffer;
    
    // 64-bit radix sort buffers (double-buffered)
    MTL::Buffer* keysBuffer[2];
    MTL::Buffer* valuesBuffer[2];
    MTL::Buffer* histogramBuffer;
    
    // Output buffers
    MTL::Buffer* tileRanges;
    
    // Per-pixel state for backward pass
    MTL::Buffer* perPixelTransmittance;
    MTL::Buffer* perPixelLastIdx;
    
    // Uniforms
    MTL::Buffer* uniformBuffer;
    
    // Capacities
    uint32_t maxGaussians;
    uint32_t maxTiles;
    uint32_t maxPairs;
    uint32_t currentWidth;
    uint32_t currentHeight;
    uint32_t numTilesX;
    uint32_t numTilesY;
    
    // Stats
    uint32_t lastPairCount;
    double lastSortTimeMs;
    
    void createPipelines(MTL::Library* library);
    void ensureBufferCapacity(uint32_t width, uint32_t height);
    void ensurePairsCapacity(uint32_t requiredPairs);
    
    // GPU sorting implementation
    void gpuSort(MTL::CommandQueue* queue, size_t gaussianCount);
    
    // CPU fallback (for debugging)
    void cpuSort(size_t gaussianCount);
};

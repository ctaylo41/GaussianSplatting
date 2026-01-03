//
//  tiled_rasterizer.hpp
//  GaussianSplatting
//
//  FIXED: Consistent buffer naming - removed gaussianValues/gaussianKeys confusion
//

#pragma once

#include <Metal/Metal.hpp>
#include <simd/simd.h>
#include "gradients.hpp"  // For GaussianGradients

// Must match shader definition EXACTLY
struct TileRange {
    uint32_t start;
    uint32_t count;
};

// Projected Gaussian data - must match shader struct ProjectedGaussian
struct ProjectedGaussian {
    simd_float2 screenPos;       // 8 bytes
    simd_float3 conic;           // 12 bytes - Inverse 2D covariance (a, b, c)
    float depth;                 // 4 bytes
    float opacity;               // 4 bytes - After sigmoid
    simd_float3 color;           // 12 bytes
    float radius;                // 4 bytes
    uint32_t tileMinX;           // 4 bytes
    uint32_t tileMinY;           // 4 bytes
    uint32_t tileMaxX;           // 4 bytes
    uint32_t tileMaxY;           // 4 bytes
    float _pad1;                 // 4 bytes - padding
    simd_float2 viewPos_xy;      // 8 bytes
    simd_float3 cov2D;           // 12 bytes - (a, b, c) before inversion
    float _pad2;                 // 4 bytes - padding
};  // Total: 88 bytes

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
    
    // Compute pipelines - Main rasterization
    MTL::ComputePipelineState* projectGaussiansPSO = nullptr;
    MTL::ComputePipelineState* tiledForwardPSO = nullptr;
    MTL::ComputePipelineState* tiledBackwardPSO = nullptr;
    
    // Compute pipelines - GPU sorting
    MTL::ComputePipelineState* countTilePairsPSO = nullptr;
    MTL::ComputePipelineState* generateTileKeysPSO = nullptr;
    MTL::ComputePipelineState* histogram64PSO = nullptr;
    MTL::ComputePipelineState* prefixSum256PSO = nullptr;
    MTL::ComputePipelineState* scatter64SimplePSO = nullptr;
    MTL::ComputePipelineState* buildTileRangesPSO = nullptr;
    MTL::ComputePipelineState* clearHistogramPSO = nullptr;
    
    // Projection buffer
    MTL::Buffer* projectedGaussians = nullptr;
    
    // Tile counting buffers (for GPU sort path)
    MTL::Buffer* tileCountsBuffer = nullptr;
    MTL::Buffer* tileOffsetsBuffer = nullptr;
    MTL::Buffer* tileWriteIdxBuffer = nullptr;
    MTL::Buffer* totalPairsBuffer = nullptr;
    
    // 64-bit radix sort buffers (double-buffered)
    // IMPORTANT: The final sorted indices are ALWAYS in valuesBuffer[0]
    MTL::Buffer* keysBuffer[2] = {nullptr, nullptr};
    MTL::Buffer* valuesBuffer[2] = {nullptr, nullptr};
    MTL::Buffer* histogramBuffer = nullptr;
    
    // Output buffers
    MTL::Buffer* tileRanges = nullptr;
    
    // Per-pixel state
    MTL::Buffer* perPixelTransmittance = nullptr;
    MTL::Buffer* perPixelLastIdx = nullptr;
    
    // Uniforms
    MTL::Buffer* uniformBuffer = nullptr;
    
    // Capacities
    uint32_t maxGaussians;
    uint32_t maxTiles = 0;
    uint32_t maxPairs = 0;
    uint32_t currentWidth = 0;
    uint32_t currentHeight = 0;
    uint32_t numTilesX = 0;
    uint32_t numTilesY = 0;
    
    // Stats
    uint32_t lastPairCount = 0;
    double lastSortTimeMs = 0;
    
    void createPipelines(MTL::Library* library);
    void ensureBufferCapacity(uint32_t width, uint32_t height);
    void ensurePairsCapacity(uint32_t requiredPairs);
    
    // GPU sorting implementation
    void gpuSort(MTL::CommandQueue* queue, size_t gaussianCount);
    
    // CPU fallback
    void cpuSort(size_t gaussianCount);
};

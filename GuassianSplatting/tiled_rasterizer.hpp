//
//  tiled_rasterizer.hpp
//  GaussianSplatting
//
//  FIXED: Proper buffer management for GPU sort
//  - Single consistent set of buffers (no gaussianKeys/gaussianValues confusion)
//  - Proper initialization to handle iteration 0 correctly
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

// Packed float3 to match Metal's packed_float3 (12 bytes, no padding)
struct __attribute__((packed)) packed_float3 {
    float x, y, z;
};

// Projected Gaussian data - must match shader struct ProjectedGaussian
// Using packed_float3 to match Metal's layout exactly
struct ProjectedGaussian {
    simd_float2 screenPos;       // 8 bytes, offset 0
    packed_float3 conic;         // 12 bytes, offset 8 - Inverse 2D covariance (a, b, c)
    float depth;                 // 4 bytes, offset 20
    float opacity;               // 4 bytes, offset 24 - After sigmoid
    packed_float3 color;         // 12 bytes, offset 28
    float radius;                // 4 bytes, offset 40
    uint32_t tileMinX;           // 4 bytes, offset 44
    uint32_t tileMinY;           // 4 bytes, offset 48
    uint32_t tileMaxX;           // 4 bytes, offset 52
    uint32_t tileMaxY;           // 4 bytes, offset 56
    float _pad1;                 // 4 bytes, offset 60 - padding
    simd_float2 viewPos_xy;      // 8 bytes, offset 64
    packed_float3 cov2D;         // 12 bytes, offset 72 - (a, b, c) before inversion
    float _pad2;                 // 4 bytes, offset 84 - padding
};  // Total: 88 bytes

// Verify struct layout matches Metal shader expectations
static_assert(sizeof(ProjectedGaussian) == 88, "ProjectedGaussian size must be 88 bytes to match Metal");
static_assert(offsetof(ProjectedGaussian, screenPos) == 0, "screenPos offset mismatch");
static_assert(offsetof(ProjectedGaussian, conic) == 8, "conic offset mismatch");
static_assert(offsetof(ProjectedGaussian, depth) == 20, "depth offset mismatch");
static_assert(offsetof(ProjectedGaussian, opacity) == 24, "opacity offset mismatch");
static_assert(offsetof(ProjectedGaussian, color) == 28, "color offset mismatch");
static_assert(offsetof(ProjectedGaussian, radius) == 40, "radius offset mismatch");
static_assert(offsetof(ProjectedGaussian, tileMinX) == 44, "tileMinX offset mismatch");
static_assert(offsetof(ProjectedGaussian, tileMinY) == 48, "tileMinY offset mismatch");
static_assert(offsetof(ProjectedGaussian, tileMaxX) == 52, "tileMaxX offset mismatch");
static_assert(offsetof(ProjectedGaussian, tileMaxY) == 56, "tileMaxY offset mismatch");
static_assert(offsetof(ProjectedGaussian, viewPos_xy) == 64, "viewPos_xy offset mismatch");
static_assert(offsetof(ProjectedGaussian, cov2D) == 72, "cov2D offset mismatch");

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
    MTL::Buffer* getSortedIndices() { return sortedIndicesBuffer; }
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
    MTL::ComputePipelineState* histogram64PSO = nullptr;
    MTL::ComputePipelineState* scatter64SimplePSO = nullptr;
    MTL::ComputePipelineState* buildTileRangesPSO = nullptr;
    
    // Projection buffer
    MTL::Buffer* projectedGaussians = nullptr;
    
    // Sorting buffers - double-buffered for radix sort
    MTL::Buffer* keysBuffer[2] = {nullptr, nullptr};
    MTL::Buffer* valuesBuffer[2] = {nullptr, nullptr};
    MTL::Buffer* histogramBuffer = nullptr;
    
    // CRITICAL: Single output buffer for sorted indices used by renderer
    // This is always populated with the final sorted result
    MTL::Buffer* sortedIndicesBuffer = nullptr;
    
    // Output buffers
    MTL::Buffer* tileRanges = nullptr;
    
    // Per-pixel state
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
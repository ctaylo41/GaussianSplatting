//
//  tile_rasterizer.hpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-28.
//

#pragma once
#include <Metal/Metal.hpp>
#include <simd/simd.h>
#include "ply_loader.hpp"
#include "gradients.hpp"
#include "gpu_sort.hpp"

// Must match shader definition exactly
struct TileRange {
    uint32_t start;
    uint32_t count;
};

// Projected Gaussian data for tiled rendering
struct ProjectedGaussian {
    simd_float2 screenPos;
    float conic[3];
    float depth;
    float opacity;
    float color[3];
    float radius;
    uint32_t tileMinX;
    uint32_t tileMinY;
    uint32_t tileMaxX;
    uint32_t tileMaxY;
    uint8_t colorClamped[3];  // Track which color channels were clamped (official 3DGS behavior)
    uint8_t _pad1;
    simd_float2 viewPos_xy;
    float cov2D[3];
    float viewDir[3];
};

// Uniforms for tiled rasterizer
struct TiledUniforms {
    simd_float4x4 viewMatrix;
    simd_float4x4 projectionMatrix;
    simd_float4x4 viewProjectionMatrix;
    simd_float2 screenSize;
    simd_float2 focalLength;
    simd_float3 cameraPos; 
    uint32_t numTilesX;
    uint32_t numTilesY;
    uint32_t numGaussians;
    uint32_t _pad2;
};

// Tiled rasterizer class for forward and backward passes
class TiledRasterizer {
public:
    // Constructor and destructor 
    TiledRasterizer(MTL::Device* device, MTL::Library* library, uint32_t maxGaussians);
    ~TiledRasterizer();
    
    // Forward and backward methods
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
    
    // Get projected gaussians buffer for density control radius tracking
    MTL::Buffer* getProjectedGaussians() const { return projectedGaussians; }

    // Get stage-1 intermediate gradient buffer (for debug telemetry)
    MTL::Buffer* getRenderGradientBuffer() const { return renderGradientBuffer; }

    // Get per-pixel loss gradient buffer (output of computePixelGradient)
    MTL::Buffer* getPixelGradientBuffer() const { return pixelGradientBuffer; }

    // True when the most recent forward/backward had a command buffer abort.
    // The buffers those passes write to are shared-storage and are NOT rolled back
    // on failure, so a caller that ignores this will feed partial/stale gradients
    // into the optimizer. Always check before stepping Adam.
    bool lastPassFailed() const { return passFailed; }
    void clearPassFailed() { passFailed = false; }

    // Cumulative count of aborted command buffers since startup.
    uint64_t getAbortCount() const { return abortCount; }

private:
    // Commits, waits, and reports any execution error. Returns false on abort.
    bool commitAndCheck(MTL::CommandBuffer* cmdBuffer, const char* label);

    bool passFailed = false;
    uint64_t abortCount = 0;

    static constexpr uint32_t TILE_SIZE = 16;
    // Average Gaussians touch 4-8 tiles depending on size
    static constexpr uint32_t AVG_TILES_PER_GAUSSIAN = 16;
    
    // Metal device and library
    MTL::Device* device;
    MTL::Library* library;
    
    // Compute pipelines
    MTL::ComputePipelineState* projectGaussiansPSO;
    MTL::ComputePipelineState* tiledForwardPSO;
    MTL::ComputePipelineState* tiledBackwardPSO;
    MTL::ComputePipelineState* buildTileRangesPSO;
    MTL::ComputePipelineState* generatePairsPSO;
    MTL::ComputePipelineState* computeSSIMGradCoeffsPSO;
    MTL::ComputePipelineState* computePixelGradientPSO;
    MTL::ComputePipelineState* preprocessBackwardPSO;

    // Buffers
    MTL::Buffer* projectedGaussians;
    MTL::Buffer* gaussianKeys;
    MTL::Buffer* gaussianValues;
    MTL::Buffer* tileRanges;
    MTL::Buffer* totalPairsBuffer;
    MTL::Buffer* perPixelLastIdx;
    MTL::Buffer* uniformBuffer;

    // Atomic counter for GPU pair generation
    MTL::Buffer* pairCounterBuffer;

    // SSIM gradient intermediate buffers 
    MTL::Buffer* ssimCoeffKBuffer = nullptr;
    MTL::Buffer* ssimCoeffLBuffer = nullptr;
    MTL::Buffer* ssimCoeffMBuffer = nullptr;
    MTL::Buffer* pixelGradientBuffer = nullptr;

    // Intermediate render gradients buffer (Stage 1 output, Stage 2 input)
    MTL::Buffer* renderGradientBuffer = nullptr;

    // Track which buffers contain the sorted data
    MTL::Buffer* activeSortedKeys;
    MTL::Buffer* activeSortedValues;  
    
    // Capacity tracking
    uint32_t maxGaussians;
    uint32_t maxTiles;
    uint32_t maxPairs;
    uint32_t currentWidth;
    uint32_t currentHeight;
    uint32_t numTilesX;
    uint32_t numTilesY;
    
    // GPU radix sort for fast 64-bit sorting on GPU
    GPURadixSort64* gpuRadixSort;
    
    // Helper methods
    void createPipelines(MTL::Library* library);
    void ensureBufferCapacity(uint32_t width, uint32_t height, size_t gaussianCount);
    void ensurePairsCapacity(uint32_t requiredPairs);
};

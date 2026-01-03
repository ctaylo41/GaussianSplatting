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

// Must match shader definition EXACTLY
struct TileRange {
    uint32_t start;
    uint32_t count;
};

// Projected Gaussian data for tiled rendering
// CRITICAL: Use float arrays instead of simd_float3 to match Metal packed_float3 layout
// simd_float2 has 8-byte alignment, so compiler adds 4 bytes padding after tileMaxY
struct ProjectedGaussian {
    simd_float2 screenPos;   // 8 bytes, offset 0
    float conic[3];          // 12 bytes, offset 8 - Inverse 2D covariance (use array for packed layout)
    float depth;             // 4 bytes, offset 20
    float opacity;           // 4 bytes, offset 24 - After sigmoid
    float color[3];          // 12 bytes, offset 28 (use array for packed layout)
    float radius;            // 4 bytes, offset 40
    uint32_t tileMinX;       // 4 bytes, offset 44
    uint32_t tileMinY;       // 4 bytes, offset 48
    uint32_t tileMaxX;       // 4 bytes, offset 52
    uint32_t tileMaxY;       // 4 bytes, offset 56
    float _pad1;             // 4 bytes, offset 60 - explicit padding for simd_float2 alignment
    simd_float2 viewPos_xy;  // 8 bytes, offset 64 - For gradient computation
    float cov2D[3];          // 12 bytes, offset 72 - (a, b, c) - 2D covariance BEFORE inversion (for backward pass)
    float _pad2;             // 4 bytes, offset 84 - padding to make struct 88 bytes (multiple of 8)
};  // Total: 88 bytes with predictable layout

struct TiledUniforms {
    simd_float4x4 viewMatrix;
    simd_float4x4 projectionMatrix;
    simd_float4x4 viewProjectionMatrix;
    simd_float2 screenSize;
    simd_float2 focalLength;
    simd_float3 cameraPos;     // 16 bytes (12 data + 4 implicit padding)
    uint32_t numTilesX;
    uint32_t numTilesY;
    uint32_t numGaussians;
    uint32_t _pad2;
};

class TiledRasterizer {
public:
    TiledRasterizer(MTL::Device* device, MTL::Library* library, uint32_t maxGaussians);
    ~TiledRasterizer();
    
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
    
private:
    static constexpr uint32_t TILE_SIZE = 16;
    // Average Gaussians touch ~4-8 tiles, set reasonable max
    static constexpr uint32_t AVG_TILES_PER_GAUSSIAN = 8;
    
    MTL::Device* device;
    
    // Compute pipelines
    MTL::ComputePipelineState* projectGaussiansPSO;
    MTL::ComputePipelineState* tiledForwardPSO;
    MTL::ComputePipelineState* tiledBackwardPSO;
    
    // Buffers
    MTL::Buffer* projectedGaussians;
    MTL::Buffer* gaussianKeys;
    MTL::Buffer* gaussianValues;
    MTL::Buffer* tileRanges;
    MTL::Buffer* totalPairsBuffer;
    MTL::Buffer* perPixelLastIdx;
    MTL::Buffer* uniformBuffer;
    
    uint32_t maxGaussians;
    uint32_t maxTiles;
    uint32_t maxPairs;
    uint32_t currentWidth;
    uint32_t currentHeight;
    uint32_t numTilesX;
    uint32_t numTilesY;
    
    void createPipelines(MTL::Library* library);
    void ensureBufferCapacity(uint32_t width, uint32_t height, size_t gaussianCount);
    void ensurePairsCapacity(uint32_t requiredPairs);
};

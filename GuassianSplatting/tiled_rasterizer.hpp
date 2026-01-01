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
struct ProjectedGaussian {
    simd_float2 screenPos;
    simd_float3 conic;       // Inverse 2D covariance
    float depth;
    float opacity;           // After sigmoid
    simd_float3 color;
    float radius;
    uint32_t tileMinX;
    uint32_t tileMinY;
    uint32_t tileMaxX;
    uint32_t tileMaxY;
    simd_float2 viewPos_xy;  // For gradient computation
    simd_float3 cov2D;       // (a, b, c) - 2D covariance BEFORE inversion (for backward pass)
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

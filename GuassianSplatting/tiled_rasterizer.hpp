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

// Must match shader definition
struct TileRange {
    uint32_t start;
    uint32_t count;
};

// Projected Gaussian data cached for backward pass
struct ProjectedGaussian {
    simd_float2 screenPos;
    simd_float3 conic;       // Inverse 2D covariance (a, b, c)
    float depth;
    float opacity;
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
    
    // Render Gaussians to texture, storing intermediate state for backward pass
    void forward(MTL::CommandQueue* queue,
                 MTL::Buffer* gaussianBuffer,
                 size_t gaussianCount,
                 const TiledUniforms& uniforms,
                 MTL::Texture* outputTexture);
    
    // Compute gradients using stored intermediate state
    void backward(MTL::CommandQueue* queue,
                  MTL::Buffer* gaussianBuffer,
                  MTL::Buffer* gradientBuffer,
                  size_t gaussianCount,
                  const TiledUniforms& uniforms,
                  MTL::Texture* renderedTexture,
                  MTL::Texture* groundTruthTexture);
    
    // Get buffers for debugging
    MTL::Buffer* getSortedIndices() { return sortedGaussianIndices; }
    MTL::Buffer* getTileRanges() { return tileRanges; }
    MTL::Buffer* getProjectedGaussians() { return projectedGaussians; }
    
private:
    static constexpr uint32_t TILE_SIZE = 16;
    static constexpr uint32_t MAX_PAIRS_PER_GAUSSIAN = 64;
    
    MTL::Device* device;
    
    // Compute pipelines
    MTL::ComputePipelineState* projectGaussiansPSO;
    MTL::ComputePipelineState* countTilesPSO;
    MTL::ComputePipelineState* writeGaussianKeysPSO;
    MTL::ComputePipelineState* tiledForwardPSO;
    MTL::ComputePipelineState* tiledBackwardPSO;
    
    // Buffers for projection
    MTL::Buffer* projectedGaussians;
    
    // Buffers for tiling
    MTL::Buffer* tileCounts;
    MTL::Buffer* tileOffsets;
    MTL::Buffer* tileWriteOffsets;
    MTL::Buffer* gaussianKeys;
    MTL::Buffer* gaussianValues;
    MTL::Buffer* tileRanges;
    MTL::Buffer* sortedGaussianIndices;
    MTL::Buffer* totalPairsBuffer;
    
    // Buffers for backward pass
    MTL::Buffer* perPixelTransmittance;
    MTL::Buffer* perPixelLastIdx;
    
    // Uniform buffer
    MTL::Buffer* uniformBuffer;
    
    uint32_t maxGaussians;
    uint32_t maxTiles;
    uint32_t maxPairs;
    uint32_t currentWidth;
    uint32_t currentHeight;
    uint32_t numTilesX;
    uint32_t numTilesY;
    
    void createPipelines(MTL::Library* library);
    void ensureBufferCapacity(uint32_t width, uint32_t height);
};

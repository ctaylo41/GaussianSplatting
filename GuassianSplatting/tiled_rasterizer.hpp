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

// Must match shader definition exactly
struct TileRange {
    uint32_t start;
    uint32_t count;
};

// Projected Gaussian data for tiled rendering
// Use float arrays instead of simd_float3 to match Metal packed_float3 layout
// simd_float2 has 8-byte alignment, so compiler adds 4 bytes padding after tileMaxY
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
    float _pad1;             
    simd_float2 viewPos_xy;  
    float cov2D[3];          
    float _pad2;             
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
    
private:
    static constexpr uint32_t TILE_SIZE = 16;
    // Average Gaussians touch ~4-8 tiles
    static constexpr uint32_t AVG_TILES_PER_GAUSSIAN = 8;
    
    // Metal device
    MTL::Device* device;
    
    // Compute pipelines
    MTL::ComputePipelineState* projectGaussiansPSO;
    MTL::ComputePipelineState* tiledForwardPSO;
    MTL::ComputePipelineState* tiledBackwardPSO;
    MTL::ComputePipelineState* buildTileRangesPSO;
    MTL::ComputePipelineState* generatePairsPSO;
    
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
    
    // Capacity tracking
    uint32_t maxGaussians;
    uint32_t maxTiles;
    uint32_t maxPairs;
    uint32_t currentWidth;
    uint32_t currentHeight;
    uint32_t numTilesX;
    uint32_t numTilesY;
    
    // Helper methods
    void createPipelines(MTL::Library* library);
    void ensureBufferCapacity(uint32_t width, uint32_t height, size_t gaussianCount);
    void ensurePairsCapacity(uint32_t requiredPairs);
};

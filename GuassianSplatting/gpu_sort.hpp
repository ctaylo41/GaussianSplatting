//
//  gpu_sort.hpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-26.
//

#pragma once

#include <Metal/Metal.hpp>
#include <simd/simd.h>

class GPURadixSort {
public:
    GPURadixSort(MTL::Device* device, size_t maxElements);
    ~GPURadixSort();
    
    //sort gaussians by depth and return buffer of sorted indices
    MTL::Buffer* sort(MTL::CommandQueue* queue, MTL::Buffer* positionBuffer, simd_float3 cameraPos, size_t numElements);
    
    // CPU fallback sort for debugging
    MTL::Buffer* sortCPU(MTL::Buffer* positionBuffer, simd_float3 cameraPos, size_t numElements);
    
    MTL::Buffer* getSortedIndices() { return valuesBuffer[0]; }
    
private:
    MTL::Device* device;
    size_t maxElements;
    
    // Compute pipelines
    MTL::ComputePipelineState* computeDepthsPipeline;
    MTL::ComputePipelineState* histogramPipeline;
    MTL::ComputePipelineState* prefixSumPipeline;
    MTL::ComputePipelineState* scatterPipeline;
    
    // double-buffered key/value arrays
    MTL::Buffer* keysBuffer[2];
    MTL::Buffer* valuesBuffer[2];
    
    // histogram and prefix sums
    MTL::Buffer* histogramBuffer;
    MTL::Buffer* prefixSumBuffer;
    MTL::Buffer* scatterOffsetsBuffer;  // Copy of prefix sums for scatter (atomic ops modify this)
    
    // CPU sort buffer
    MTL::Buffer* cpuSortedIndices;
    
    void createPipelines(MTL::Device* device);
};

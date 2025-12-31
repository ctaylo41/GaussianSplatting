//
//  gpu_sort.mm
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-26.
//

#include "gpu_sort.hpp"
#include <iostream>
#include <algorithm>
#include <vector>

GPURadixSort::GPURadixSort(MTL::Device* device, size_t maxElements)
    :device(device),
    maxElements(maxElements)
{
    createPipelines(device);
    
    keysBuffer[0] = device->newBuffer(maxElements * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    keysBuffer[1] = device->newBuffer(maxElements * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    valuesBuffer[0] = device->newBuffer(maxElements * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    valuesBuffer[1] = device->newBuffer(maxElements * sizeof(uint32_t), MTL::ResourceStorageModePrivate);

    histogramBuffer = device->newBuffer(256 * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    prefixSumBuffer = device->newBuffer(256 * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    // Separate buffer for scatter - atomic ops will modify this, keeping prefixSumBuffer clean
    scatterOffsetsBuffer = device->newBuffer(256 * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    
    // CPU sort buffer for fallback
    cpuSortedIndices = device->newBuffer(maxElements * sizeof(uint32_t), MTL::ResourceStorageModeShared);
}

GPURadixSort::~GPURadixSort() {
    keysBuffer[0]->release();
    keysBuffer[1]->release();
    valuesBuffer[0]->release();
    valuesBuffer[1]->release();
    histogramBuffer->release();
    prefixSumBuffer->release();
    scatterOffsetsBuffer->release();
    cpuSortedIndices->release();
    computeDepthsPipeline->release();
    histogramPipeline->release();
    prefixSumPipeline->release();
    scatterPipeline->release();
}

void GPURadixSort::createPipelines(MTL::Device *device) {
    NS::Error* error = nullptr;
    
    MTL::Library* library = device->newDefaultLibrary();
    if(!library) {
        std::cerr << "Failed to load sort shader library" << std::endl;
        std::exit(1);
    }
    
    auto makeComputePipeline = [&](const char* name) -> MTL::ComputePipelineState* {
        MTL::Function* func = library->newFunction(NS::String::string(name, NS::ASCIIStringEncoding));
        if(!func) {
            std::cerr << "Failed to find function " << name << std::endl;
            std::exit(1);
        }
        
        MTL::ComputePipelineState* pipeline = device->newComputePipelineState(func, &error);
        if(!pipeline) {
            std::cerr << "Failed to create pipeline for " << name << ": " << error << std::endl;
            std::exit(1);
        }
        func->release();
        return pipeline;
    };
    
    computeDepthsPipeline = makeComputePipeline("computeDepth");
    histogramPipeline = makeComputePipeline("histogram");
    prefixSumPipeline = makeComputePipeline("prefixSum");
    scatterPipeline = makeComputePipeline("scatter");
    
    library->release();
}

MTL::Buffer* GPURadixSort::sort(MTL::CommandQueue *queue, MTL::Buffer *positionBuffer, simd_float3 cameraPos, size_t numElements) {
    
    static bool sortDebugPrinted = false;
    
    MTL::CommandBuffer* cmdBuffer = queue->commandBuffer();
    
    //inner scope block
    // compute depths and init key value pairs
    {
        MTL::ComputeCommandEncoder* encoder = cmdBuffer->computeCommandEncoder();
        encoder->setComputePipelineState(computeDepthsPipeline);
        encoder->setBuffer(positionBuffer,0, 0);
        encoder->setBytes(&cameraPos, sizeof(simd_float3), 1);
        encoder->setBuffer(keysBuffer[0],0,2);
        encoder->setBuffer(valuesBuffer[0],0,3);
        
        MTL::Size gridSize = MTL::Size(numElements, 1, 1);
        MTL::Size threadgroupSize = MTL::Size(std::min((size_t)256, numElements), 1, 1);
        encoder->dispatchThreads(gridSize, threadgroupSize);
        encoder->endEncoding();
    }
    
    //radix sort passes 4 passes 8 bits each
    int srcIdx = 0;
    uint32_t numElementsU32 = (uint32_t)numElements;
    
    for(uint32_t bitOffset = 0; bitOffset<32;bitOffset+=8) {
        int dstIdx = 1 - srcIdx;
        
        MTL::BlitCommandEncoder* blit = cmdBuffer->blitCommandEncoder();
        blit->fillBuffer(histogramBuffer, NS::Range(0,256*sizeof(uint32_t)), 0);
        blit->endEncoding();
        
        {
            MTL::ComputeCommandEncoder* encoder = cmdBuffer->computeCommandEncoder();
            encoder->setComputePipelineState(histogramPipeline);
            encoder->setBuffer(keysBuffer[srcIdx],0, 0);
            encoder->setBuffer(histogramBuffer,0,1);
            encoder->setBytes(&bitOffset, sizeof(uint32_t),2);
            encoder->setBytes(&numElementsU32, sizeof(uint32_t), 3);
            
            MTL::Size gridSize = MTL::Size(numElements, 1, 1);
            MTL::Size threadgroupSize = MTL::Size(256, 1, 1);
            encoder->dispatchThreads(gridSize, threadgroupSize);
            encoder->endEncoding();
        }
        
        {
            MTL::ComputeCommandEncoder* encoder = cmdBuffer->computeCommandEncoder();
            encoder->setComputePipelineState(prefixSumPipeline);
            encoder->setBuffer(histogramBuffer,0, 0);
            encoder->setBuffer(prefixSumBuffer,0, 1);
            
            MTL::Size gridSize = MTL::Size(256,1,1);
            MTL::Size threadgroupSize = MTL::Size(256, 1, 1);
            encoder->dispatchThreadgroups(MTL::Size(1,1,1), threadgroupSize);
            encoder->endEncoding();
        }
        
        // Copy prefix sums to scatter offsets buffer (scatter will atomically modify this)
        {
            MTL::BlitCommandEncoder* copyBlit = cmdBuffer->blitCommandEncoder();
            copyBlit->copyFromBuffer(prefixSumBuffer, 0, scatterOffsetsBuffer, 0, 256 * sizeof(uint32_t));
            copyBlit->endEncoding();
        }
        
        {
            MTL::ComputeCommandEncoder* encoder = cmdBuffer->computeCommandEncoder();
            encoder->setComputePipelineState(scatterPipeline);
            encoder->setBuffer(keysBuffer[srcIdx], 0, 0);
            encoder->setBuffer(valuesBuffer[srcIdx],0, 1);
            encoder->setBuffer(keysBuffer[dstIdx],0,2);
            encoder->setBuffer(valuesBuffer[dstIdx], 0, 3);
            encoder->setBuffer(scatterOffsetsBuffer, 0, 4);  // Use copy, not original
            encoder->setBytes(&bitOffset, sizeof(uint32_t), 5);
            encoder->setBytes(&numElementsU32, sizeof(uint32_t), 6);
            
            MTL::Size gridSize = MTL::Size(numElements, 1, 1);
            MTL::Size threadgroupSize = MTL::Size(256, 1, 1);
            encoder->dispatchThreads(gridSize, threadgroupSize);
            encoder->endEncoding();
        }
        
        srcIdx = dstIdx;
    }
    
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();
    
    // Debug: verify sorting is working
    if (!sortDebugPrinted && numElements > 0) {
        sortDebugPrinted = true;
        
        // Copy sorted indices to readable buffer
        MTL::Buffer* debugBuffer = device->newBuffer(numElements * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        MTL::CommandBuffer* copyCmd = queue->commandBuffer();
        MTL::BlitCommandEncoder* blit = copyCmd->blitCommandEncoder();
        blit->copyFromBuffer(valuesBuffer[srcIdx], 0, debugBuffer, 0, numElements * sizeof(uint32_t));
        blit->endEncoding();
        copyCmd->commit();
        copyCmd->waitUntilCompleted();
        
        uint32_t* sortedIdx = (uint32_t*)debugBuffer->contents();
        simd_float3* positions = (simd_float3*)positionBuffer->contents();
        
        printf("\n=== SORT DEBUG ===\n");
        printf("Camera pos: (%.3f, %.3f, %.3f)\n", cameraPos.x, cameraPos.y, cameraPos.z);
        printf("Num elements: %zu\n", numElements);
        
        // Check first 10 sorted indices and their depths
        printf("First 10 sorted Gaussians (should be back-to-front, farthest first):\n");
        float prevDepth = FLT_MAX;
        bool sortingCorrect = true;
        for (int i = 0; i < 10 && i < (int)numElements; i++) {
            uint32_t idx = sortedIdx[i];
            simd_float3 pos = positions[idx];
            simd_float3 diff = pos - cameraPos;
            float depth = simd_dot(diff, diff);  // squared distance
            if (depth > prevDepth) {
                sortingCorrect = false;
            }
            printf("  [%d] idx=%u pos=(%.3f,%.3f,%.3f) depth²=%.3f\n", 
                   i, idx, pos.x, pos.y, pos.z, depth);
            prevDepth = depth;
        }
        
        // Check last 10 (should be closest)
        printf("Last 10 sorted Gaussians (should be closest):\n");
        for (int i = 0; i < 10 && i < (int)numElements; i++) {
            int idx_pos = (int)numElements - 10 + i;
            if (idx_pos < 0) idx_pos = i;
            uint32_t idx = sortedIdx[idx_pos];
            simd_float3 pos = positions[idx];
            simd_float3 diff = pos - cameraPos;
            float depth = simd_dot(diff, diff);
            printf("  [%d] idx=%u pos=(%.3f,%.3f,%.3f) depth²=%.3f\n", 
                   idx_pos, idx, pos.x, pos.y, pos.z, depth);
        }
        
        printf("Sorting appears %s\n", sortingCorrect ? "CORRECT" : "INCORRECT");
        
        debugBuffer->release();
    }
    
    return valuesBuffer[srcIdx];
}

// CPU fallback sort for debugging - guaranteed correct
MTL::Buffer* GPURadixSort::sortCPU(MTL::Buffer* positionBuffer, simd_float3 cameraPos, size_t numElements) {
    static bool debugPrinted = false;
    
    simd_float3* positions = (simd_float3*)positionBuffer->contents();
    uint32_t* sortedIndices = (uint32_t*)cpuSortedIndices->contents();
    
    // Create vector of (depth, index) pairs
    std::vector<std::pair<float, uint32_t>> depthIndex(numElements);
    for (size_t i = 0; i < numElements; i++) {
        simd_float3 diff = positions[i] - cameraPos;
        float depth = simd_dot(diff, diff);  // squared distance
        depthIndex[i] = {depth, (uint32_t)i};
    }
    
    // Sort by depth descending (back-to-front: farthest first)
    // With pre-multiplied alpha and blend mode (src*1 + dst*(1-srcAlpha)),
    // back-to-front rendering produces correct alpha compositing
    std::sort(depthIndex.begin(), depthIndex.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;  // Descending order - FARTHEST FIRST
    });
    
    // Copy sorted indices to buffer
    for (size_t i = 0; i < numElements; i++) {
        sortedIndices[i] = depthIndex[i].second;
    }
    
    if (!debugPrinted) {
        debugPrinted = true;
        printf("\n=== CPU SORT DEBUG ===\n");
        printf("Camera pos: (%.3f, %.3f, %.3f)\n", cameraPos.x, cameraPos.y, cameraPos.z);
        printf("Num elements: %zu\n", numElements);
        
        printf("First 10 sorted Gaussians (farthest first):\n");
        for (int i = 0; i < 10 && i < (int)numElements; i++) {
            uint32_t idx = sortedIndices[i];
            simd_float3 pos = positions[idx];
            simd_float3 diff = pos - cameraPos;
            float depth = simd_dot(diff, diff);
            printf("  [%d] idx=%u pos=(%.3f,%.3f,%.3f) depth²=%.3f\n", 
                   i, idx, pos.x, pos.y, pos.z, depth);
        }
        
        printf("Last 10 sorted Gaussians (closest):\n");
        for (int i = 0; i < 10 && i < (int)numElements; i++) {
            int idx_pos = (int)numElements - 10 + i;
            if (idx_pos < 0) idx_pos = i;
            uint32_t idx = sortedIndices[idx_pos];
            simd_float3 pos = positions[idx];
            simd_float3 diff = pos - cameraPos;
            float depth = simd_dot(diff, diff);
            printf("  [%d] idx=%u pos=(%.3f,%.3f,%.3f) depth²=%.3f\n", 
                   idx_pos, idx, pos.x, pos.y, pos.z, depth);
        }
    }
    
    return cpuSortedIndices;
}

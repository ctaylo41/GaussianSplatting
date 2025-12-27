//
//  gpu_sort.mm
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-26.
//

#include "gpu_sort.hpp"
#include <iostream>

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
}

GPURadixSort::~GPURadixSort() {
    keysBuffer[0]->release();
    keysBuffer[1]->release();
    valuesBuffer[0]->release();
    valuesBuffer[1]->release();
    histogramBuffer->release();
    prefixSumBuffer->release();
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
        
        {
            MTL::ComputeCommandEncoder* encoder = cmdBuffer->computeCommandEncoder();
            encoder->setComputePipelineState(scatterPipeline);
            encoder->setBuffer(keysBuffer[srcIdx], 0, 0);
            encoder->setBuffer(valuesBuffer[srcIdx],0, 1);
            encoder->setBuffer(keysBuffer[dstIdx],0,2);
            encoder->setBuffer(valuesBuffer[dstIdx], 0, 3);
            encoder->setBuffer(prefixSumBuffer, 0, 4);
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
    
    return valuesBuffer[srcIdx];
}

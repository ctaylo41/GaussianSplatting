//
//  gpu_sort.mm
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-31.
//

#include "gpu_sort.hpp"
#include <iostream>
#include <algorithm>
#include <cstring>

// Helper to create compute pipeline
static MTL::ComputePipelineState* createPipeline(MTL::Device* device,
                                                  MTL::Library* library,
                                                  const char* functionName) {
    NS::Error* error = nullptr;
    
    // Create function from library
    auto funcName = NS::String::string(functionName, NS::ASCIIStringEncoding);
    MTL::Function* func = library->newFunction(funcName);
    
    if (!func) {
        std::cerr << "Failed to find function: " << functionName << std::endl;
        return nullptr;
    }
    
    // Create compute pipeline state
    MTL::ComputePipelineState* pso = device->newComputePipelineState(func, &error);
    func->release();
    
    if (!pso) {
        std::cerr << "Failed to create pipeline for " << functionName;
        if (error) {
            std::cerr << ": " << error->localizedDescription()->utf8String();
        }
        std::cerr << std::endl;
        return nullptr;
    }
    
    return pso;
}

// GPURadixSort32 Implementation
GPURadixSort32::GPURadixSort32(MTL::Device* device, MTL::Library* library, size_t maxElements)
    : device(device)
    , maxElements(maxElements)
{
    createPipelines(library);
    
    // Allocate double buffers initialize to zero to avoid undefined behavior
    keysBuffers[0] = device->newBuffer(maxElements * sizeof(uint32_t),
                                       MTL::ResourceStorageModeShared);
    keysBuffers[1] = device->newBuffer(maxElements * sizeof(uint32_t),
                                       MTL::ResourceStorageModeShared);
    valuesBuffers[0] = device->newBuffer(maxElements * sizeof(uint32_t),
                                         MTL::ResourceStorageModeShared);
    valuesBuffers[1] = device->newBuffer(maxElements * sizeof(uint32_t),
                                         MTL::ResourceStorageModeShared);
    
    // Zero initialize the value buffers with identity permutation
    // This ensures that even if sorting fails we have valid indices
    uint32_t* vals0 = (uint32_t*)valuesBuffers[0]->contents();
    uint32_t* vals1 = (uint32_t*)valuesBuffers[1]->contents();
    for (size_t i = 0; i < maxElements; i++) {
        vals0[i] = (uint32_t)i;
        vals1[i] = (uint32_t)i;
    }
    
    // Histogram buffer
    histogramBuffer = device->newBuffer(RADIX_SIZE * sizeof(uint32_t),
                                        MTL::ResourceStorageModeShared);
    memset(histogramBuffer->contents(), 0, RADIX_SIZE * sizeof(uint32_t));
    
    // Digit counters buffer
    digitCountersBuffer = device->newBuffer(RADIX_SIZE * sizeof(uint32_t),
                                            MTL::ResourceStorageModeShared);
    memset(digitCountersBuffer->contents(), 0, RADIX_SIZE * sizeof(uint32_t));
    
    // Camera position buffer
    cameraPosBuffer = device->newBuffer(sizeof(simd_float3),
                                        MTL::ResourceStorageModeShared);
}

// Destructor
GPURadixSort32::~GPURadixSort32() {
    if (keysBuffers[0]) keysBuffers[0]->release();
    if (keysBuffers[1]) keysBuffers[1]->release();
    if (valuesBuffers[0]) valuesBuffers[0]->release();
    if (valuesBuffers[1]) valuesBuffers[1]->release();
    if (histogramBuffer) histogramBuffer->release();
    if (digitCountersBuffer) digitCountersBuffer->release();
    if (cameraPosBuffer) cameraPosBuffer->release();
    
    if (computeDepthsPSO) computeDepthsPSO->release();
    if (histogram32PSO) histogram32PSO->release();
    if (prefixSum256PSO) prefixSum256PSO->release();
    if (scatter32SimplePSO) scatter32SimplePSO->release();
    if (scatter32OptimizedPSO) scatter32OptimizedPSO->release();
    if (clearHistogramPSO) clearHistogramPSO->release();
}

// Create compute pipelines
void GPURadixSort32::createPipelines(MTL::Library* library) {
    computeDepthsPSO = createPipeline(device, library, "computeDepths");
    histogram32PSO = createPipeline(device, library, "histogram32");
    prefixSum256PSO = createPipeline(device, library, "prefixSum256");
    scatter32SimplePSO = createPipeline(device, library, "scatter32Simple");
    scatter32OptimizedPSO = createPipeline(device, library, "scatterOptimized32");
    clearHistogramPSO = createPipeline(device, library, "clearHistogram");
    
    // Verify all pipelines were created
    if (!computeDepthsPSO) {
        std::cerr << "ERROR: computeDepthsPSO is null!" << std::endl;
    }
    if (!histogram32PSO) {
        std::cerr << "ERROR: histogram32PSO is null!" << std::endl;
    }
    if (!scatter32SimplePSO) {
        std::cerr << "ERROR: scatter32SimplePSO is null!" << std::endl;
    }
    if (!clearHistogramPSO) {
        std::cerr << "ERROR: clearHistogramPSO is null!" << std::endl;
    }
}

// Ensure capacity of buffers
void GPURadixSort32::ensureCapacity(size_t numElements) {
    if (numElements <= maxElements) return;
    
    maxElements = std::max(numElements, maxElements * 2);
    
    // Reallocate buffers
    if (keysBuffers[0]) keysBuffers[0]->release();
    if (keysBuffers[1]) keysBuffers[1]->release();
    if (valuesBuffers[0]) valuesBuffers[0]->release();
    if (valuesBuffers[1]) valuesBuffers[1]->release();
    
    keysBuffers[0] = device->newBuffer(maxElements * sizeof(uint32_t),
                                       MTL::ResourceStorageModeShared);
    keysBuffers[1] = device->newBuffer(maxElements * sizeof(uint32_t),
                                       MTL::ResourceStorageModeShared);
    valuesBuffers[0] = device->newBuffer(maxElements * sizeof(uint32_t),
                                         MTL::ResourceStorageModeShared);
    valuesBuffers[1] = device->newBuffer(maxElements * sizeof(uint32_t),
                                         MTL::ResourceStorageModeShared);
    
    // Initialize new buffers with identity permutation
    uint32_t* vals0 = (uint32_t*)valuesBuffers[0]->contents();
    uint32_t* vals1 = (uint32_t*)valuesBuffers[1]->contents();
    for (size_t i = 0; i < maxElements; i++) {
        vals0[i] = (uint32_t)i;
        vals1[i] = (uint32_t)i;
    }
}

// Main sort function
MTL::Buffer* GPURadixSort32::sort(MTL::CommandQueue* queue,
                                   MTL::Buffer* positionBuffer,
                                   simd_float3 cameraPos,
                                   size_t numElements) {
    // Handle edge cases
    if (numElements == 0) {
        return valuesBuffers[0];
    }
    
    // Verify pipeline state objects exist
    if (!computeDepthsPSO || !histogram32PSO || !scatter32SimplePSO || !clearHistogramPSO) {
        std::cerr << "ERROR: GPU sort pipelines not initialized! Returning identity permutation." << std::endl;
        // Return identity permutation as fallback
        uint32_t* vals = (uint32_t*)valuesBuffers[0]->contents();
        for (size_t i = 0; i < numElements; i++) {
            vals[i] = (uint32_t)i;
        }
        currentBuffer = 0;
        return valuesBuffers[0];
    }
    
    // Ensure capacity
    ensureCapacity(numElements);
    
    uint32_t numElementsU32 = (uint32_t)numElements;
    
    // Copy camera position to buffer
    memcpy(cameraPosBuffer->contents(), &cameraPos, sizeof(simd_float3));

    MTL::CommandBuffer* cmdBuffer = queue->commandBuffer();
    
    // Step 1 Compute depths and initialize keys/values
    {
        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(computeDepthsPSO);
        enc->setBuffer(positionBuffer, 0, 0);
        enc->setBuffer(cameraPosBuffer, 0, 1);
        enc->setBuffer(keysBuffers[0], 0, 2);
        enc->setBuffer(valuesBuffers[0], 0, 3);
        enc->setBytes(&numElementsU32, sizeof(uint32_t), 4);
        
        MTL::Size grid = MTL::Size(numElements, 1, 1);
        MTL::Size tg = MTL::Size(THREADGROUP_SIZE, 1, 1);
        enc->dispatchThreads(grid, tg);
        enc->endEncoding();
    }
    
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();

    // 4 passes of radix sort 8 bits per pass
    int srcIdx = 0;
    
    for (uint32_t pass = 0; pass < NUM_PASSES; pass++) {
        uint32_t bitOffset = pass * 8;
        int dstIdx = 1 - srcIdx;
        
        // Clear histogram using CPU memset for guaranteed zeroing
        // The GPU clear was potentially racing with histogram32?
        memset(histogramBuffer->contents(), 0, RADIX_SIZE * sizeof(uint32_t));
        
        // Clear digit counters for scatter phase
        memset(digitCountersBuffer->contents(), 0, RADIX_SIZE * sizeof(uint32_t));
        
        cmdBuffer = queue->commandBuffer();
        
        // Build histogram
        {
            MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
            enc->setComputePipelineState(histogram32PSO);
            enc->setBuffer(keysBuffers[srcIdx], 0, 0);
            enc->setBuffer(histogramBuffer, 0, 1);
            enc->setBytes(&bitOffset, sizeof(uint32_t), 2);
            enc->setBytes(&numElementsU32, sizeof(uint32_t), 3);
            
            MTL::Size grid = MTL::Size(numElements, 1, 1);
            MTL::Size tg = MTL::Size(THREADGROUP_SIZE, 1, 1);
            enc->dispatchThreads(grid, tg);
            enc->endEncoding();
        }
        
        cmdBuffer->commit();
        cmdBuffer->waitUntilCompleted();
        
        // CPU prefix sum
        {
            uint32_t* hist = (uint32_t*)histogramBuffer->contents();
            uint32_t sum = 0;
            for (int i = 0; i < RADIX_SIZE; i++) {
                uint32_t count = hist[i];
                hist[i] = sum;
                sum += count;
            }
        }

        // Scatter
        cmdBuffer = queue->commandBuffer();
        {
            MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
            enc->setComputePipelineState(scatter32SimplePSO);
            enc->setBuffer(keysBuffers[srcIdx], 0, 0);
            enc->setBuffer(valuesBuffers[srcIdx], 0, 1);
            enc->setBuffer(keysBuffers[dstIdx], 0, 2);
            enc->setBuffer(valuesBuffers[dstIdx], 0, 3);
            enc->setBuffer(histogramBuffer, 0, 4);
            enc->setBuffer(digitCountersBuffer, 0, 5);
            enc->setBytes(&bitOffset, sizeof(uint32_t), 6);
            enc->setBytes(&numElementsU32, sizeof(uint32_t), 7);
            
            MTL::Size grid = MTL::Size(numElements, 1, 1);
            MTL::Size tg = MTL::Size(THREADGROUP_SIZE, 1, 1);
            enc->dispatchThreads(grid, tg);
            enc->endEncoding();
        }
        
        // Finalize command buffer
        cmdBuffer->commit();
        cmdBuffer->waitUntilCompleted();
        
        srcIdx = dstIdx;
    }
    
    currentBuffer = srcIdx;
    return valuesBuffers[srcIdx];
}


// GPURadixSort64 Implementation
GPURadixSort64::GPURadixSort64(MTL::Device* device, MTL::Library* library, size_t maxElements)
    : device(device)
    , maxElements(maxElements)
{
    createPipelines(library);
    
    // Allocate double buffers for 64-bit keys
    keysBuffers[0] = device->newBuffer(maxElements * sizeof(uint64_t),
                                       MTL::ResourceStorageModeShared);
    keysBuffers[1] = device->newBuffer(maxElements * sizeof(uint64_t),
                                       MTL::ResourceStorageModeShared);
    valuesBuffers[0] = device->newBuffer(maxElements * sizeof(uint32_t),
                                         MTL::ResourceStorageModeShared);
    valuesBuffers[1] = device->newBuffer(maxElements * sizeof(uint32_t),
                                         MTL::ResourceStorageModeShared);
    
    histogramBuffer = device->newBuffer(RADIX_SIZE * sizeof(uint32_t),
                                        MTL::ResourceStorageModeShared);
    digitCountersBuffer = device->newBuffer(RADIX_SIZE * sizeof(uint32_t),
                                            MTL::ResourceStorageModeShared);

    // Block histograms and offsets buffers for stable scatter
    // Max blocks = maxElements / THREADGROUP_SIZE
    maxBlocks = (maxElements + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
    blockHistogramsBuffer = device->newBuffer(maxBlocks * RADIX_SIZE * sizeof(uint32_t),
                                              MTL::ResourceStorageModeShared);
    blockOffsetsBuffer = device->newBuffer(maxBlocks * RADIX_SIZE * sizeof(uint32_t),
                                           MTL::ResourceStorageModeShared);
}

// Destructor
GPURadixSort64::~GPURadixSort64() {
    if (keysBuffers[0]) keysBuffers[0]->release();
    if (keysBuffers[1]) keysBuffers[1]->release();
    if (valuesBuffers[0]) valuesBuffers[0]->release();
    if (valuesBuffers[1]) valuesBuffers[1]->release();
    if (histogramBuffer) histogramBuffer->release();
    if (digitCountersBuffer) digitCountersBuffer->release();
    if (blockHistogramsBuffer) blockHistogramsBuffer->release();
    if (blockOffsetsBuffer) blockOffsetsBuffer->release();

    if (histogram64PSO) histogram64PSO->release();
    if (prefixSum256PSO) prefixSum256PSO->release();
    if (prefixSum256KernelPSO) prefixSum256KernelPSO->release();
    if (computeBlockOffsetsGPUPSO) computeBlockOffsetsGPUPSO->release();
    if (scatter64StablePSO) scatter64StablePSO->release();
    if (scatter64WithAtomicRankPSO) scatter64WithAtomicRankPSO->release();
    if (scatter64OptimizedPSO) scatter64OptimizedPSO->release();
    if (clearHistogramPSO) clearHistogramPSO->release();
    if (computeBlockHistogramsPSO) computeBlockHistogramsPSO->release();
    if (scatter64BlockStablePSO) scatter64BlockStablePSO->release();
}

// Create compute pipelines
void GPURadixSort64::createPipelines(MTL::Library* library) {
    histogram64PSO = createPipeline(device, library, "histogram64");
    histogram64CombinedPSO = createPipeline(device, library, "histogram64Combined");
    prefixSum256PSO = createPipeline(device, library, "prefixSum256");
    prefixSum256KernelPSO = createPipeline(device, library, "prefixSum256Kernel");
    computeBlockOffsetsGPUPSO = createPipeline(device, library, "computeBlockOffsetsGPU");
    scatter64StablePSO = createPipeline(device, library, "scatter64Stable");
    scatter64WithAtomicRankPSO = createPipeline(device, library, "scatter64WithAtomicRank");
    scatter64OptimizedPSO = createPipeline(device, library, "scatter64Optimized");
    clearHistogramPSO = createPipeline(device, library, "clearHistogram");
    computeBlockHistogramsPSO = createPipeline(device, library, "computeBlockHistograms64");
    scatter64BlockStablePSO = createPipeline(device, library, "scatter64BlockStable");
}

// Ensure capacity of buffers
void GPURadixSort64::ensureCapacity(size_t numElements) {
    if (numElements <= maxElements) return;

    // Double buffer size until sufficient
    maxElements = std::max(numElements, maxElements * 2);

    if (keysBuffers[0]) keysBuffers[0]->release();
    if (keysBuffers[1]) keysBuffers[1]->release();
    if (valuesBuffers[0]) valuesBuffers[0]->release();
    if (valuesBuffers[1]) valuesBuffers[1]->release();
    if (blockHistogramsBuffer) blockHistogramsBuffer->release();
    if (blockOffsetsBuffer) blockOffsetsBuffer->release();

    keysBuffers[0] = device->newBuffer(maxElements * sizeof(uint64_t),
                                       MTL::ResourceStorageModeShared);
    keysBuffers[1] = device->newBuffer(maxElements * sizeof(uint64_t),
                                       MTL::ResourceStorageModeShared);
    valuesBuffers[0] = device->newBuffer(maxElements * sizeof(uint32_t),
                                         MTL::ResourceStorageModeShared);
    valuesBuffers[1] = device->newBuffer(maxElements * sizeof(uint32_t),
                                         MTL::ResourceStorageModeShared);

    // Reallocate block histograms and offsets buffers
    maxBlocks = (maxElements + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
    blockHistogramsBuffer = device->newBuffer(maxBlocks * RADIX_SIZE * sizeof(uint32_t),
                                              MTL::ResourceStorageModeShared);
    blockOffsetsBuffer = device->newBuffer(maxBlocks * RADIX_SIZE * sizeof(uint32_t),
                                           MTL::ResourceStorageModeShared);
}

void GPURadixSort64::sort(MTL::CommandQueue* queue,
                           MTL::Buffer* keysIn,
                           MTL::Buffer* valuesIn,
                           size_t numElements) {
    if (numElements == 0) return;

    // Verify pipelines exist
    if (!histogram64PSO || !prefixSum256KernelPSO || !scatter64WithAtomicRankPSO || !clearHistogramPSO) {
        std::cerr << "ERROR: GPURadixSort64 pipelines not initialized!" << std::endl;
        return;
    }

    ensureCapacity(numElements);

    uint32_t numElementsU32 = (uint32_t)numElements;
    uint32_t numBlocksU32 = (uint32_t)((numElements + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE);
    MTL::Size grid = MTL::Size(numElements, 1, 1);
    MTL::Size tg = MTL::Size(THREADGROUP_SIZE, 1, 1);

    int srcIdx = 0;

    // Single command buffer for blit + all 8 passes (eliminates extra GPU-CPU sync)
    MTL::CommandBuffer* cmdBuffer = queue->commandBuffer();

    // Copy input data as part of the same command buffer
    MTL::BlitCommandEncoder* initialBlit = cmdBuffer->blitCommandEncoder();
    initialBlit->copyFromBuffer(keysIn, 0, keysBuffers[0], 0, numElements * sizeof(uint64_t));
    initialBlit->copyFromBuffer(valuesIn, 0, valuesBuffers[0], 0, numElements * sizeof(uint32_t));
    initialBlit->endEncoding();

    for (uint32_t pass = 0; pass < NUM_PASSES; pass++) {
        uint32_t bitOffset = pass * 8;
        int dstIdx = 1 - srcIdx;

        // GPU clear histogram
        MTL::BlitCommandEncoder* blit = cmdBuffer->blitCommandEncoder();
        blit->fillBuffer(histogramBuffer, NS::Range(0, RADIX_SIZE * sizeof(uint32_t)), 0);
        blit->endEncoding();

        MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();

        // Combined kernel: builds both global and per-block histograms in one pass
        enc->setComputePipelineState(histogram64CombinedPSO);
        enc->setBuffer(keysBuffers[srcIdx], 0, 0);
        enc->setBuffer(histogramBuffer, 0, 1);
        enc->setBuffer(blockHistogramsBuffer, 0, 2);
        enc->setBytes(&bitOffset, sizeof(uint32_t), 3);
        enc->setBytes(&numElementsU32, sizeof(uint32_t), 4);
        enc->dispatchThreads(grid, tg);

        // GPU prefix sum on histogram
        enc->setComputePipelineState(prefixSum256KernelPSO);
        enc->setBuffer(histogramBuffer, 0, 0);
        enc->dispatchThreads(MTL::Size(256, 1, 1), MTL::Size(256, 1, 1));

        enc->endEncoding();

        // Compute block offsets - single threadgroup, 256 threads (one per digit)
        enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(computeBlockOffsetsGPUPSO);
        enc->setBuffer(blockHistogramsBuffer, 0, 0);
        enc->setBuffer(blockOffsetsBuffer, 0, 1);
        enc->setBuffer(histogramBuffer, 0, 2);
        enc->setBytes(&numBlocksU32, sizeof(uint32_t), 3);
        enc->dispatchThreads(MTL::Size(256, 1, 1), MTL::Size(256, 1, 1));
        enc->endEncoding();

        // Parallel scatter with threadgroup-local atomics
        enc = cmdBuffer->computeCommandEncoder();
        enc->setComputePipelineState(scatter64BlockStablePSO);
        enc->setBuffer(keysBuffers[srcIdx], 0, 0);
        enc->setBuffer(valuesBuffers[srcIdx], 0, 1);
        enc->setBuffer(keysBuffers[dstIdx], 0, 2);
        enc->setBuffer(valuesBuffers[dstIdx], 0, 3);
        enc->setBuffer(blockOffsetsBuffer, 0, 4);
        enc->setBytes(&bitOffset, sizeof(uint32_t), 5);
        enc->setBytes(&numElementsU32, sizeof(uint32_t), 6);
        enc->dispatchThreads(grid, tg);
        enc->endEncoding();

        srcIdx = dstIdx;
    }

    // Single wait at the end
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();

    currentBuffer = srcIdx;
}

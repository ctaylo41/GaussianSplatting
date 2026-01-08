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
    
    // Allocate double buffers - initialize to zero to avoid undefined behavior
    keysBuffers[0] = device->newBuffer(maxElements * sizeof(uint32_t),
                                       MTL::ResourceStorageModeShared);
    keysBuffers[1] = device->newBuffer(maxElements * sizeof(uint32_t),
                                       MTL::ResourceStorageModeShared);
    valuesBuffers[0] = device->newBuffer(maxElements * sizeof(uint32_t),
                                         MTL::ResourceStorageModeShared);
    valuesBuffers[1] = device->newBuffer(maxElements * sizeof(uint32_t),
                                         MTL::ResourceStorageModeShared);
    
    // Zero-initialize the value buffers with identity permutation
    // This ensures that even if sorting fails, we have valid indices
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
    
    // Print first call info
    static bool firstCall = true;
    if (firstCall) {
        std::cout << "GPU Sort first call: numElements=" << numElements
                  << ", cameraPos=(" << cameraPos.x << "," << cameraPos.y << "," << cameraPos.z << ")" << std::endl;
        firstCall = false;
    }
    
    MTL::CommandBuffer* cmdBuffer = queue->commandBuffer();
    
    // Step 1: Compute depths and initialize keys/values
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
    
    // Verify computeDepths output on first few calls
    static int debugCounter = 0;
    if (debugCounter < 3) {
        uint32_t* keys = (uint32_t*)keysBuffers[0]->contents();
        uint32_t* vals = (uint32_t*)valuesBuffers[0]->contents();
        std::cout << "After computeDepths (call " << debugCounter << "):" << std::endl;
        std::cout << "  First 5 keys: ";
        for (int i = 0; i < std::min((size_t)5, numElements); i++) {
            std::cout << keys[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "  First 5 values: ";
        for (int i = 0; i < std::min((size_t)5, numElements); i++) {
            std::cout << vals[i] << " ";
        }
        std::cout << std::endl;
        
        // Verify all values are valid indices
        bool allValid = true;
        for (size_t i = 0; i < numElements; i++) {
            if (vals[i] >= numElements) {
                std::cout << "  ERROR: Invalid value at index " << i << ": " << vals[i] << std::endl;
                allValid = false;
                break;
            }
        }
        if (allValid) {
            std::cout << "  All values are valid indices." << std::endl;
        }
    }
    
    // 4 passes of radix sort (8 bits per pass)
    int srcIdx = 0;
    
    for (uint32_t pass = 0; pass < NUM_PASSES; pass++) {
        uint32_t bitOffset = pass * 8;
        int dstIdx = 1 - srcIdx;
        
        // Clear histogram using CPU memset for guaranteed zeroing
        // The GPU clear was potentially racing with histogram32
        memset(histogramBuffer->contents(), 0, RADIX_SIZE * sizeof(uint32_t));
        
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
        
        // CPU prefix sum (simple and correct)
        {
            uint32_t* hist = (uint32_t*)histogramBuffer->contents();
            
            // Verify histogram
            if (debugCounter < 3 && pass == 0) {
                uint32_t totalCount = 0;
                for (int i = 0; i < RADIX_SIZE; i++) {
                    totalCount += hist[i];
                }
                std::cout << "Pass " << pass << " histogram total: " << totalCount
                          << " (expected " << numElements << ")" << std::endl;
                if (totalCount != numElements) {
                    std::cout << "  ERROR: Histogram count mismatch!" << std::endl;
                }
            }
            
            uint32_t sum = 0;
            for (int i = 0; i < RADIX_SIZE; i++) {
                uint32_t count = hist[i];
                hist[i] = sum;
                sum += count;
            }
            
            // Verify prefix sum
            if (debugCounter < 3 && pass == 0) {
                std::cout << "Pass " << pass << " prefix sum final: " << sum
                          << " (expected " << numElements << ")" << std::endl;
            }
        }
        
        // For StorageModeShared, CPU writes are immediately visible to GPU
        
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
            enc->setBytes(&bitOffset, sizeof(uint32_t), 5);
            enc->setBytes(&numElementsU32, sizeof(uint32_t), 6);
            
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
    
    // Verify final output
    if (debugCounter < 3) {
        uint32_t* sortedVals = (uint32_t*)valuesBuffers[srcIdx]->contents();
        std::cout << "After sort:" << std::endl;
        std::cout << "  First 5 sorted indices: ";
        for (int i = 0; i < std::min((size_t)5, numElements); i++) {
            std::cout << sortedVals[i] << " ";
        }
        std::cout << std::endl;
        
        // Verify all sorted indices are valid
        bool allValid = true;
        int invalidCount = 0;
        for (size_t i = 0; i < numElements; i++) {
            if (sortedVals[i] >= numElements) {
                if (invalidCount < 5) {
                    std::cout << "  ERROR: Invalid sorted index at " << i << ": " << sortedVals[i] << std::endl;
                }
                invalidCount++;
                allValid = false;
            }
        }
        if (!allValid) {
            std::cout << "  Total invalid indices: " << invalidCount << " / " << numElements << std::endl;
            // Return identity permutation if sort produced garbage
            std::cout << "  FALLBACK: Returning identity permutation" << std::endl;
            for (size_t i = 0; i < numElements; i++) {
                sortedVals[i] = (uint32_t)i;
            }
        } else {
            std::cout << "  All sorted indices valid." << std::endl;
        }
        
        debugCounter++;
    }
    
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
}

// Destructor
GPURadixSort64::~GPURadixSort64() {
    if (keysBuffers[0]) keysBuffers[0]->release();
    if (keysBuffers[1]) keysBuffers[1]->release();
    if (valuesBuffers[0]) valuesBuffers[0]->release();
    if (valuesBuffers[1]) valuesBuffers[1]->release();
    if (histogramBuffer) histogramBuffer->release();
    if (digitCountersBuffer) digitCountersBuffer->release();
    
    if (histogram64PSO) histogram64PSO->release();
    if (prefixSum256PSO) prefixSum256PSO->release();
    if (scatter64SimplePSO) scatter64SimplePSO->release();
    if (scatter64OptimizedPSO) scatter64OptimizedPSO->release();
    if (clearHistogramPSO) clearHistogramPSO->release();
}

// Create compute pipelines
void GPURadixSort64::createPipelines(MTL::Library* library) {
    histogram64PSO = createPipeline(device, library, "histogram64");
    prefixSum256PSO = createPipeline(device, library, "prefixSum256");
    scatter64SimplePSO = createPipeline(device, library, "scatter64Simple");
    scatter64OptimizedPSO = createPipeline(device, library, "scatter64Optimized");
    clearHistogramPSO = createPipeline(device, library, "clearHistogram");
}

// Ensure capacity of buffers
void GPURadixSort64::ensureCapacity(size_t numElements) {
    if (numElements <= maxElements) return;
    
    maxElements = std::max(numElements, maxElements * 2);
    
    if (keysBuffers[0]) keysBuffers[0]->release();
    if (keysBuffers[1]) keysBuffers[1]->release();
    if (valuesBuffers[0]) valuesBuffers[0]->release();
    if (valuesBuffers[1]) valuesBuffers[1]->release();
    
    keysBuffers[0] = device->newBuffer(maxElements * sizeof(uint64_t),
                                       MTL::ResourceStorageModeShared);
    keysBuffers[1] = device->newBuffer(maxElements * sizeof(uint64_t),
                                       MTL::ResourceStorageModeShared);
    valuesBuffers[0] = device->newBuffer(maxElements * sizeof(uint32_t),
                                         MTL::ResourceStorageModeShared);
    valuesBuffers[1] = device->newBuffer(maxElements * sizeof(uint32_t),
                                         MTL::ResourceStorageModeShared);
}

// Main sort function
void GPURadixSort64::sort(MTL::CommandQueue* queue,
                           MTL::Buffer* keysIn,
                           MTL::Buffer* valuesIn,
                           size_t numElements) {
    if (numElements == 0) return;
    
    ensureCapacity(numElements);
    
    uint32_t numElementsU32 = (uint32_t)numElements;
    
    // Copy input to our buffers
    memcpy(keysBuffers[0]->contents(), keysIn->contents(), numElements * sizeof(uint64_t));
    memcpy(valuesBuffers[0]->contents(), valuesIn->contents(), numElements * sizeof(uint32_t));
    
    int srcIdx = 0;
    
    // 8 passes for 64-bit keys
    for (uint32_t pass = 0; pass < NUM_PASSES; pass++) {
        uint32_t bitOffset = pass * 8;
        int dstIdx = 1 - srcIdx;
        
        // Clear histogram using CPU memset
        memset(histogramBuffer->contents(), 0, RADIX_SIZE * sizeof(uint32_t));
        
        MTL::CommandBuffer* cmdBuffer = queue->commandBuffer();
        
        // Build histogram
        {
            MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
            enc->setComputePipelineState(histogram64PSO);
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
            enc->setComputePipelineState(scatter64SimplePSO);
            enc->setBuffer(keysBuffers[srcIdx], 0, 0);
            enc->setBuffer(valuesBuffers[srcIdx], 0, 1);
            enc->setBuffer(keysBuffers[dstIdx], 0, 2);
            enc->setBuffer(valuesBuffers[dstIdx], 0, 3);
            enc->setBuffer(histogramBuffer, 0, 4);
            enc->setBytes(&bitOffset, sizeof(uint32_t), 5);
            enc->setBytes(&numElementsU32, sizeof(uint32_t), 6);
            
            MTL::Size grid = MTL::Size(numElements, 1, 1);
            MTL::Size tg = MTL::Size(THREADGROUP_SIZE, 1, 1);
            enc->dispatchThreads(grid, tg);
            enc->endEncoding();
        }
        
        cmdBuffer->commit();
        cmdBuffer->waitUntilCompleted();
        
        srcIdx = dstIdx;
    }
    
    currentBuffer = srcIdx;
}

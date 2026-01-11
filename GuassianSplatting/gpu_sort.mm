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
            enc->setBuffer(histogramBuffer, 0, 4);  // prefixSums (read-only)
            enc->setBuffer(digitCountersBuffer, 0, 5);  // digitCounters (atomic)
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
    // REMOVED: localRanksBuffer - no longer needed with atomic scatter
}

// Destructor
GPURadixSort64::~GPURadixSort64() {
    if (keysBuffers[0]) keysBuffers[0]->release();
    if (keysBuffers[1]) keysBuffers[1]->release();
    if (valuesBuffers[0]) valuesBuffers[0]->release();
    if (valuesBuffers[1]) valuesBuffers[1]->release();
    if (histogramBuffer) histogramBuffer->release();
    if (digitCountersBuffer) digitCountersBuffer->release();
    // REMOVED: localRanksBuffer - no longer needed
    
    if (histogram64PSO) histogram64PSO->release();
    if (prefixSum256PSO) prefixSum256PSO->release();
    if (scatter64WithAtomicRankPSO) scatter64WithAtomicRankPSO->release();
    if (scatter64OptimizedPSO) scatter64OptimizedPSO->release();
    if (clearHistogramPSO) clearHistogramPSO->release();
}

// Create compute pipelines
void GPURadixSort64::createPipelines(MTL::Library* library) {
    histogram64PSO = createPipeline(device, library, "histogram64");
    prefixSum256PSO = createPipeline(device, library, "prefixSum256");
    scatter64WithAtomicRankPSO = createPipeline(device, library, "scatter64WithAtomicRank");
    scatter64OptimizedPSO = createPipeline(device, library, "scatter64Optimized");
    clearHistogramPSO = createPipeline(device, library, "clearHistogram");
    // REMOVED: computeLocalRanks64PSO - no longer needed with atomic scatter
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
    
    // REMOVED: localRanksBuffer - no longer needed with atomic scatter
}

void GPURadixSort64::sort(MTL::CommandQueue* queue,
                           MTL::Buffer* keysIn,
                           MTL::Buffer* valuesIn,
                           size_t numElements) {
    if (numElements == 0) return;
    
    // Verify pipelines exist
    if (!histogram64PSO || !scatter64WithAtomicRankPSO || !clearHistogramPSO) {
        std::cerr << "ERROR: GPURadixSort64 pipelines not initialized!" << std::endl;
        std::cerr << "  histogram64PSO: " << (histogram64PSO ? "OK" : "NULL") << std::endl;
        std::cerr << "  scatter64WithAtomicRankPSO: " << (scatter64WithAtomicRankPSO ? "OK" : "NULL") << std::endl;
        std::cerr << "  clearHistogramPSO: " << (clearHistogramPSO ? "OK" : "NULL") << std::endl;
        return;
    }
    
    ensureCapacity(numElements);
    
    uint32_t numElementsU32 = (uint32_t)numElements;
    
    // Copy input to our buffers
    memcpy(keysBuffers[0]->contents(), keysIn->contents(), numElements * sizeof(uint64_t));
    memcpy(valuesBuffers[0]->contents(), valuesIn->contents(), numElements * sizeof(uint32_t));
    
    static bool firstRun = true;
    if (firstRun) {
        std::cout << "GPURadixSort64 first run: numElements=" << numElements << std::endl;
        firstRun = false;
    }
    
    int srcIdx = 0;
    
    // 8 passes for 64-bit keys
    for (uint32_t pass = 0; pass < NUM_PASSES; pass++) {
        uint32_t bitOffset = pass * 8;
        int dstIdx = 1 - srcIdx;
        
        static int debugCallCount = 0;
        bool doDebug = (debugCallCount < 1);
        
        if (doDebug) {
            std::cout << "Pass " << pass << ": bitOffset=" << bitOffset 
                      << " src=" << srcIdx << " dst=" << dstIdx << std::endl;
        }
        
        // Clear histogram using CPU memset (reliable with SharedStorage)
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
        
        // CPU prefix sum (in shared memory, CPU writes are immediately visible)
        {
            uint32_t* hist = (uint32_t*)histogramBuffer->contents();
            
            // Verify histogram BEFORE prefix sum
            if (doDebug && pass < 2) {
                uint32_t totalHist = 0;
                for (int i = 0; i < RADIX_SIZE; i++) {
                    totalHist += hist[i];
                }
                std::cout << "  Pass " << pass << " histogram sum: " << totalHist << " (expected " << numElements << ")";
                if (totalHist == numElements) {
                    std::cout << " ✓" << std::endl;
                } else {
                    std::cout << " ✗ MISMATCH!" << std::endl;
                }
            }
            
            uint32_t sum = 0;
            for (int i = 0; i < RADIX_SIZE; i++) {
                uint32_t count = hist[i];
                hist[i] = sum;
                sum += count;
            }
        }
        
        // Clear digit counters using CPU memset (reliable with SharedStorage)
        memset(digitCountersBuffer->contents(), 0, RADIX_SIZE * sizeof(uint32_t));
        
        cmdBuffer = queue->commandBuffer();
        
        // Scatter using atomic-based kernel (O(1) per element)
        {
            MTL::ComputeCommandEncoder* enc = cmdBuffer->computeCommandEncoder();
            enc->setComputePipelineState(scatter64WithAtomicRankPSO);
            enc->setBuffer(keysBuffers[srcIdx], 0, 0);
            enc->setBuffer(valuesBuffers[srcIdx], 0, 1);
            enc->setBuffer(keysBuffers[dstIdx], 0, 2);
            enc->setBuffer(valuesBuffers[dstIdx], 0, 3);
            enc->setBuffer(histogramBuffer, 0, 4);  // prefixSums
            enc->setBuffer(digitCountersBuffer, 0, 5);  // atomic counters
            enc->setBytes(&bitOffset, sizeof(uint32_t), 6);
            enc->setBytes(&numElementsU32, sizeof(uint32_t), 7);
            
            MTL::Size grid = MTL::Size(numElements, 1, 1);
            MTL::Size tg = MTL::Size(THREADGROUP_SIZE, 1, 1);
            enc->dispatchThreads(grid, tg);
            enc->endEncoding();
        }
        
        cmdBuffer->commit();
        cmdBuffer->waitUntilCompleted();
        
        // Debug: Verify each pass
        if (doDebug) {
            uint64_t* dst = (uint64_t*)keysBuffers[dstIdx]->contents();
            uint64_t* src = (uint64_t*)keysBuffers[srcIdx]->contents();
            
            // Check if data changed
            bool dataChanged = (dst[0] != src[0]);
            std::cout << "  Pass " << pass << " result: first key changed from 0x" << std::hex << src[0] 
                      << " to 0x" << dst[0] << std::dec;
            if (dataChanged) {
                std::cout << " ✓" << std::endl;
            } else {
                std::cout << " ✗ UNCHANGED!" << std::endl;
            }
        }
        
        if (doDebug && pass == NUM_PASSES - 1) {
            std::cout << "After all " << NUM_PASSES << " passes, final sorted data is in buffer " << dstIdx << std::endl;
            debugCallCount++;
        }
        
        srcIdx = dstIdx;
    }
    
    currentBuffer = srcIdx;
    
    // CRITICAL: Check if final output differs from original input
    static int finalCheckCount = 0;
    if (finalCheckCount < 1) {
        uint64_t* finalKeys = (uint64_t*)keysBuffers[currentBuffer]->contents();
        uint64_t* inputKeys = (uint64_t*)keysIn->contents();
        
        std::cout << "\n=== Final Buffer Check ===" << std::endl;
        std::cout << "currentBuffer = " << currentBuffer << std::endl;
        std::cout << "First input key:  0x" << std::hex << inputKeys[0] << std::dec << std::endl;
        std::cout << "First output key: 0x" << std::hex << finalKeys[0] << std::dec << std::endl;
        
        if (finalKeys[0] == inputKeys[0] && finalKeys[1] == inputKeys[1] && finalKeys[2] == inputKeys[2]) {
            std::cout << "ERROR: Final output IDENTICAL to input! Sort did nothing!" << std::endl;
        }
        std::cout << "=======================" << std::endl;
        
        finalCheckCount++;
    }
    
    // Debug: Verify sort output
    static int debugCount = 0;
    if (debugCount < 2) {
        uint64_t* sortedKeys = (uint64_t*)keysBuffers[srcIdx]->contents();
        uint32_t* sortedVals = (uint32_t*)valuesBuffers[srcIdx]->contents();
        uint64_t* inputKeys = (uint64_t*)keysBuffers[0]->contents();
        
        std::cout << "\n=== GPURadixSort64 Debug (call " << debugCount << ") ===" << std::endl;
        std::cout << "Input count: " << numElements << std::endl;
        
        // Show first 5 input keys
        std::cout << "First 5 INPUT keys:" << std::endl;
        for (size_t i = 0; i < std::min((size_t)5, numElements); i++) {
            uint64_t key = inputKeys[i];
            uint32_t tileID = (uint32_t)(key >> 32);
            uint32_t depth = (uint32_t)(key & 0xFFFFFFFF);
            char buf[32];
            snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)key);
            std::cout << "  [" << i << "] 0x" << buf << " -> tile=" << tileID << " depth=0x" << std::hex << depth << std::dec << std::endl;
        }
        
        // Show first 5 output keys  
        std::cout << "First 5 OUTPUT keys:" << std::endl;
        for (size_t i = 0; i < std::min((size_t)5, numElements); i++) {
            uint64_t key = sortedKeys[i];
            uint32_t tileID = (uint32_t)(key >> 32);
            uint32_t depth = (uint32_t)(key & 0xFFFFFFFF);
            char buf[32];
            snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)key);
            std::cout << "  [" << i << "] 0x" << buf << " -> tile=" << tileID << " depth=0x" << std::hex << depth << std::dec << std::endl;
        }
        
        // Verify monotonically increasing
        bool isSorted = true;
        size_t firstBadIdx = 0;
        for (size_t i = 1; i < numElements; i++) {
            if (sortedKeys[i] < sortedKeys[i-1]) {
                char buf1[32], buf2[32];
                snprintf(buf1, sizeof(buf1), "%016llx", (unsigned long long)sortedKeys[i-1]);
                snprintf(buf2, sizeof(buf2), "%016llx", (unsigned long long)sortedKeys[i]);
                std::cout << "ERROR: Not sorted at index " << i << ": 0x" << buf1 << " > 0x" << buf2 << std::endl;
                isSorted = false;
                firstBadIdx = i;
                break;
            }
        }
        if (isSorted) {
            std::cout << "✓ All " << numElements << " keys correctly sorted" << std::endl;
        } else {
            std::cout << "✗ Sort FAILED at index " << firstBadIdx << std::endl;
        }
        std::cout << "=== End Debug ===" << std::endl;
        
        debugCount++;
    }
}

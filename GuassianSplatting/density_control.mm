//
//  density_control.mm
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-28.
//

#include "density_control.hpp"
#include <iostream>
#include "gradients.hpp"

DensityController::DensityController(MTL::Device* device, MTL::Library* library):device(device),maxGaussians(2000000) {
    NS::Error* error = nullptr;
    
    MTL::Function* markFunc = library->newFunction(NS::String::string("markGaussians", NS::ASCIIStringEncoding));
    markPSO = device->newComputePipelineState(markFunc, &error);
    markFunc->release();
    
    MTL::Function* splitCloneFunc = library->newFunction(NS::String::string("splitCloneGaussians", NS::ASCIIStringEncoding));
    splitClonePSO = device->newComputePipelineState(splitCloneFunc, &error);
    splitCloneFunc->release();
    
    gradientAccum = device->newBuffer(maxGaussians * sizeof(float), MTL::ResourceStorageModeShared);
    gradientCount = device->newBuffer(maxGaussians * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    markerBuffer = device->newBuffer(maxGaussians * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    prefixSumBuffer = device->newBuffer(maxGaussians * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
}

DensityController::~DensityController() {
    gradientAccum->release();
    gradientCount->release();
    markerBuffer->release();
    prefixSumBuffer->release();
    markPSO->release();
    splitClonePSO->release();
}

void DensityController::resetAccumulator(size_t gaussianCount) {
    memset(gradientAccum->contents(), 0, gaussianCount * sizeof(float));
    memset(gradientCount->contents(), 0, gaussianCount * sizeof(uint32_t));
}

void DensityController::accumulateGradients(MTL::CommandQueue *queue, MTL::Buffer *gradients, size_t gaussianCount) {
    GaussianGradients* grads = (GaussianGradients*)gradients->contents();
    float* accum = (float*)gradientAccum->contents();
    uint32_t* count = (uint32_t*)gradientCount->contents();
    
    for(size_t i=0;i<gaussianCount;i++) {
        float gradMag = simd_length(grads[i].position);
        accum[i] += gradMag;
        count[i]++;
    }
}

DensityStats DensityController::apply(MTL::CommandQueue* queue,
                                      MTL::Buffer*& gaussianBuffer,
                                      MTL::Buffer*& positionBuffer,
                                      MTL::Buffer* gradientAccumBuffer,
                                      size_t& gaussianCount,
                                      size_t iteration,
                                      float gradThreshold,
                                      float minOpacity,
                                      float maxScale) {
    DensityStats stats = {0,0,0};
    
    Gaussian* gaussians = (Gaussian*)gaussianBuffer->contents();
    float* accumGrad = (float*)gradientAccum->contents();
    uint32_t* counts = (uint32_t*)gradientCount->contents();
    uint32_t* markers = (uint32_t*)markerBuffer->contents();
    
    for(size_t i=0;i<gaussianCount;i++) {
        Gaussian& g = gaussians[i];
        float avgGrad = counts[i] > 0 ? accumGrad[i] / counts[i] : 0;
        float opacity = 1.0f / (1.0f + exp(-g.opacity));
        float maxScaleVal = fmax(fmax(exp(g.scale.x), exp(g.scale.y)), exp(g.scale.z));
        
        if (opacity < minOpacity || maxScaleVal > maxScale) {
            markers[i] = 1;
            stats.numPruned++;
        } else if(avgGrad > gradThreshold) {
            if(maxScaleVal > 0.01f) {
                markers[i] = 3;
                stats.numSplit++;
            } else {
                markers[i] = 2;
                stats.numCloned++;
            }
        } else {
            markers[i] = 0;
        }
    }
    
    size_t newCount = gaussianCount - stats.numPruned + stats.numSplit + stats.numSplit;
    
    if(newCount > maxGaussians) {
        std::cout << "Warning would exceed max gaussians skipping density control" << std::endl;
        resetAccumulator(gaussianCount);
        return {0, 0, 0};
    }
    
    MTL::Buffer* newGaussianBuffer = device->newBuffer(newCount * sizeof(Gaussian), MTL::ResourceStorageModeShared);
    MTL::Buffer* newPositionsBuffer = device->newBuffer(newCount * sizeof(simd_float3), MTL::ResourceStorageModeShared);
    
    Gaussian* newGaussians = (Gaussian*)newGaussianBuffer->contents();
    simd_float3* newPositions = (simd_float3*)newPositionsBuffer->contents();
    
    size_t writeIdx = 0;
    for(size_t i=0;i<gaussianCount;i++) {
        Gaussian& g = gaussians[i];
        uint32_t marker = markers[i];
        
        if(marker==1) {
            continue;
        }
        
        if(marker==0) {
            newGaussians[writeIdx] = g;
            newPositions[writeIdx] = g.position;
            writeIdx++;
        } else if(marker==2) {
            newGaussians[writeIdx] = g;
            newPositions[writeIdx] = g.position;
            writeIdx++;
            
            Gaussian cloned = g;
            float perturbScale = 0.01f;
            cloned.position.x += perturbScale * ((float)rand() / RAND_MAX - 0.5f);
            cloned.position.y += perturbScale * ((float)rand() / RAND_MAX - 0.5f);
            cloned.position.z += perturbScale * ((float)rand() / RAND_MAX - 0.5f);
            newGaussians[writeIdx] = cloned;
            newPositions[writeIdx] = cloned.position;
            writeIdx++;
        } else if(marker == 3) {
            float scaleFactor = 0.6f;
            
            simd_float3 scale = simd_make_float3(exp(g.scale.x), exp(g.scale.y), exp(g.scale.z));
            simd_float3 offset;
            if(scale.x >= scale.y && scale.x >= scale.z) {
                offset = simd_make_float3(scale.x * 0.5f, 0, 0);
            } else if (scale.y >= scale.z) {
                offset = simd_make_float3(0, scale.y * 0.5f, 0);
            } else {
                offset = simd_make_float3(0,0, scale.z * 0.5f);
            }
            
            simd_float4 q = g.rotation;
            float r = q.x, x = q.y, y = q.z, z = q.w;
            simd_float3x3 R = {{
                            {1 - 2*(y*y + z*z), 2*(x*y + r*z), 2*(x*z - r*y)},
                            {2*(x*y - r*z), 1 - 2*(x*x + z*z), 2*(y*z + r*x)},
                            {2*(x*z + r*y), 2*(y*z - r*x), 1 - 2*(x*x + y*y)}
                        }};
            offset = simd_mul(R, offset);
            
            Gaussian child1 = g;
            child1.position = g.position + offset;
            child1.scale = simd_make_float3(g.scale.x + log(scaleFactor),
                                            g.scale.y + log(scaleFactor),
                                            g.scale.z + log(scaleFactor));
            
            newGaussians[writeIdx] = child1;
            newPositions[writeIdx] = child1.position;
            writeIdx++;
            
            Gaussian child2 = g;
            child2.position = g.position - offset;
            child2.scale = child1.scale;
            newGaussians[writeIdx] = child2;
            newPositions[writeIdx] = child2.position;
            writeIdx++;
            
        }
    }
    
    gaussianBuffer->release();
    positionBuffer->release();
    gaussianBuffer = newGaussianBuffer;
    positionBuffer = newPositionsBuffer;
    gaussianCount = writeIdx;
        
    // Reset accumulator for new count
    resetAccumulator(gaussianCount);
        
    std::cout << "Density control: pruned=" << stats.numPruned << " cloned=" << stats.numCloned << " split=" << stats.numSplit << " total=" << gaussianCount << std::endl;
        
    return stats;
}

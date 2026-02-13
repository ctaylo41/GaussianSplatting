//
//  optimizer.hpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-27.
//

#pragma once
#include <Metal/Metal.hpp>
#include <simd/simd.h>
#include <vector>
#include <utility>

// Adam optimizer parameters structure
struct AdamParams {
    float lr;
    float beta1;
    float beta2;
    float epsilon;
    uint32_t t;
    uint32_t numGaussians;
};

// Adam Optimizer class for updating Gaussian parameters
class AdamOptimizer {
public:
    // Constructor and destructor
    AdamOptimizer(MTL::Device* device, MTL::Library* library, size_t numGaussians);
    ~AdamOptimizer();
    
    void step(MTL::CommandQueue* queue,
              MTL::Buffer* gausians,
              MTL::Buffer* gradients,
              float lr_position = 0.00016f,
              float lr_scale = 0.005f,
              float lr_rotation = 0.001f,
              float lr_opacity = 0.05f,
              float lr_sh = 0.0025f,
              float lr_sh_rest = 0.000125f,  
              float maxLogScaleTrain = 4.0f,
              bool wait = true);

    // Wait for the last async step to complete (call before reading Gaussian buffer)
    void waitForLastStep();

    void reset();
    
    // Resize buffers when Gaussian count increases 
    void resizeIfNeeded(size_t newNumGaussians);

    // Set actual Gaussian count
    void setGaussianCount(size_t count) { resizeIfNeeded(count); }

    // Reset opacity momentum after opacity reset
    void resetOpacityMomentum();

    // Reset position momentum after opacity reset 
    void resetPositionMomentum();

    // Reset scale momentum after opacity reset 
    void resetScaleMomentum();

    // Reset rotation momentum after opacity reset
    void resetRotationMomentum();

    // Reset SH momentum after opacity reset 
    void resetSHMomentum();

    // Reset Adam state for new Gaussians starting at startIdx 
    void resetStateForNewGaussians(size_t startIdx);

    // Reset ALL Adam momentum 
    // Must be called after density control which rearranges Gaussian indices
    void resetAllMomentum();

    // Remap momentum buffers after density control using old→new index mapping
    // Preserves momentum for surviving Gaussians, zeros momentum for new ones
    void remapMomentum(const std::vector<std::pair<size_t, size_t>>& indexMapping,
                       size_t newCount);
    
    // Debug print Adam state for first Gaussian
    void debugPrintState(int gaussianIdx = 0);
    
    // Get current timestep
    uint32_t getTimestep() const { return timestep; }
    
    // Debug accessors
    MTL::Buffer* getMOpacityBuffer() { return m_opacity; }
    MTL::Buffer* getMShBuffer() { return m_sh; }
    MTL::Buffer* getVShBuffer() { return v_sh; }
    
    // Read GPU debug buffer after step
    void printGPUDebug();

private:
    // Metal device and compute pipeline
    MTL::Device* device;
    MTL::ComputePipelineState* adamPSO;
    
    // GPU debug buffer 16 floats for debugging
    MTL::Buffer* debugBuffer;
    
    // Adam state buffers
    MTL::Buffer* m_position;
    MTL::Buffer* m_scale;
    MTL::Buffer* m_rotation;
    MTL::Buffer* m_opacity;
    MTL::Buffer* m_sh;
    
    MTL::Buffer* v_position;
    MTL::Buffer* v_scale;
    MTL::Buffer* v_rotation;
    MTL::Buffer* v_opacity;
    MTL::Buffer* v_sh;
    
    // Adam parameters buffer
    MTL::Buffer* paramsBuffer;

    size_t numGaussians;      // Actual Gaussian count
    size_t bufferCapacity;    // Allocated buffer capacity
    uint32_t timestep = 0;
    MTL::CommandBuffer* lastCmdBuffer = nullptr;  // For async step
    
    // Allocate or reallocate Adam state buffers
    void allocateBuffers(size_t count);
};

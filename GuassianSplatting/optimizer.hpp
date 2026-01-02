//
//  optimizer.hpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-27.
//

#pragma once
#include <Metal/Metal.hpp>
#include <simd/simd.h>

struct AdamParams {
    float lr;
    float beta1;
    float beta2;
    float epsilon;
    uint32_t t;
    uint32_t numGaussians;
};

class AdamOptimizer {
public:
    AdamOptimizer(MTL::Device* device, MTL::Library* library, size_t numGaussians);
    ~AdamOptimizer();
    
    void step(MTL::CommandQueue* queue,
              MTL::Buffer* gausians,
              MTL::Buffer* gradients,
              float lr_position = 0.00016f,   // Official default
              float lr_scale = 0.005f,        // Official default
              float lr_rotation = 0.001f,     // Official default
              float lr_opacity = 0.05f,       // Official default
              float lr_sh = 0.0025f);         // Official default
    
    void reset();
    
    // Resize buffers when Gaussian count increases (after density control)
    void resizeIfNeeded(size_t newNumGaussians);
    
    // Reset opacity momentum after opacity reset
    void resetOpacityMomentum();

private:
    MTL::Device* device;
    MTL::ComputePipelineState* adamPSO;
    
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
    
    MTL::Buffer* paramsBuffer;
    
    size_t numGaussians;
    uint32_t timestep = 0;
    
    void allocateBuffers(size_t count);
};

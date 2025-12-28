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
              float lr_position = 0.00016f,
              float lr_scale = 0.005f,
              float lr_rotation = 0.001f,
              float lr_opacity = 0.05f,
              float lr_sh = 0.0025f);
    
    void reset();

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
};

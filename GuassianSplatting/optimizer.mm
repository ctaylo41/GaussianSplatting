//
//  optimizer.mm
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-27.
//

#include "optimizer.hpp"
#include <iostream>

AdamOptimizer::AdamOptimizer(MTL::Device* device, MTL::Library* library, size_t numGaussians)
: device(device), numGaussians(numGaussians), timestep(0) {
    
    NS::Error* error = nullptr;
    MTL::Function* func = library->newFunction(NS::String::string("adamStep", NS::ASCIIStringEncoding));
    adamPSO = device->newComputePipelineState(func, &error);
    if(!adamPSO) {
        std::cerr << "Failed to create Adam pipeline" << std::endl;
    }
    
    func->release();
    
    size_t posSize = numGaussians * sizeof(simd_float3);
    size_t scaleSize = numGaussians * sizeof(simd_float3);
    size_t rotSize = numGaussians * sizeof(simd_float4);
    size_t opacitySize = numGaussians * sizeof(float);
    size_t shSize = numGaussians * 12 * sizeof(float);
    
    m_position = device->newBuffer(posSize, MTL::ResourceStorageModeShared);
    m_scale = device->newBuffer(scaleSize, MTL::ResourceStorageModeShared);
    m_rotation = device->newBuffer(rotSize, MTL::ResourceStorageModeShared);
    m_opacity = device->newBuffer(opacitySize, MTL::ResourceStorageModeShared);
    m_sh = device->newBuffer(shSize, MTL::ResourceStorageModeShared);
    
    v_position = device->newBuffer(posSize, MTL::ResourceStorageModeShared);
    v_scale = device->newBuffer(scaleSize, MTL::ResourceStorageModeShared);
    v_rotation = device->newBuffer(rotSize, MTL::ResourceStorageModeShared);
    v_opacity = device->newBuffer(opacitySize, MTL::ResourceStorageModeShared);
    v_sh = device->newBuffer(shSize, MTL::ResourceStorageModeShared);
    
    paramsBuffer = device->newBuffer(sizeof(AdamParams), MTL::ResourceStorageModeShared);
    
    reset();
}

AdamOptimizer::~AdamOptimizer() {
    m_position->release();
    m_scale->release();
    m_rotation->release();
    m_opacity->release();
    m_sh->release();
    v_position->release();
    v_rotation->release();
    v_scale->release();
    v_opacity->release();
    v_sh->release();
    paramsBuffer->release();
    adamPSO->release();
}

void AdamOptimizer::reset() {
    timestep = 0;
    memset(m_position->contents(), 0, m_position->length());
    memset(m_scale->contents(), 0, m_scale->length());
    memset(m_rotation->contents(), 0, m_rotation->length());
    memset(m_opacity->contents(), 0, m_opacity->length());
    memset(m_sh->contents(), 0, m_sh->length());
    memset(v_position->contents(), 0, v_position->length());
    memset(v_scale->contents(), 0, v_scale->length());
    memset(v_rotation->contents(), 0, v_rotation->length());
    memset(v_opacity->contents(), 0, v_opacity->length());
    memset(v_sh->contents(), 0, v_sh->length());
}

void AdamOptimizer::step(MTL::CommandQueue* queue,
                         MTL::Buffer* gaussians,
                         MTL::Buffer* gradients,
                         float lr_position,
                         float lr_scale,
                         float lr_rotation,
                         float lr_opacity,
                         float lr_sh) {
    timestep++;
    
    MTL::CommandBuffer* cmd = queue->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    
    enc->setComputePipelineState(adamPSO);
    enc->setBuffer(gaussians, 0, 0);
    enc->setBuffer(gradients, 0, 1);
    
    enc->setBuffer(m_position, 0, 2);
    enc->setBuffer(m_scale, 0, 3);
    enc->setBuffer(m_rotation, 0, 4);
    enc->setBuffer(m_opacity, 0, 5);
    enc->setBuffer(m_sh, 0, 6);
    enc->setBuffer(v_position, 0, 7);
    enc->setBuffer(v_scale, 0, 8);
    enc->setBuffer(v_rotation, 0, 9);
    enc->setBuffer(v_opacity, 0, 10);
    enc->setBuffer(v_sh, 0, 11);
    
    float lrs[5] = {lr_position, lr_scale, lr_rotation, lr_opacity, lr_sh};
    enc->setBytes(lrs, sizeof(lrs), 12);

    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    uint32_t params[2] = {timestep, (uint32_t)numGaussians};
    enc->setBytes(&beta1, sizeof(float), 13);
    enc->setBytes(&beta2, sizeof(float), 14);
    enc->setBytes(&epsilon, sizeof(float), 15);
    enc->setBytes(params, sizeof(params), 16);

    MTL::Size grid = MTL::Size(numGaussians, 1, 1);
    MTL::Size threadgroup = MTL::Size(64, 1, 1);
    enc->dispatchThreads(grid, threadgroup);
    
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
}

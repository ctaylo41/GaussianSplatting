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
    
    allocateBuffers(numGaussians);
    reset();
}

void AdamOptimizer::allocateBuffers(size_t count) {
    // Using 3 floats (12 bytes) per vec3, not simd_float3 (16 bytes)
    // This matches the shader's manual indexing: tid * 3 + 0/1/2
    size_t posSize = count * 3 * sizeof(float);   // 12 bytes per element
    size_t scaleSize = count * 3 * sizeof(float); // 12 bytes per element
    size_t rotSize = count * sizeof(simd_float4);
    size_t opacitySize = count * sizeof(float);
    size_t shSize = count * 12 * sizeof(float);
    
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
    
    // Debug buffer: 16 floats for GPU-side debugging
    debugBuffer = device->newBuffer(16 * sizeof(float), MTL::ResourceStorageModeShared);
    memset(debugBuffer->contents(), 0, 16 * sizeof(float));
    
    paramsBuffer = device->newBuffer(sizeof(AdamParams), MTL::ResourceStorageModeShared);
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
    debugBuffer->release();
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

void AdamOptimizer::resizeIfNeeded(size_t newNumGaussians) {
    if (newNumGaussians <= numGaussians) return;
    
    std::cout << "Resizing optimizer buffers from " << numGaussians << " to " << newNumGaussians << std::endl;
    
    auto resizeBuffer = [this](MTL::Buffer*& buf, size_t newSize) {
        MTL::Buffer* newBuf = device->newBuffer(newSize, MTL::ResourceStorageModeShared);
        // Copy existing data
        size_t copySize = std::min(buf->length(), newSize);
        memcpy(newBuf->contents(), buf->contents(), copySize);
        // Zero-initialize new space
        if (newSize > buf->length()) {
            memset((char*)newBuf->contents() + buf->length(), 0, newSize - buf->length());
        }
        buf->release();
        buf = newBuf;
    };
    
    // Using 3 floats (12 bytes) per vec3, not simd_float3 (16 bytes)
    size_t posSize = newNumGaussians * 3 * sizeof(float);
    size_t scaleSize = newNumGaussians * 3 * sizeof(float);
    size_t rotSize = newNumGaussians * sizeof(simd_float4);
    size_t opacitySize = newNumGaussians * sizeof(float);
    size_t shSize = newNumGaussians * 12 * sizeof(float);
    
    resizeBuffer(m_position, posSize);
    resizeBuffer(m_scale, scaleSize);
    resizeBuffer(m_rotation, rotSize);
    resizeBuffer(m_opacity, opacitySize);
    resizeBuffer(m_sh, shSize);
    resizeBuffer(v_position, posSize);
    resizeBuffer(v_scale, scaleSize);
    resizeBuffer(v_rotation, rotSize);
    resizeBuffer(v_opacity, opacitySize);
    resizeBuffer(v_sh, shSize);
    
    numGaussians = newNumGaussians;
}

void AdamOptimizer::resetOpacityMomentum() {
    // Reset momentum for opacity to allow fresh learning after opacity reset
    memset(m_opacity->contents(), 0, numGaussians * sizeof(float));
    memset(v_opacity->contents(), 0, numGaussians * sizeof(float));
}

void AdamOptimizer::debugPrintState(int idx) {
    // Now using float* with stride 3, not simd_float3*
    float* m_pos = (float*)m_position->contents();
    float* v_pos = (float*)v_position->contents();
    float* m_scl = (float*)m_scale->contents();
    float* v_scl = (float*)v_scale->contents();
    
    // Verify buffer pointers are valid
    printf("[Adam Debug] Buffer pointers: m_scale=%p, v_scale=%p\n", (void*)m_scale->contents(), (void*)v_scale->contents());
    printf("[Adam Debug] Buffer lengths: m_scale=%zu, v_scale=%zu (expected: %zu)\n", 
           m_scale->length(), v_scale->length(), numGaussians * 3 * sizeof(float));
    
    printf("[Adam State] timestep=%u\n", timestep);
    printf("[Adam State] m_position[%d] = (%f, %f, %f)\n", idx, m_pos[idx*3+0], m_pos[idx*3+1], m_pos[idx*3+2]);
    printf("[Adam State] v_position[%d] = (%f, %f, %f)\n", idx, v_pos[idx*3+0], v_pos[idx*3+1], v_pos[idx*3+2]);
    printf("[Adam State] m_scale[%d] = (%f, %f, %f)\n", idx, m_scl[idx*3+0], m_scl[idx*3+1], m_scl[idx*3+2]);
    printf("[Adam State] v_scale[%d] = (%f, %f, %f)\n", idx, v_scl[idx*3+0], v_scl[idx*3+1], v_scl[idx*3+2]);
    
    // Print first few raw bytes to verify we're reading correctly
    uint8_t* rawBytes = (uint8_t*)m_scale->contents();
    printf("[Adam Debug] m_scale raw bytes at idx %d offset %zu: ", idx, idx * 3 * sizeof(float));
    for (int b = 0; b < 12; b++) {  // 3 floats = 12 bytes
        printf("%02X ", rawBytes[idx * 3 * sizeof(float) + b]);
    }
    printf("\n");
    
    // Sanity check: momentum should be in reasonable range [-1, 1] with 0.5 gradient clip
    // After clipping gradients to [-0.5, 0.5], momentum can at most grow by 0.1*0.5 = 0.05 per step
    if (fabsf(m_scl[idx*3+0]) > 1.0f || fabsf(m_scl[idx*3+1]) > 1.0f || fabsf(m_scl[idx*3+2]) > 1.0f) {
        printf("[WARNING] m_scale out of expected range! Max expected ~0.5 with gradient clip.\n");
        printf("[WARNING] Actual magnitude: (%f, %f, %f)\n", fabsf(m_scl[idx*3+0]), fabsf(m_scl[idx*3+1]), fabsf(m_scl[idx*3+2]));
    }
    if (fabsf(m_pos[idx*3+0]) > 1.0f || fabsf(m_pos[idx*3+1]) > 1.0f || fabsf(m_pos[idx*3+2]) > 1.0f) {
        printf("[WARNING] m_position out of expected range! Max expected ~0.5 with gradient clip.\n");
    }
}

void AdamOptimizer::printGPUDebug() {
    float* debug = (float*)debugBuffer->contents();
    printf("\n[GPU Debug] timestep=%u\n", timestep);
    printf("[GPU Debug] raw_grad = (%.6f, %.6f, %.6f)\n", debug[0], debug[1], debug[2]);
    printf("[GPU Debug] clamped_grad = (%.6f, %.6f, %.6f)\n", debug[3], debug[4], debug[5]);
    printf("[GPU Debug] m_old = (%.6f, %.6f, %.6f)\n", debug[6], debug[7], debug[8]);
    printf("[GPU Debug] m_new = (%.6f, %.6f, %.6f)\n", debug[9], debug[10], debug[11]);
    printf("[GPU Debug] beta1=%.6f, (1-beta1)=%.6f\n", debug[12], debug[13]);
    printf("[GPU Debug] scale_old = (%.6f, %.6f, %.6f)\n", debug[14], debug[15], 0.0f);  // Only 2 values fit
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
    enc->setBuffer(debugBuffer, 0, 17);  // Debug buffer for GPU-side debugging

    MTL::Size grid = MTL::Size(numGaussians, 1, 1);
    MTL::Size threadgroup = MTL::Size(64, 1, 1);
    enc->dispatchThreads(grid, threadgroup);
    
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
}


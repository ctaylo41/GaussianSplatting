//
//  optimizer.mm
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-27.
//

#include "optimizer.hpp"
#include <iostream>

// Adam Optimizer class for updating Gaussian parameters
AdamOptimizer::AdamOptimizer(MTL::Device* device, MTL::Library* library, size_t numGaussians)
: device(device), numGaussians(numGaussians), bufferCapacity(numGaussians), timestep(0) {
    
    // Create compute pipeline for Adam optimizer
    NS::Error* error = nullptr;
    // Load Adam compute function
    MTL::Function* func = library->newFunction(NS::String::string("adamStep", NS::ASCIIStringEncoding));
    // Create compute pipeline state
    adamPSO = device->newComputePipelineState(func, &error);
    if(!adamPSO) {
        std::cerr << "Failed to create Adam pipeline" << std::endl;
    }
    
    // Release function
    func->release();
    
    // Allocate Adam state buffers
    allocateBuffers(numGaussians);
    reset();
}

// Allocate or reallocate Adam state buffers
void AdamOptimizer::allocateBuffers(size_t count) {
    size_t posSize = count * 3 * sizeof(float);   
    size_t scaleSize = count * 3 * sizeof(float);
    size_t rotSize = count * sizeof(simd_float4);
    size_t opacitySize = count * sizeof(float);
    size_t shSize = count * 12 * sizeof(float);
    
    // Allocate buffers
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
    
    // Debug buffer 16 floats for GPU-side debugging
    debugBuffer = device->newBuffer(16 * sizeof(float), MTL::ResourceStorageModeShared);
    memset(debugBuffer->contents(), 0, 16 * sizeof(float));
    
    paramsBuffer = device->newBuffer(sizeof(AdamParams), MTL::ResourceStorageModeShared);
}

// Destructor to release buffers
AdamOptimizer::~AdamOptimizer() {
    waitForLastStep();
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

// Reset Adam state buffers to zero
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

// Resize Adam state buffers if number of Gaussians exceeds capacity
void AdamOptimizer::resizeIfNeeded(size_t newNumGaussians) {
    // Update actual count
    numGaussians = newNumGaussians;

    // Only resize buffers if we exceed capacity
    if (newNumGaussians <= bufferCapacity) return;

    std::cout << "Resizing optimizer buffers from " << bufferCapacity << " to " << newNumGaussians << std::endl;

    // Helper lambda to resize a buffer
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

    // Calculate new sizes
    size_t posSize = newNumGaussians * 3 * sizeof(float);
    size_t scaleSize = newNumGaussians * 3 * sizeof(float);
    size_t rotSize = newNumGaussians * sizeof(simd_float4);
    size_t opacitySize = newNumGaussians * sizeof(float);
    size_t shSize = newNumGaussians * 12 * sizeof(float);

    // Resize all buffers
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

    bufferCapacity = newNumGaussians;
}

// Reset momentum for opacity to allow fresh learning after opacity reset
void AdamOptimizer::resetOpacityMomentum() {
    memset(m_opacity->contents(), 0, numGaussians * sizeof(float));
    memset(v_opacity->contents(), 0, numGaussians * sizeof(float));
}

// Reset position momentum after opacity reset - prevents stale momentum from causing drift
void AdamOptimizer::resetPositionMomentum() {
    memset(m_position->contents(), 0, numGaussians * 3 * sizeof(float));
    memset(v_position->contents(), 0, numGaussians * 3 * sizeof(float));
    std::cout << "Reset position momentum after opacity reset" << std::endl;
}

// Reset scale momentum after opacity reset
void AdamOptimizer::resetScaleMomentum() {
    memset(m_scale->contents(), 0, numGaussians * 3 * sizeof(float));
    memset(v_scale->contents(), 0, numGaussians * 3 * sizeof(float));
    std::cout << "Reset scale momentum after opacity reset" << std::endl;
}

// Reset rotation momentum after opacity reset
void AdamOptimizer::resetRotationMomentum() {
    simd_float4* m_rot = (simd_float4*)m_rotation->contents();
    simd_float4* v_rot = (simd_float4*)v_rotation->contents();
    for (size_t i = 0; i < numGaussians; i++) {
        m_rot[i] = simd_make_float4(0, 0, 0, 0);
        v_rot[i] = simd_make_float4(0, 0, 0, 0);
    }
    std::cout << "Reset rotation momentum after opacity reset" << std::endl;
}

// Reset SH momentum after opacity reset
void AdamOptimizer::resetSHMomentum() {
    memset(m_sh->contents(), 0, numGaussians * 12 * sizeof(float));
    memset(v_sh->contents(), 0, numGaussians * 12 * sizeof(float));
    std::cout << "Reset SH momentum after opacity reset" << std::endl;
}

// Reset ALL Adam momentum after density control (keeps timestep for bias correction)
void AdamOptimizer::resetAllMomentum() {
    memset(m_position->contents(), 0, numGaussians * 3 * sizeof(float));
    memset(v_position->contents(), 0, numGaussians * 3 * sizeof(float));
    memset(m_scale->contents(), 0, numGaussians * 3 * sizeof(float));
    memset(v_scale->contents(), 0, numGaussians * 3 * sizeof(float));
    memset(m_opacity->contents(), 0, numGaussians * sizeof(float));
    memset(v_opacity->contents(), 0, numGaussians * sizeof(float));
    memset(m_sh->contents(), 0, numGaussians * 12 * sizeof(float));
    memset(v_sh->contents(), 0, numGaussians * 12 * sizeof(float));
    simd_float4* m_rot = (simd_float4*)m_rotation->contents();
    simd_float4* v_rot = (simd_float4*)v_rotation->contents();
    for (size_t i = 0; i < numGaussians; i++) {
        m_rot[i] = simd_make_float4(0, 0, 0, 0);
        v_rot[i] = simd_make_float4(0, 0, 0, 0);
    }
}

// Remap momentum buffers after density control using old→new index mapping
void AdamOptimizer::remapMomentum(const std::vector<std::pair<size_t, size_t>>& indexMapping,
                                   size_t newCount) {
    // Allocate temporary buffers to build remapped state
    size_t posSize = newCount * 3 * sizeof(float);
    size_t scaleSize = newCount * 3 * sizeof(float);
    size_t rotSize = newCount * sizeof(simd_float4);
    size_t opacitySize = newCount * sizeof(float);
    size_t shSize = newCount * 12 * sizeof(float);

    // Create new zeroed buffers
    MTL::Buffer* new_m_pos = device->newBuffer(posSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* new_v_pos = device->newBuffer(posSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* new_m_scl = device->newBuffer(scaleSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* new_v_scl = device->newBuffer(scaleSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* new_m_rot = device->newBuffer(rotSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* new_v_rot = device->newBuffer(rotSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* new_m_op = device->newBuffer(opacitySize, MTL::ResourceStorageModeShared);
    MTL::Buffer* new_v_op = device->newBuffer(opacitySize, MTL::ResourceStorageModeShared);
    MTL::Buffer* new_m_sh = device->newBuffer(shSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* new_v_sh = device->newBuffer(shSize, MTL::ResourceStorageModeShared);

    // Zero-initialize all new buffers
    memset(new_m_pos->contents(), 0, posSize);
    memset(new_v_pos->contents(), 0, posSize);
    memset(new_m_scl->contents(), 0, scaleSize);
    memset(new_v_scl->contents(), 0, scaleSize);
    memset(new_m_rot->contents(), 0, rotSize);
    memset(new_v_rot->contents(), 0, rotSize);
    memset(new_m_op->contents(), 0, opacitySize);
    memset(new_v_op->contents(), 0, opacitySize);
    memset(new_m_sh->contents(), 0, shSize);
    memset(new_v_sh->contents(), 0, shSize);

    // Get pointers to old buffers
    float* old_m_pos = (float*)m_position->contents();
    float* old_v_pos = (float*)v_position->contents();
    float* old_m_scl = (float*)m_scale->contents();
    float* old_v_scl = (float*)v_scale->contents();
    simd_float4* old_m_rot = (simd_float4*)m_rotation->contents();
    simd_float4* old_v_rot = (simd_float4*)v_rotation->contents();
    float* old_m_op = (float*)m_opacity->contents();
    float* old_v_op = (float*)v_opacity->contents();
    float* old_m_sh = (float*)m_sh->contents();
    float* old_v_sh = (float*)v_sh->contents();

    // Get pointers to new buffers
    float* nm_pos = (float*)new_m_pos->contents();
    float* nv_pos = (float*)new_v_pos->contents();
    float* nm_scl = (float*)new_m_scl->contents();
    float* nv_scl = (float*)new_v_scl->contents();
    simd_float4* nm_rot = (simd_float4*)new_m_rot->contents();
    simd_float4* nv_rot = (simd_float4*)new_v_rot->contents();
    float* nm_op = (float*)new_m_op->contents();
    float* nv_op = (float*)new_v_op->contents();
    float* nm_sh = (float*)new_m_sh->contents();
    float* nv_sh = (float*)new_v_sh->contents();

    // Copy momentum from old index to new index
    size_t preserved = 0;
    for (const auto& mapping : indexMapping) {
        size_t oldIdx = mapping.first;
        size_t newIdx = mapping.second;

        if (oldIdx < numGaussians && newIdx < newCount) {
            // Position
            nm_pos[newIdx * 3 + 0] = old_m_pos[oldIdx * 3 + 0];
            nm_pos[newIdx * 3 + 1] = old_m_pos[oldIdx * 3 + 1];
            nm_pos[newIdx * 3 + 2] = old_m_pos[oldIdx * 3 + 2];
            nv_pos[newIdx * 3 + 0] = old_v_pos[oldIdx * 3 + 0];
            nv_pos[newIdx * 3 + 1] = old_v_pos[oldIdx * 3 + 1];
            nv_pos[newIdx * 3 + 2] = old_v_pos[oldIdx * 3 + 2];

            // Scale
            nm_scl[newIdx * 3 + 0] = old_m_scl[oldIdx * 3 + 0];
            nm_scl[newIdx * 3 + 1] = old_m_scl[oldIdx * 3 + 1];
            nm_scl[newIdx * 3 + 2] = old_m_scl[oldIdx * 3 + 2];
            nv_scl[newIdx * 3 + 0] = old_v_scl[oldIdx * 3 + 0];
            nv_scl[newIdx * 3 + 1] = old_v_scl[oldIdx * 3 + 1];
            nv_scl[newIdx * 3 + 2] = old_v_scl[oldIdx * 3 + 2];

            // Rotation
            nm_rot[newIdx] = old_m_rot[oldIdx];
            nv_rot[newIdx] = old_v_rot[oldIdx];

            // Opacity
            nm_op[newIdx] = old_m_op[oldIdx];
            nv_op[newIdx] = old_v_op[oldIdx];

            // SH (12 coefficients)
            for (int s = 0; s < 12; s++) {
                nm_sh[newIdx * 12 + s] = old_m_sh[oldIdx * 12 + s];
                nv_sh[newIdx * 12 + s] = old_v_sh[oldIdx * 12 + s];
            }

            preserved++;
        }
    }

    // Release old buffers
    m_position->release();
    v_position->release();
    m_scale->release();
    v_scale->release();
    m_rotation->release();
    v_rotation->release();
    m_opacity->release();
    v_opacity->release();
    m_sh->release();
    v_sh->release();

    // Assign new buffers
    m_position = new_m_pos;
    v_position = new_v_pos;
    m_scale = new_m_scl;
    v_scale = new_v_scl;
    m_rotation = new_m_rot;
    v_rotation = new_v_rot;
    m_opacity = new_m_op;
    v_opacity = new_v_op;
    m_sh = new_m_sh;
    v_sh = new_v_sh;

    numGaussians = newCount;
    bufferCapacity = newCount;

    std::cout << "Remapped Adam momentum: " << preserved << " preserved, "
              << (newCount - preserved) << " new (zeroed)" << std::endl;
}

// Reset Adam state for Gaussians starting at index startIdx after split/clone
void AdamOptimizer::resetStateForNewGaussians(size_t startIdx) {
    if (startIdx >= numGaussians) return;
    
    size_t numNew = numGaussians - startIdx;
    
    // Zero position momentum for new Gaussians
    float* m_pos = (float*)m_position->contents();
    float* v_pos = (float*)v_position->contents();
    memset(m_pos + startIdx * 3, 0, numNew * 3 * sizeof(float));
    memset(v_pos + startIdx * 3, 0, numNew * 3 * sizeof(float));
    
    // Zero scale momentum
    float* m_scl = (float*)m_scale->contents();
    float* v_scl = (float*)v_scale->contents();
    memset(m_scl + startIdx * 3, 0, numNew * 3 * sizeof(float));
    memset(v_scl + startIdx * 3, 0, numNew * 3 * sizeof(float));
    
    // Zero rotation momentum
    simd_float4* m_rot = (simd_float4*)m_rotation->contents();
    simd_float4* v_rot = (simd_float4*)v_rotation->contents();
    for (size_t i = startIdx; i < numGaussians; i++) {
        m_rot[i] = simd_make_float4(0, 0, 0, 0);
        v_rot[i] = simd_make_float4(0, 0, 0, 0);
    }
    
    // Zero opacity momentum
    float* m_op = (float*)m_opacity->contents();
    float* v_op = (float*)v_opacity->contents();
    memset(m_op + startIdx, 0, numNew * sizeof(float));
    memset(v_op + startIdx, 0, numNew * sizeof(float));
    
    // Zero SH momentum
    float* m_sh_ptr = (float*)m_sh->contents();
    float* v_sh_ptr = (float*)v_sh->contents();
    memset(m_sh_ptr + startIdx * 12, 0, numNew * 12 * sizeof(float));
    memset(v_sh_ptr + startIdx * 12, 0, numNew * 12 * sizeof(float));
    
    std::cout << "Reset Adam state for " << numNew << " new Gaussians starting at " << startIdx << std::endl;
}

// Debug print Adam state for a specific Gaussian
void AdamOptimizer::debugPrintState(int idx) {
    // Get state pointers
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
    for (int b = 0; b < 12; b++) {  
        printf("%02X ", rawBytes[idx * 3 * sizeof(float) + b]);
    }
    printf("\n");
    
    // Momentum should be in reasonable range -1 to 1 with 0.5 gradient clip
    // After clipping gradients to -0.5 to 0.5, momentum can at most grow by 0.1*0.5 = 0.05 per step
    if (fabsf(m_scl[idx*3+0]) > 1.0f || fabsf(m_scl[idx*3+1]) > 1.0f || fabsf(m_scl[idx*3+2]) > 1.0f) {
        printf("[WARNING] m_scale out of expected range! Max expected ~0.5 with gradient clip.\n");
        printf("[WARNING] Actual magnitude: (%f, %f, %f)\n", fabsf(m_scl[idx*3+0]), fabsf(m_scl[idx*3+1]), fabsf(m_scl[idx*3+2]));
    }
    if (fabsf(m_pos[idx*3+0]) > 1.0f || fabsf(m_pos[idx*3+1]) > 1.0f || fabsf(m_pos[idx*3+2]) > 1.0f) {
        printf("[WARNING] m_position out of expected range! Max expected ~0.5 with gradient clip.\n");
    }
}

// Print GPU debug information for Adam step
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

// Wait for the last async step to complete
void AdamOptimizer::waitForLastStep() {
    if (lastCmdBuffer) {
        lastCmdBuffer->waitUntilCompleted();
        lastCmdBuffer->release();
        lastCmdBuffer = nullptr;
    }
}

// Perform one Adam optimization step
void AdamOptimizer::step(MTL::CommandQueue* queue,
                         MTL::Buffer* gaussians,
                         MTL::Buffer* gradients,
                         float lr_position,
                         float lr_scale,
                         float lr_rotation,
                         float lr_opacity,
                         float lr_sh,
                         float lr_sh_rest,
                         float maxLogScaleTrain,
                         bool wait) {
    // Ensure any previous async step is complete before reusing state buffers
    waitForLastStep();

    timestep++;

    // Create command buffer and encoder
    MTL::CommandBuffer* cmd = queue->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();

    // Set pipeline and buffers
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

    // Set learning rates
    float lrs[7] = {lr_position, lr_scale, lr_rotation, lr_opacity, lr_sh, lr_sh_rest, maxLogScaleTrain};
    enc->setBytes(lrs, sizeof(lrs), 12);

    // Set Adam hyperparameters
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    uint32_t params[2] = {timestep, (uint32_t)numGaussians};
    enc->setBytes(&beta1, sizeof(float), 13);
    enc->setBytes(&beta2, sizeof(float), 14);
    enc->setBytes(&epsilon, sizeof(float), 15);
    enc->setBytes(params, sizeof(params), 16);
    // Debug buffer for GPU-side debugging
    enc->setBuffer(debugBuffer, 0, 17);

    // Dispatch threads
    MTL::Size grid = MTL::Size(numGaussians, 1, 1);
    MTL::Size threadgroup = MTL::Size(64, 1, 1);
    enc->dispatchThreads(grid, threadgroup);

    // End encoding and commit
    enc->endEncoding();
    cmd->commit();

    if (wait) {
        cmd->waitUntilCompleted();
    } else {
        // Retain for later waitForLastStep()
        lastCmdBuffer = cmd->retain();
    }
}


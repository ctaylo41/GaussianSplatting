#pragma once

#define GLFW_INCLUDE_NONE
#import <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#import <GLFW/glfw3native.h>

#include <Metal/Metal.hpp>
#include <Metal/Metal.h>
#include <QuartzCore/CAMetalLayer.hpp>
#include <QuartzCore/CAMetalLayer.h>
#include <QuartzCore/QuartzCore.hpp>
#include "ply_loader.hpp"
#include "camera.hpp"

struct Uniforms {
    simd_float4x4 viewProjection;
};

class MTLEngine {
public:
    void init();
    void run(Camera& camera);
    void cleanup();
    void loadGaussians(const std::vector<Gaussian>& gaussians);


private:
    void initDevice();
    void initWindow();
    
    void initCommandQueue();
    void createPipeline();
    void render(Camera& camera);
    void loadShaders();
    void createDepthTexture();
    
    MTL::Device* metalDevice;
    GLFWwindow* glfwWindow;
    NSWindow* metalWindow;
    CAMetalLayer* metalLayer;
    
    MTL::Buffer* gaussianBuffer = nullptr;
    size_t gaussianCount;
    
    MTL::Library* shaderLibrary;
    MTL::RenderPipelineState* pipelineState;
    MTL::CommandQueue* commandQueue;
    MTL::Buffer* uniformBuffer;
    MTL::RenderPipelineState* metalRenderPSO;
    MTL::DepthStencilState* depthStencilState;
    MTL::Texture* depthTexture;

};

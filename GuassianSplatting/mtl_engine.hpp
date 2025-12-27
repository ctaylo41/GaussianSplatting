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
#include "gpu_sort.hpp"

struct Uniforms {
    simd_float4x4 viewMatrix;
    simd_float4x4 projectionMatrix;
    simd_float4x4 viewProjectionMatrix;
    simd_float2 screenSize;
    simd_float2 focalLength;
    simd_float3 cameraPos;
    float _pad;
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
    void setupCallbacks();
    
    void initCommandQueue();
    void createPipeline();
    void render(Camera& camera);
    void loadShaders();
    void createDepthTexture();
    
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    
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
    
    bool isDragging = false;
    bool isPanning = false;
    double lastMouseX = 0;
    double lastMouseY = 0;
    Camera* activeCamera = nullptr;
    
    MTL::Buffer* positionBuffer = nullptr;
    GPURadixSort* gpuSort = nullptr;
    
    int windowWidth = 800;
    int windowHeight = 600;
    
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
};

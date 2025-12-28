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
#include "image_loader.hpp"
#include "colmap_loader.hpp"
#include "optimizer.hpp"
#include "density_control.hpp"

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
    void loadTrainingData(const ColmapData& colmap, const std::string& imagePath);
    void train(size_t numEpochs);
    void initHeadless();

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
    MTL::Buffer* sequentialIndexBuffer = nullptr;
    
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
    
    std::vector<TrainingImage> trainingImages;
    
    bool useTrainingView = false;
    size_t currentTrainingIndex = 0;
    
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    
    ColmapData colmapData;
    
    simd_float4x4 projectionFromColmap(const ColmapCamera& cam, float nearZ, float farZ);
    simd_float4x4 viewMatrixFromColmap(simd_float4 quat, simd_float3 translation);
    
    
    MTL::Texture* renderTarget = nullptr;
    MTL::ComputePipelineState* lossComputePSO = nullptr;
    MTL::ComputePipelineState* reductionPSO = nullptr;
    MTL::Buffer* lossBuffer = nullptr;
    MTL::Buffer* totalLossBuffer = nullptr;
    
    void createRenderTarget(uint32_t width, uint32_t height);
    void createLossPipeline();
    float computeLoss(MTL::Texture* rendered, MTL::Texture* groundTruth);
    void renderToTexture(const TrainingImage& img);
    float trainStep(size_t imageIndex);
    
    MTL::Texture* gradientTexture = nullptr;
    MTL::Buffer* gaussianGradients = nullptr;
    MTL::ComputePipelineState* pixelGradientPSO = nullptr;
    MTL::ComputePipelineState* backwardPSO = nullptr;
    
    void createBackwardPipeline();
    void backward(const TrainingImage& img, MTL::Buffer* sortedIndices);
    
    AdamOptimizer* optimizer = nullptr;
    
    bool isTraining = false;
    size_t currentEpoch = 0;
    size_t currentImageIdx = 0;
    float epochLoss = 0.0f;
    size_t epochIterations = 0;
    void updatePositionBuffer();
    
    DensityController* densityController = nullptr;
    size_t densityControlInterval = 100;
    
    

};

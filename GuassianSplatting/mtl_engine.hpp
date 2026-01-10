//
//  mtl_engine.hpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-24.
//
#pragma once

// Metal and GLFW imports
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
#include "tiled_rasterizer.hpp"

// Uniforms structure for passing camera and screen data to shaders
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
    // Public interface methods for engine creation and operation
    void init();
    void run(Camera& camera);
    void cleanup();
    void loadGaussians(const std::vector<Gaussian>& gaussians, float sceneExtent = 0.0f);
    void loadTrainingData(const ColmapData& colmap, const std::string& imagePath);
    void train(size_t numEpochs);
    void initHeadless();
    void exportTrainingViews(const std::string& outputFolder);
    const Gaussian* getGaussians() const {
            return gaussianBuffer ? (const Gaussian*)gaussianBuffer->contents() : nullptr;
    };
    size_t getGaussianCount() const {
        return gaussianCount;
    };


private:
    // Private helper methods and members
    // Initialization methods
    void initDevice();
    void initWindow();
    void setupCallbacks();
    
    // Metal setup methods
    void initCommandQueue();
    void createPipeline();
    void render(Camera& camera);
    void loadShaders();
    void createDepthTexture();
    
    // Input handling callbacks
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    
    // Member variables
    // Metal and windowing
    MTL::Device* metalDevice;
    GLFWwindow* glfwWindow;
    NSWindow* metalWindow;
    CAMetalLayer* metalLayer;
    
    // Metal resources
    MTL::Buffer* gaussianBuffer = nullptr;
    size_t gaussianCount;
    
    // Shader and pipeline
    MTL::Library* shaderLibrary;
    MTL::RenderPipelineState* pipelineState;
    MTL::CommandQueue* commandQueue;
    MTL::Buffer* uniformBuffer;
    MTL::RenderPipelineState* metalRenderPSO;
    MTL::DepthStencilState* depthStencilState;
    MTL::Texture* depthTexture;
    MTL::Buffer* sequentialIndexBuffer = nullptr;
    
    // Input handling
    bool isDragging = false;
    bool isPanning = false;
    double lastMouseX = 0;
    double lastMouseY = 0;
    Camera* activeCamera = nullptr;
    
    // Buffers and utilities
    MTL::Buffer* positionBuffer = nullptr;
    GPURadixSort32* gpuSort = nullptr;
    
    // Window dimensions
    int windowWidth = 800;
    int windowHeight = 600;
    
    // Callback for window resize
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    
    // Training data
    std::vector<TrainingImage> trainingImages;
    // Default to training view for proper scale
    bool useTrainingView = true;  
    size_t currentTrainingIndex = 0;
    
    // Input handling
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    
    // COLMAP data
    ColmapData colmapData;
    
    // Camera matrix conversions
    simd_float4x4 projectionFromColmap(const ColmapCamera& cam, float nearZ, float farZ);
    simd_float4x4 viewMatrixFromColmap(simd_float4 quat, simd_float3 translation);
    
    // Render target for training
    MTL::Texture* renderTarget = nullptr;
    void createRenderTarget(uint32_t width, uint32_t height);
    
    // Loss computation
    MTL::ComputePipelineState* lossComputePSO = nullptr;
    MTL::ComputePipelineState* ssimComputePSO = nullptr;
    MTL::ComputePipelineState* combinedLossPSO = nullptr;
    MTL::ComputePipelineState* reductionPSO = nullptr;
    MTL::Buffer* lossBuffer = nullptr;
    MTL::Buffer* ssimBuffer = nullptr;
    MTL::Buffer* combinedLossBuffer = nullptr;
    MTL::Buffer* totalLossBuffer = nullptr;
    // Paper: 0.2 weight for D-SSIM
    float lambdaDSSIM = 0.2f;  
    void createLossPipeline();
    float computeLoss(MTL::Texture* rendered, MTL::Texture* groundTruth);
    
    // Tiled rasterizer for training
    TiledRasterizer* tiledRasterizer = nullptr;
    MTL::Buffer* gaussianGradients = nullptr;
    
    // Training step with learning rate parameters
    float trainStep(size_t imageIndex, 
                    float lr_position = 0.00016f,
                    float lr_scale = 0.005f,
                    float lr_rotation = 0.001f,
                    float lr_opacity = 0.05f,
                    float lr_sh = 0.0025f);
    void updatePositionBuffer();
    
    // Optimizer and density control
    AdamOptimizer* optimizer = nullptr;
    DensityController* densityController = nullptr;
    size_t densityControlInterval = 100;
    float sceneExtent = 1.0f;  // Scene extent for density control thresholds
    
    // Training state
    bool isTraining = false;
    size_t currentEpoch = 0;
    size_t currentImageIdx = 0;
    float epochLoss = 0.0f;
    size_t epochIterations = 0;

};

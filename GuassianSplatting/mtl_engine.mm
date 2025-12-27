//
//  mtl_engine.mm
//  Metal-Guide
//

#include "mtl_engine.hpp"
#include "gpu_sort.hpp"
#include <iostream>

void MTLEngine::init() {
    initDevice();
    initCommandQueue();
    initWindow();
    setupCallbacks();
    createDepthTexture();
    loadShaders();
    createPipeline();
    uniformBuffer = metalDevice->newBuffer(sizeof(Uniforms), MTL::ResourceStorageModeShared);
}

void MTLEngine::run(Camera& camera) {
    activeCamera = &camera;
    while (!glfwWindowShouldClose(glfwWindow)) {
        glfwPollEvents();
        render(camera);
    }
    activeCamera=nullptr;
}

void MTLEngine::cleanup() {
    glfwTerminate();
    if(gaussianBuffer) {
        gaussianBuffer->release();
    }
    
    if(positionBuffer) {
        positionBuffer->release();
    }
    
    if(gpuSort) {
        delete gpuSort;
    }
}

void MTLEngine::initDevice() {
    metalDevice = MTL::CreateSystemDefaultDevice();
}

void MTLEngine::initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindow = glfwCreateWindow(800, 600, "Gaussian Splatting", NULL, NULL);
    if (!glfwWindow) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    
    metalWindow = glfwGetCocoaWindow(glfwWindow);
    metalLayer = [CAMetalLayer layer];
    metalLayer.device = (__bridge id<MTLDevice>)metalDevice;
    metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    metalWindow.contentView.layer = metalLayer;
    metalWindow.contentView.wantsLayer = YES;
    metalLayer.frame = metalWindow.contentView.bounds;
    metalLayer.drawableSize = CGSizeMake(800,600);
}

void MTLEngine::setupCallbacks() {
    glfwSetWindowUserPointer(glfwWindow, this);
    glfwSetFramebufferSizeCallback(glfwWindow, framebufferSizeCallback);
    glfwSetMouseButtonCallback(glfwWindow, mouseButtonCallback);
    glfwSetCursorPosCallback(glfwWindow, cursorPosCallback);
    glfwSetScrollCallback(glfwWindow, scrollCallback);
}

void MTLEngine::mouseButtonCallback(GLFWwindow *window, int button, int action, int mods) {
    MTLEngine* engine = static_cast<MTLEngine*>(glfwGetWindowUserPointer(window));
    
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            engine->isDragging = true;
            glfwGetCursorPos(window, &engine->lastMouseX, &engine->lastMouseY);
        } else if (action == GLFW_RELEASE) {
            engine->isDragging = false;
        }
    }
    
    if (button == GLFW_MOUSE_BUTTON_RIGHT || button == GLFW_MOUSE_BUTTON_MIDDLE) {
        if (action == GLFW_PRESS) {
            engine->isPanning = true;
            glfwGetCursorPos(window, &engine->lastMouseX, &engine->lastMouseY);
        } else if (action == GLFW_RELEASE) {
            engine->isPanning = false;
        }
    }
}

void MTLEngine::cursorPosCallback(GLFWwindow *window, double xpos, double ypos) {
    MTLEngine* engine = static_cast<MTLEngine*>(glfwGetWindowUserPointer(window));
    
    if (!engine->activeCamera) return;
    
    double deltaX = xpos - engine->lastMouseX;
    double deltaY = ypos - engine->lastMouseY;
    
    if(engine->isDragging) {
        float orbitSpeed = 0.005f;
        engine->activeCamera->orbit(-deltaX * orbitSpeed, -deltaY * orbitSpeed);
    }
    
    if(engine->isPanning) {
        engine->activeCamera->pan(deltaX, deltaY);
    }
    
    engine->lastMouseX = xpos;
    engine->lastMouseY = ypos;
}

void MTLEngine::scrollCallback(GLFWwindow *window, double xoffset, double yoffset) {
    MTLEngine* engine = static_cast<MTLEngine*>(glfwGetWindowUserPointer(window));
    
    if(!engine->activeCamera) return;
    
    float zoomSpeed = 0.5f;
    engine->activeCamera->zoom(-yoffset*zoomSpeed);
}

void MTLEngine::loadGaussians(const std::vector<Gaussian>& gaussians) {
    gaussianCount = gaussians.size();
    gaussianBuffer = metalDevice->newBuffer(gaussians.data(), gaussianCount*sizeof(gaussians[0]),MTL::ResourceStorageModeShared);
    
    std::vector<simd_float3> positions(gaussianCount);
    for (size_t i=0;i<gaussianCount;i++) {
        positions[i] = gaussians[i].position;
    }
    
    positionBuffer = metalDevice->newBuffer(positions.data(), gaussianCount * sizeof(simd_float3), MTL::ResourceStorageModeShared);
    
    gpuSort = new GPURadixSort(metalDevice, gaussianCount);
}

void MTLEngine::initCommandQueue() {
    commandQueue = metalDevice->newCommandQueue();
}

void MTLEngine::createPipeline() {
    MTL::Function* vertexShader = shaderLibrary->newFunction(NS::String::string("vertexShader", NS::ASCIIStringEncoding));
    assert(vertexShader);
    MTL::Function* fragmentShader = shaderLibrary->newFunction(NS::String::string("fragmentShader", NS::ASCIIStringEncoding));
    assert(fragmentShader);
    
    MTL::RenderPipelineDescriptor* renderPipelineDescriptor = MTL::RenderPipelineDescriptor::alloc()->init();
    renderPipelineDescriptor->setVertexFunction(vertexShader);
    renderPipelineDescriptor->setFragmentFunction(fragmentShader);

    auto colorAttachment = renderPipelineDescriptor->colorAttachments()->object(0);
    colorAttachment->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    colorAttachment->setBlendingEnabled(true);
    colorAttachment->setSourceRGBBlendFactor(MTL::BlendFactorOne);
    colorAttachment->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
    colorAttachment->setRgbBlendOperation(MTL::BlendOperationAdd);
    colorAttachment->setSourceAlphaBlendFactor(MTL::BlendFactorOne);
    colorAttachment->setDestinationAlphaBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
    colorAttachment->setAlphaBlendOperation(MTL::BlendOperationAdd);
    
    NS::Error* error;
    metalRenderPSO = metalDevice->newRenderPipelineState(renderPipelineDescriptor, &error);
    
    if(metalRenderPSO == nil) {
        std::cout << "Error creating render pipeline state: " << error << std::endl;
        std::exit(0);
    }
    
    renderPipelineDescriptor->release();
    vertexShader->release();
    fragmentShader->release();
    
}

void MTLEngine::loadShaders() {
    shaderLibrary = metalDevice->newDefaultLibrary();
    if(!shaderLibrary) {
        std::cerr << "Failed to load shader library" << std::endl;
        std::exit(1);
    }
}

void MTLEngine::createDepthTexture() {
    MTL::TextureDescriptor* depthDesc = MTL::TextureDescriptor::alloc()->init();
    depthDesc->setWidth(800);
    depthDesc->setHeight(600);
    depthDesc->setPixelFormat(MTL::PixelFormatDepth32Float);
    depthDesc->setStorageMode(MTL::StorageModePrivate);
    depthDesc->setUsage(MTL::TextureUsageRenderTarget);
    depthTexture = metalDevice->newTexture(depthDesc);
    depthDesc->release();
}

void MTLEngine::render(Camera &camera) {
    MTL::Buffer* sortedIndices = gpuSort->sort(commandQueue, positionBuffer, camera.get_position(), gaussianCount);
    
    float fovY = 45.0f * M_PI / 180.0f;
    float fy = windowHeight/ (2.0f * tan(fovY/2.0f));
    float fx = fy;
    
    Uniforms uniforms;
    uniforms.viewMatrix = camera.get_view_matrix();
    uniforms.projectionMatrix = camera.get_projection_matrix();
    uniforms.viewProjectionMatrix = matrix_multiply(camera.get_projection_matrix(), camera.get_view_matrix());
    uniforms.screenSize = simd_make_float2((float)windowWidth, (float)windowHeight);
    uniforms.focalLength = simd_make_float2(fx, fy);
    uniforms.cameraPos = camera.get_position();
    memcpy(uniformBuffer->contents(), &uniforms, sizeof(Uniforms));
    
    CA::MetalDrawable* drawable = (__bridge CA::MetalDrawable*)[metalLayer nextDrawable];
    if (!drawable) return;
    
    MTL::RenderPassDescriptor* renderPassDesc = MTL::RenderPassDescriptor::alloc()->init();
    renderPassDesc->colorAttachments()->object(0)->setTexture(drawable->texture());
    renderPassDesc->colorAttachments()->object(0)->setLoadAction(MTL::LoadActionClear);
    renderPassDesc->colorAttachments()->object(0)->setStoreAction(MTL::StoreActionStore);
    renderPassDesc->colorAttachments()->object(0)->setClearColor(MTL::ClearColor(0.0, 0.0, 0.0, 1.0));
    
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::RenderCommandEncoder* encoder = commandBuffer->renderCommandEncoder(renderPassDesc);
    
    encoder->setRenderPipelineState(metalRenderPSO);
    encoder->setVertexBuffer(gaussianBuffer,0, 0);
    encoder->setVertexBuffer(uniformBuffer, 0, 1);
    encoder->setVertexBuffer(sortedIndices, 0, 2);
    encoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4), gaussianCount);
    
    encoder->endEncoding();
    commandBuffer->presentDrawable(drawable);
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    renderPassDesc->release();
}

void MTLEngine::framebufferSizeCallback(GLFWwindow *window, int width, int height) {
    if (width==0 || height==0) return;
    
    MTLEngine* engine = static_cast<MTLEngine*>(glfwGetWindowUserPointer(window));
    engine->windowWidth = width;
    engine->windowHeight = height;
    
    engine->metalLayer.drawableSize = CGSizeMake(width, height);
    
    if (engine->activeCamera) {
        engine->activeCamera->setAspectRatio((float)width / (float)height);
    }
}

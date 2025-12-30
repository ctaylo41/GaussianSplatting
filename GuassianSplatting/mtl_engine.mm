//
//  mtl_engine.mm
//  GuassianSplatting
//

#include "mtl_engine.hpp"
#include "gpu_sort.hpp"
#include <iostream>
#include "image_loader.hpp"
#include "gradients.hpp"
#include <chrono>

void MTLEngine::init() {
    initDevice();
    initCommandQueue();
    initWindow();
    setupCallbacks();
    createDepthTexture();
    loadShaders();
    createPipeline();
    createLossPipeline();
    uniformBuffer = metalDevice->newBuffer(sizeof(Uniforms), MTL::ResourceStorageModeShared);
}

void MTLEngine::initHeadless() {
    initDevice();
    initCommandQueue();
    loadShaders();
    createPipeline();
    createLossPipeline();
    uniformBuffer = metalDevice->newBuffer(sizeof(Uniforms), MTL::ResourceStorageModeShared);
}

void MTLEngine::run(Camera& camera) {
    activeCamera = &camera;
    size_t totalIterations = 0;
    
    while (!glfwWindowShouldClose(glfwWindow)) {
        glfwPollEvents();
        
        if (isTraining && !trainingImages.empty()) {
            float loss = trainStep(currentImageIdx);
            epochLoss += loss;
            epochIterations++;
            totalIterations++;
            
            if (epochIterations % 10 == 0) {
                std::cout << "\rEpoch " << currentEpoch
                          << " | Image " << currentImageIdx << "/" << trainingImages.size()
                          << " | Loss: " << (epochLoss / epochIterations) << std::flush;
            }
            
            if (totalIterations % densityControlInterval == 0 && totalIterations > 500) {
                densityController->apply(commandQueue, gaussianBuffer, positionBuffer,
                                         nullptr, gaussianCount, totalIterations);
                
                if (gaussianGradients->length() < gaussianCount * sizeof(GaussianGradients)) {
                    gaussianGradients->release();
                    gaussianGradients = metalDevice->newBuffer(gaussianCount * sizeof(GaussianGradients),
                                                               MTL::ResourceStorageModeShared);
                }
            }
            
            currentImageIdx++;
            if (currentImageIdx >= trainingImages.size()) {
                std::cout << std::endl << "=== Epoch " << currentEpoch << " complete | Avg Loss: "
                          << (epochLoss / epochIterations) << " ===" << std::endl;
                currentImageIdx = 0;
                currentEpoch++;
                epochLoss = 0.0f;
                epochIterations = 0;
                updatePositionBuffer();
            }
        }
        
        render(camera);
    }
    activeCamera = nullptr;
}

void MTLEngine::cleanup() {
    glfwTerminate();
    if (gaussianBuffer) gaussianBuffer->release();
    if (positionBuffer) positionBuffer->release();
    if (gpuSort) delete gpuSort;
    if (optimizer) delete optimizer;
    if (densityController) delete densityController;
    if (tiledRasterizer) delete tiledRasterizer;
    if (gaussianGradients) gaussianGradients->release();
}

void MTLEngine::initDevice() {
    metalDevice = MTL::CreateSystemDefaultDevice();
    if (!metalDevice) {
        std::cerr << "Failed to create Metal device" << std::endl;
        std::exit(1);
    }
    std::cout << "Using Metal device: " << metalDevice->name()->utf8String() << std::endl;
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
    metalLayer.pixelFormat = MTLPixelFormatRGBA8Unorm;
    metalWindow.contentView.layer = metalLayer;
    metalWindow.contentView.wantsLayer = YES;
    metalLayer.frame = metalWindow.contentView.bounds;
    metalLayer.drawableSize = CGSizeMake(800, 600);
}

void MTLEngine::setupCallbacks() {
    glfwSetWindowUserPointer(glfwWindow, this);
    glfwSetFramebufferSizeCallback(glfwWindow, framebufferSizeCallback);
    glfwSetMouseButtonCallback(glfwWindow, mouseButtonCallback);
    glfwSetCursorPosCallback(glfwWindow, cursorPosCallback);
    glfwSetScrollCallback(glfwWindow, scrollCallback);
    glfwSetKeyCallback(glfwWindow, keyCallback);
}

void MTLEngine::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
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

void MTLEngine::cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    MTLEngine* engine = static_cast<MTLEngine*>(glfwGetWindowUserPointer(window));
    if (!engine->activeCamera) return;
    
    double deltaX = xpos - engine->lastMouseX;
    double deltaY = ypos - engine->lastMouseY;
    
    if (engine->isDragging) {
        float orbitSpeed = 0.005f;
        engine->activeCamera->orbit(-deltaX * orbitSpeed, -deltaY * orbitSpeed);
    }
    
    if (engine->isPanning) {
        engine->activeCamera->pan(deltaX, deltaY);
    }
    
    engine->lastMouseX = xpos;
    engine->lastMouseY = ypos;
}

void MTLEngine::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    MTLEngine* engine = static_cast<MTLEngine*>(glfwGetWindowUserPointer(window));
    if (!engine->activeCamera) return;
    
    float zoomSpeed = 0.5f;
    engine->activeCamera->zoom(-yoffset * zoomSpeed);
}

void MTLEngine::loadGaussians(const std::vector<Gaussian>& gaussians) {
    gaussianCount = gaussians.size();
    gaussianBuffer = metalDevice->newBuffer(gaussians.data(), gaussianCount * sizeof(Gaussian),
                                            MTL::ResourceStorageModeShared);
    
    std::vector<simd_float3> positions(gaussianCount);
    for (size_t i = 0; i < gaussianCount; i++) {
        positions[i] = gaussians[i].position;
    }
    positionBuffer = metalDevice->newBuffer(positions.data(), gaussianCount * sizeof(simd_float3),
                                            MTL::ResourceStorageModeShared);
    
    gpuSort = new GPURadixSort(metalDevice, 2000000);
    optimizer = new AdamOptimizer(metalDevice, shaderLibrary, 2000000);
    densityController = new DensityController(metalDevice, shaderLibrary);
    densityController->resetAccumulator(gaussianCount);
    
    gaussianGradients = metalDevice->newBuffer(gaussianCount * sizeof(GaussianGradients),
                                               MTL::ResourceStorageModeShared);
    
    tiledRasterizer = new TiledRasterizer(metalDevice, shaderLibrary, 2000000);
    
    std::cout << "Loaded " << gaussianCount << " Gaussians" << std::endl;
}

void MTLEngine::initCommandQueue() {
    commandQueue = metalDevice->newCommandQueue();
}

void MTLEngine::createPipeline() {
    MTL::Function* vertexShader = shaderLibrary->newFunction(NS::String::string("vertexShader", NS::ASCIIStringEncoding));
    assert(vertexShader);
    MTL::Function* fragmentShader = shaderLibrary->newFunction(NS::String::string("fragmentShader", NS::ASCIIStringEncoding));
    assert(fragmentShader);
    
    MTL::RenderPipelineDescriptor* desc = MTL::RenderPipelineDescriptor::alloc()->init();
    desc->setVertexFunction(vertexShader);
    desc->setFragmentFunction(fragmentShader);
    
    auto colorAttachment = desc->colorAttachments()->object(0);
    colorAttachment->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
    colorAttachment->setBlendingEnabled(true);
    colorAttachment->setSourceRGBBlendFactor(MTL::BlendFactorOne);
    colorAttachment->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
    colorAttachment->setRgbBlendOperation(MTL::BlendOperationAdd);
    colorAttachment->setSourceAlphaBlendFactor(MTL::BlendFactorOne);
    colorAttachment->setDestinationAlphaBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
    colorAttachment->setAlphaBlendOperation(MTL::BlendOperationAdd);
    
    NS::Error* error;
    metalRenderPSO = metalDevice->newRenderPipelineState(desc, &error);
    
    if (metalRenderPSO == nil) {
        std::cout << "Error creating render pipeline state: " << error << std::endl;
        std::exit(0);
    }
    
    desc->release();
    vertexShader->release();
    fragmentShader->release();
}

void MTLEngine::loadShaders() {
    shaderLibrary = metalDevice->newDefaultLibrary();
    if (!shaderLibrary) {
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

void MTLEngine::render(Camera& camera) {
    MTL::Buffer* sortedIndices = gpuSort->sort(commandQueue, positionBuffer, camera.get_position(), gaussianCount);
    
    Uniforms uniforms;
    
    if (useTrainingView && !trainingImages.empty()) {
        const TrainingImage& img = trainingImages[currentTrainingIndex];
        const ColmapCamera& cam = colmapData.cameras.at(img.cameraId);
        
        uniforms.viewMatrix = viewMatrixFromColmap(img.rotation, img.translation);
        uniforms.projectionMatrix = projectionFromColmap(cam, 0.1f, 1000.0f);
        uniforms.viewProjectionMatrix = matrix_multiply(uniforms.projectionMatrix, uniforms.viewMatrix);
        uniforms.screenSize = simd_make_float2((float)cam.width, (float)cam.height);
        uniforms.focalLength = simd_make_float2(cam.fx, cam.fy);
        
        simd_float3x3 R;
        R.columns[0] = uniforms.viewMatrix.columns[0].xyz;
        R.columns[1] = uniforms.viewMatrix.columns[1].xyz;
        R.columns[2] = uniforms.viewMatrix.columns[2].xyz;
        uniforms.cameraPos = -matrix_multiply(simd_transpose(R), img.translation);
    } else {
        float fovY = 45.0f * M_PI / 180.0f;
        float fy = windowHeight / (2.0f * tan(fovY / 2.0f));
        float fx = fy;
        
        uniforms.viewMatrix = camera.get_view_matrix();
        uniforms.projectionMatrix = camera.get_projection_matrix();
        uniforms.viewProjectionMatrix = matrix_multiply(camera.get_projection_matrix(), camera.get_view_matrix());
        uniforms.screenSize = simd_make_float2((float)windowWidth, (float)windowHeight);
        uniforms.focalLength = simd_make_float2(fx, fy);
        uniforms.cameraPos = camera.get_position();
    }
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
    encoder->setVertexBuffer(gaussianBuffer, 0, 0);
    encoder->setVertexBuffer(uniformBuffer, 0, 1);
    encoder->setVertexBuffer(sortedIndices, 0, 2);
    encoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4), gaussianCount);
    
    encoder->endEncoding();
    commandBuffer->presentDrawable(drawable);
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    renderPassDesc->release();
}

void MTLEngine::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    if (width == 0 || height == 0) return;
    
    MTLEngine* engine = static_cast<MTLEngine*>(glfwGetWindowUserPointer(window));
    engine->windowWidth = width;
    engine->windowHeight = height;
    
    engine->metalLayer.drawableSize = CGSizeMake(width, height);
    
    if (engine->activeCamera) {
        engine->activeCamera->setAspectRatio((float)width / (float)height);
    }
}

void MTLEngine::loadTrainingData(const ColmapData& colmap, const std::string& imagePath) {
    colmapData = colmap;
    trainingImages = loadTrainingImages(metalDevice, colmap, imagePath);
    std::cout << "Loaded " << trainingImages.size() << " training images" << std::endl;
}

simd_float4x4 MTLEngine::viewMatrixFromColmap(simd_float4 quat, simd_float3 translation) {
    float w = quat.x, x = quat.y, y = quat.z, z = quat.w;
    
    // Camera-to-world rotation from quaternion
    matrix_float3x3 R;
    R.columns[0] = simd_make_float3(1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y));
    R.columns[1] = simd_make_float3(2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x));
    R.columns[2] = simd_make_float3(2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y));
    
    // For world-to-view: invert the camera-to-world transform
    // View matrix = [R^T | -R^T * t] where R is cam-to-world rotation, t is cam position
    matrix_float3x3 Rt = simd_transpose(R);
    simd_float3 viewTranslation = -simd_mul(Rt, translation);
    
    simd_float4x4 view;
    view.columns[0] = simd_make_float4(Rt.columns[0], 0);
    view.columns[1] = simd_make_float4(Rt.columns[1], 0);
    view.columns[2] = simd_make_float4(Rt.columns[2], 0);
    view.columns[3] = simd_make_float4(viewTranslation, 1);
    
    return view;
}

simd_float4x4 MTLEngine::projectionFromColmap(const ColmapCamera& cam, float nearZ, float farZ) {
    float fx = cam.fx;
    float fy = cam.fy;
    float cx = cam.cx;
    float cy = cam.cy;
    float w = (float)cam.width;
    float h = (float)cam.height;
    
    simd_float4x4 proj = {0};
    proj.columns[0][0] = 2.0f * fx / w;
    proj.columns[1][1] = 2.0f * fy / h;
    proj.columns[2][0] = (w - 2.0f * cx) / w;
    proj.columns[2][1] = (2.0f * cy - h) / h;
    proj.columns[2][2] = farZ / (farZ - nearZ);
    proj.columns[2][3] = 1.0f;
    proj.columns[3][2] = -(farZ * nearZ) / (farZ - nearZ);
    
    return proj;
}

void MTLEngine::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    MTLEngine* engine = static_cast<MTLEngine*>(glfwGetWindowUserPointer(window));
    
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_T) {
            engine->useTrainingView = !engine->useTrainingView;
            std::cout << "Training view: " << (engine->useTrainingView ? "ON" : "OFF") << std::endl;
        }
        
        if (key == GLFW_KEY_SPACE) {
            engine->isTraining = !engine->isTraining;
            std::cout << "Training: " << (engine->isTraining ? "STARTED" : "PAUSED") << std::endl;
        }
        
        if (key == GLFW_KEY_LEFT && engine->useTrainingView) {
            if (engine->currentTrainingIndex > 0) engine->currentTrainingIndex--;
            std::cout << "Training image: " << engine->currentTrainingIndex << std::endl;
        }
        
        if (key == GLFW_KEY_RIGHT && engine->useTrainingView) {
            if (engine->currentTrainingIndex < engine->trainingImages.size() - 1) engine->currentTrainingIndex++;
            std::cout << "Training image: " << engine->currentTrainingIndex << std::endl;
        }
    }
}

void MTLEngine::createRenderTarget(uint32_t width, uint32_t height) {
    if (renderTarget) renderTarget->release();
    
    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::alloc()->init();
    desc->setWidth(width);
    desc->setHeight(height);
    desc->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
    desc->setStorageMode(MTL::StorageModeShared);
    desc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    renderTarget = metalDevice->newTexture(desc);
    desc->release();
}

void MTLEngine::createLossPipeline() {
    NS::Error* error = nullptr;
    
    MTL::Function* lossFunc = shaderLibrary->newFunction(NS::String::string("computeL1Loss", NS::ASCIIStringEncoding));
    lossComputePSO = metalDevice->newComputePipelineState(lossFunc, &error);
    if (!lossComputePSO) {
        std::cerr << "Failed to create loss pipeline: " << error->localizedDescription()->utf8String() << std::endl;
    }
    lossFunc->release();
    
    MTL::Function* reduceFunc = shaderLibrary->newFunction(NS::String::string("reduceLoss", NS::ASCIIStringEncoding));
    reductionPSO = metalDevice->newComputePipelineState(reduceFunc, &error);
    if (!reductionPSO) {
        std::cerr << "Failed to create reduction pipeline: " << error->localizedDescription()->utf8String() << std::endl;
    }
    reduceFunc->release();
}

float MTLEngine::computeLoss(MTL::Texture* rendered, MTL::Texture* groundTruth) {
    uint32_t width = rendered->width();
    uint32_t height = rendered->height();
    uint32_t pixelCount = width * height;
    
    if (!lossBuffer || lossBuffer->length() < pixelCount * sizeof(float)) {
        if (lossBuffer) lossBuffer->release();
        lossBuffer = metalDevice->newBuffer(pixelCount * sizeof(float), MTL::ResourceStorageModeShared);
    }
    
    if (!totalLossBuffer) {
        totalLossBuffer = metalDevice->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
    }
    
    float zero = 0.0f;
    memcpy(totalLossBuffer->contents(), &zero, sizeof(float));
    
    MTL::CommandBuffer* cmdBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = cmdBuffer->computeCommandEncoder();
    
    encoder->setComputePipelineState(lossComputePSO);
    encoder->setTexture(rendered, 0);
    encoder->setTexture(groundTruth, 1);
    encoder->setBuffer(lossBuffer, 0, 0);
    
    MTL::Size gridSize = MTL::Size(width, height, 1);
    MTL::Size threadGroupSize = MTL::Size(16, 16, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    
    encoder->setComputePipelineState(reductionPSO);
    encoder->setBuffer(lossBuffer, 0, 0);
    encoder->setBuffer(totalLossBuffer, 0, 1);
    encoder->setBytes(&pixelCount, sizeof(uint32_t), 2);
    
    uint32_t reductionThreads = 1024;
    encoder->dispatchThreads(MTL::Size(reductionThreads, 1, 1), MTL::Size(64, 1, 1));
    
    encoder->endEncoding();
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();
    
    float totalLoss = *(float*)totalLossBuffer->contents();
    return totalLoss / pixelCount;
}

float MTLEngine::trainStep(size_t imageIndex) {
    if (imageIndex >= trainingImages.size()) return 0.0f;
    
    const TrainingImage& img = trainingImages[imageIndex];
    const ColmapCamera& cam = colmapData.cameras.at(img.cameraId);
    
    if (!renderTarget || renderTarget->width() != cam.width || renderTarget->height() != cam.height) {
        createRenderTarget(cam.width, cam.height);
    }
    
    TiledUniforms uniforms;
    uniforms.viewMatrix = viewMatrixFromColmap(img.rotation, img.translation);
    uniforms.projectionMatrix = projectionFromColmap(cam, 0.1f, 1000.0f);
    uniforms.viewProjectionMatrix = matrix_multiply(uniforms.projectionMatrix, uniforms.viewMatrix);
    uniforms.screenSize = simd_make_float2((float)cam.width, (float)cam.height);
    uniforms.focalLength = simd_make_float2(cam.fx, cam.fy);
    
    simd_float3x3 R;
    R.columns[0] = uniforms.viewMatrix.columns[0].xyz;
    R.columns[1] = uniforms.viewMatrix.columns[1].xyz;
    R.columns[2] = uniforms.viewMatrix.columns[2].xyz;
    uniforms.cameraPos = -matrix_multiply(simd_transpose(R), img.translation);
    
    // Forward pass
    tiledRasterizer->forward(commandQueue, gaussianBuffer, gaussianCount, uniforms, renderTarget);
    
    // Compute loss
    float loss = computeLoss(renderTarget, img.texture);
    
    // Backward pass
    tiledRasterizer->backward(commandQueue, gaussianBuffer, gaussianGradients, gaussianCount,
                              uniforms, renderTarget, img.texture);
    
    // Accumulate for density control
    densityController->accumulateGradients(commandQueue, gaussianGradients, gaussianCount);
    
    // Optimizer step
    optimizer->step(commandQueue, gaussianBuffer, gaussianGradients,
                    0.0008f,   // position lr
                    0.005f,     // scale lr
                    0.001f,     // rotation lr
                    0.1f,      // opacity lr
                    0.005f);   // sh lr
    
    return loss;
}

void MTLEngine::updatePositionBuffer() {
    Gaussian* gaussians = (Gaussian*)gaussianBuffer->contents();
    simd_float3* positions = (simd_float3*)positionBuffer->contents();
    
    for (size_t i = 0; i < gaussianCount; i++) {
        positions[i] = gaussians[i].position;
    }
}

void MTLEngine::train(size_t numEpochs) {
    std::cout << "Starting training for " << numEpochs << " epochs..." << std::endl;
    std::cout << "Gaussians: " << gaussianCount << " | Images: " << trainingImages.size() << std::endl;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    size_t totalIterations = 0;
    
    for (size_t epoch = 0; epoch < numEpochs; epoch++) {
        float epochLoss = 0.0f;
        auto epochStart = std::chrono::high_resolution_clock::now();
        
        for (size_t imgIdx = 0; imgIdx < trainingImages.size(); imgIdx++) {
            float loss = trainStep(imgIdx);
            epochLoss += loss;
            totalIterations++;
            
            // Progress update every 20 images
            if (imgIdx % 20 == 0) {
                std::cout << "\rEpoch " << epoch << " [" << imgIdx << "/" << trainingImages.size()
                          << "] Loss: " << (epochLoss / (imgIdx + 1)) << std::flush;
            }
            
            // Density control
            if (totalIterations % densityControlInterval == 0 && totalIterations > 500) {
                densityController->apply(commandQueue, gaussianBuffer, positionBuffer,
                                         nullptr, gaussianCount, totalIterations);
                
                if (gaussianGradients->length() < gaussianCount * sizeof(GaussianGradients)) {
                    gaussianGradients->release();
                    gaussianGradients = metalDevice->newBuffer(gaussianCount * sizeof(GaussianGradients),
                                                               MTL::ResourceStorageModeShared);
                }
            }
        }
        
        updatePositionBuffer();
        
        auto epochEnd = std::chrono::high_resolution_clock::now();
        auto epochDuration = std::chrono::duration_cast<std::chrono::seconds>(epochEnd - epochStart).count();
        
        std::cout << std::endl << "=== Epoch " << epoch << " | Loss: "
                  << (epochLoss / trainingImages.size())
                  << " | Gaussians: " << gaussianCount
                  << " | Time: " << epochDuration << "s ===" << std::endl;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    
    std::cout << "Training complete! Total time: " << totalDuration << "s" << std::endl;
}

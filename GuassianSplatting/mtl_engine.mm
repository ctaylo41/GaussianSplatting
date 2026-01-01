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
#include <fstream>

// Debug: Save a Metal texture to a PPM file
void saveTextureToPPM(MTL::Texture* texture, MTL::Device* device, MTL::CommandQueue* queue, const char* filename) {
    uint32_t width = texture->width();
    uint32_t height = texture->height();
    size_t bytesPerPixel = 4;  // RGBA
    size_t bytesPerRow = width * bytesPerPixel;
    size_t totalBytes = bytesPerRow * height;
    
    // Create a shared buffer to read back the texture
    MTL::Buffer* readbackBuffer = device->newBuffer(totalBytes, MTL::ResourceStorageModeShared);
    
    MTL::CommandBuffer* cmdBuffer = queue->commandBuffer();
    MTL::BlitCommandEncoder* blit = cmdBuffer->blitCommandEncoder();
    blit->copyFromTexture(texture, 0, 0, MTL::Origin(0, 0, 0), MTL::Size(width, height, 1),
                          readbackBuffer, 0, bytesPerRow, 0);
    blit->endEncoding();
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();
    
    uint8_t* data = (uint8_t*)readbackBuffer->contents();
    
    // Write PPM file (simple format)
    std::ofstream file(filename, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    
    // Convert RGBA float to RGB bytes
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            size_t offset = y * bytesPerRow + x * bytesPerPixel;
            // Assuming texture format is RGBA8Unorm or similar
            // If float, we need different handling
            file.put(data[offset]);     // R
            file.put(data[offset + 1]); // G
            file.put(data[offset + 2]); // B
        }
    }
    
    file.close();
    readbackBuffer->release();
    printf("Saved render to %s (%dx%d)\n", filename, width, height);
}

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
    
    // Compute scene extent for density control (official 3DGS approach)
    float min_x = FLT_MAX, min_y = FLT_MAX, min_z = FLT_MAX;
    float max_x = -FLT_MAX, max_y = -FLT_MAX, max_z = -FLT_MAX;
    
    std::vector<simd_float3> positions(gaussianCount);
    for (size_t i = 0; i < gaussianCount; i++) {
        positions[i] = gaussians[i].position;
        min_x = std::min(min_x, gaussians[i].position.x);
        min_y = std::min(min_y, gaussians[i].position.y);
        min_z = std::min(min_z, gaussians[i].position.z);
        max_x = std::max(max_x, gaussians[i].position.x);
        max_y = std::max(max_y, gaussians[i].position.y);
        max_z = std::max(max_z, gaussians[i].position.z);
    }
    positionBuffer = metalDevice->newBuffer(positions.data(), gaussianCount * sizeof(simd_float3),
                                            MTL::ResourceStorageModeShared);
    
    // Set scene extent for density control (matches official 3DGS)
    float sceneExtent = std::sqrt((max_x - min_x) * (max_x - min_x) +
                                   (max_y - min_y) * (max_y - min_y) +
                                   (max_z - min_z) * (max_z - min_z));
    DensityController::setSceneExtent(sceneExtent);
    
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
    // Use CPU sort for now (debugging GPU sort issues)
    MTL::Buffer* sortedIndices = gpuSort->sortCPU(positionBuffer, camera.get_position(), gaussianCount);
    
    Uniforms uniforms;
    
    static bool renderDebugPrinted = false;
    
    if (useTrainingView && !trainingImages.empty()) {
        const TrainingImage& img = trainingImages[currentTrainingIndex];
        const ColmapCamera& cam = colmapData.cameras.at(img.cameraId);
        
        // Set view matrix from COLMAP pose (must be done BEFORE any transformations)
        uniforms.viewMatrix = viewMatrixFromColmap(img.rotation, img.translation);

        
        // For display, we render to the window, not the training image size
        // Scale focal length proportionally to window size
        float scaleX = (float)windowWidth / (float)cam.width;
        float scaleY = (float)windowHeight / (float)cam.height;
        float scaledFx = cam.fx * scaleX;
        float scaledFy = cam.fy * scaleY;
        
        // Create projection for window size with scaled focal lengths
        float near = 0.1f;
        float far = 1000.0f;
        float cx = cam.cx * scaleX;
        float cy = cam.cy * scaleY;
        
        // COLMAP uses OpenCV convention: +Z forward, Y-down
        // Build projection matrix in column-major format directly (like projectionFromColmap)
        simd_float4x4 proj = {0};
        proj.columns[0][0] = 2.0f * scaledFx / windowWidth;
        proj.columns[1][1] = -2.0f * scaledFy / windowHeight;  // Y flip for COLMAP Y-down
        proj.columns[2][0] = 1.0f - 2.0f * cx / windowWidth;
        proj.columns[2][1] = 2.0f * cy / windowHeight - 1.0f;
        proj.columns[2][2] = far / (far - near);
        proj.columns[2][3] = 1.0f;  // clipW = viewZ (COLMAP +Z forward)
        proj.columns[3][2] = -(far * near) / (far - near);
        uniforms.projectionMatrix = proj;
        
        uniforms.viewProjectionMatrix = matrix_multiply(uniforms.projectionMatrix, uniforms.viewMatrix);
        uniforms.screenSize = simd_make_float2((float)windowWidth, (float)windowHeight);
        uniforms.focalLength = simd_make_float2(scaledFx, scaledFy);
        
        simd_float3x3 R;
        R.columns[0] = uniforms.viewMatrix.columns[0].xyz;
        R.columns[1] = uniforms.viewMatrix.columns[1].xyz;
        R.columns[2] = uniforms.viewMatrix.columns[2].xyz;
        uniforms.cameraPos = -matrix_multiply(simd_transpose(R), img.translation);
        
        if (!renderDebugPrinted) {
            printf("\n=== RENDER DEBUG (Training View) ===\n");
            printf("Window: %dx%d, COLMAP: %dx%d\n", windowWidth, windowHeight, cam.width, cam.height);
            printf("Scale: scaleX=%.4f, scaleY=%.4f\n", scaleX, scaleY);
            printf("Focal: original fx=%.2f fy=%.2f, scaled fx=%.2f fy=%.2f\n", 
                   cam.fx, cam.fy, scaledFx, scaledFy);
            printf("Principal: original cx=%.2f cy=%.2f, scaled cx=%.2f cy=%.2f\n",
                   cam.cx, cam.cy, cx, cy);
            printf("Camera position: (%.3f, %.3f, %.3f)\n", 
                   uniforms.cameraPos.x, uniforms.cameraPos.y, uniforms.cameraPos.z);
            printf("View matrix:\n");
            for (int r = 0; r < 4; r++) {
                printf("  [%.4f %.4f %.4f %.4f]\n",
                       uniforms.viewMatrix.columns[0][r],
                       uniforms.viewMatrix.columns[1][r],
                       uniforms.viewMatrix.columns[2][r],
                       uniforms.viewMatrix.columns[3][r]);
            }
            printf("Projection matrix:\n");
            for (int r = 0; r < 4; r++) {
                printf("  [%.4f %.4f %.4f %.4f]\n",
                       uniforms.projectionMatrix.columns[0][r],
                       uniforms.projectionMatrix.columns[1][r],
                       uniforms.projectionMatrix.columns[2][r],
                       uniforms.projectionMatrix.columns[3][r]);
            }
            
            // Test transform a sample point
            Gaussian* gaussians = (Gaussian*)gaussianBuffer->contents();
            simd_float3 testPos = gaussians[0].position;
            simd_float4 viewPos = matrix_multiply(uniforms.viewMatrix, simd_make_float4(testPos, 1.0f));
            simd_float4 clipPos = matrix_multiply(uniforms.viewProjectionMatrix, simd_make_float4(testPos, 1.0f));
            printf("Sample Gaussian 0:\n");
            printf("  world pos: (%.3f, %.3f, %.3f)\n", testPos.x, testPos.y, testPos.z);
            printf("  view pos: (%.3f, %.3f, %.3f, %.3f)\n", viewPos.x, viewPos.y, viewPos.z, viewPos.w);
            printf("  clip pos: (%.3f, %.3f, %.3f, %.3f)\n", clipPos.x, clipPos.y, clipPos.z, clipPos.w);
            printf("  NDC: (%.3f, %.3f)\n", clipPos.x/clipPos.w, clipPos.y/clipPos.w);
            printf("  depth (z in view): %.3f\n", viewPos.z);
            printf("  scale (log): (%.3f, %.3f, %.3f)\n", 
                   gaussians[0].scale.x, gaussians[0].scale.y, gaussians[0].scale.z);
            printf("  scale (exp): (%.6f, %.6f, %.6f)\n", 
                   exp(gaussians[0].scale.x), exp(gaussians[0].scale.y), exp(gaussians[0].scale.z));
            
            // Compute expected 2D size
            float scale = exp(gaussians[0].scale.x);
            float depth = viewPos.z;
            float size2d = scale * scaledFx / depth;
            printf("  Expected 2D size: scale * fx / depth = %.6f * %.2f / %.3f = %.2f pixels\n",
                   scale, scaledFx, depth, size2d);
            
            renderDebugPrinted = true;
        }
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
        
        if (!renderDebugPrinted && gaussianBuffer) {
            printf("\n=== RENDER DEBUG (Free Camera) ===\n");
            printf("Window: %dx%d\n", windowWidth, windowHeight);
            printf("Focal: fx=%.2f fy=%.2f\n", fx, fy);
            printf("Camera position: (%.3f, %.3f, %.3f)\n", 
                   uniforms.cameraPos.x, uniforms.cameraPos.y, uniforms.cameraPos.z);
            printf("View matrix:\n");
            for (int r = 0; r < 4; r++) {
                printf("  [%.4f %.4f %.4f %.4f]\n",
                       uniforms.viewMatrix.columns[0][r],
                       uniforms.viewMatrix.columns[1][r],
                       uniforms.viewMatrix.columns[2][r],
                       uniforms.viewMatrix.columns[3][r]);
            }
            
            // Test transform a sample point
            Gaussian* gaussians = (Gaussian*)gaussianBuffer->contents();
            simd_float3 testPos = gaussians[0].position;
            simd_float4 viewPos = matrix_multiply(uniforms.viewMatrix, simd_make_float4(testPos, 1.0f));
            simd_float4 clipPos = matrix_multiply(uniforms.viewProjectionMatrix, simd_make_float4(testPos, 1.0f));
            printf("Sample Gaussian 0:\n");
            printf("  world pos: (%.3f, %.3f, %.3f)\n", testPos.x, testPos.y, testPos.z);
            printf("  view pos: (%.3f, %.3f, %.3f, %.3f)\n", viewPos.x, viewPos.y, viewPos.z, viewPos.w);
            printf("  clip pos: (%.3f, %.3f, %.3f, %.3f)\n", clipPos.x, clipPos.y, clipPos.z, clipPos.w);
            if (clipPos.w != 0) {
                printf("  NDC: (%.3f, %.3f)\n", clipPos.x/clipPos.w, clipPos.y/clipPos.w);
            }
            printf("  depth (z in view): %.3f\n", viewPos.z);
            printf("  scale (log): (%.3f, %.3f, %.3f)\n", 
                   gaussians[0].scale.x, gaussians[0].scale.y, gaussians[0].scale.z);
            printf("  scale (exp): (%.6f, %.6f, %.6f)\n", 
                   exp(gaussians[0].scale.x), exp(gaussians[0].scale.y), exp(gaussians[0].scale.z));
            
            // Compute expected 2D size
            float scale = exp(gaussians[0].scale.x);
            float depth = viewPos.z;
            float size2d = scale * fx / depth;
            printf("  Expected 2D size: scale * fx / depth = %.6f * %.2f / %.3f = %.2f pixels\n",
                   scale, fx, depth, size2d);
            
            renderDebugPrinted = true;
        }
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
    // COLMAP loader stores quaternion as: quat = (qx, qy, qz, qw)
    // So quat.x=qx, quat.y=qy, quat.z=qz, quat.w=qw
    float x = quat.x, y = quat.y, z = quat.z, w = quat.w;
    
    // COLMAP stores world-to-camera rotation and camera-space translation
    // p_cam = R * p_world + t
    // So the view matrix is simply [R | t]
    matrix_float3x3 R;
    R.columns[0] = simd_make_float3(1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y));
    R.columns[1] = simd_make_float3(2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x));
    R.columns[2] = simd_make_float3(2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y));
    
    simd_float4x4 view;
    view.columns[0] = simd_make_float4(R.columns[0], 0);
    view.columns[1] = simd_make_float4(R.columns[1], 0);
    view.columns[2] = simd_make_float4(R.columns[2], 0);
    view.columns[3] = simd_make_float4(translation, 1);
    
    return view;
}

simd_float4x4 MTLEngine::projectionFromColmap(const ColmapCamera& cam, float nearZ, float farZ) {
    float fx = cam.fx;
    float fy = cam.fy;
    float cx = cam.cx;
    float cy = cam.cy;
    float w = (float)cam.width;
    float h = (float)cam.height;
    
    // COLMAP uses OpenCV convention: camera looks down +Z axis (objects in front have positive viewZ)
    // COLMAP image Y-axis points DOWN (origin at top-left)
    // Metal NDC: X=[-1,1] left-to-right, Y=[-1,1] bottom-to-top, Z=[0,1] near-to-far
    // We flip Y in projection to match COLMAP's Y-down with Metal's Y-up NDC
    simd_float4x4 proj = {0};
    proj.columns[0][0] = 2.0f * fx / w;
    proj.columns[1][1] = -2.0f * fy / h;  // NEGATIVE to flip Y (COLMAP Y-down -> Metal Y-up)
    proj.columns[2][0] = 1.0f - 2.0f * cx / w;  // X offset
    proj.columns[2][1] = 2.0f * cy / h - 1.0f;  // Y offset (adjusted for flip)
    proj.columns[2][2] = farZ / (farZ - nearZ);
    proj.columns[2][3] = 1.0f;  // POSITIVE: clipW = viewZ (COLMAP has +Z forward)
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
    
    // Get ACTUAL image dimensions from the loaded texture (may differ from COLMAP camera)
    uint32_t actualWidth = img.texture->width();
    uint32_t actualHeight = img.texture->height();
    
    // Scale factor from COLMAP resolution to actual image resolution
    float scaleX = (float)actualWidth / (float)cam.width;
    float scaleY = (float)actualHeight / (float)cam.height;
    
    // Scale camera intrinsics to match actual image size
    float scaledFx = cam.fx * scaleX;
    float scaledFy = cam.fy * scaleY;
    float scaledCx = cam.cx * scaleX;
    float scaledCy = cam.cy * scaleY;
    
    // DEBUG: Print uniforms once
    static bool uniformsDebugPrinted = false;
    if (!uniformsDebugPrinted) {
        std::cout << "=== Training Uniforms Debug ===" << std::endl;
        std::cout << "COLMAP camera size: " << cam.width << "x" << cam.height << std::endl;
        std::cout << "Actual texture size: " << actualWidth << "x" << actualHeight << std::endl;
        std::cout << "Scale factor: " << scaleX << "x" << scaleY << std::endl;
        std::cout << "Original focal: fx=" << cam.fx << " fy=" << cam.fy << std::endl;
        std::cout << "Scaled focal: fx=" << scaledFx << " fy=" << scaledFy << std::endl;
        std::cout << "Original principal: cx=" << cam.cx << " cy=" << cam.cy << std::endl;
        std::cout << "Scaled principal: cx=" << scaledCx << " cy=" << scaledCy << std::endl;
        uniformsDebugPrinted = true;
    }
    
    if (!renderTarget || renderTarget->width() != actualWidth || renderTarget->height() != actualHeight) {
        createRenderTarget(actualWidth, actualHeight);
    }
    
    // Create scaled camera for projection
    ColmapCamera scaledCam = cam;
    scaledCam.width = actualWidth;
    scaledCam.height = actualHeight;
    scaledCam.fx = scaledFx;
    scaledCam.fy = scaledFy;
    scaledCam.cx = scaledCx;
    scaledCam.cy = scaledCy;
    
    TiledUniforms uniforms;
    uniforms.viewMatrix = viewMatrixFromColmap(img.rotation, img.translation);
    uniforms.projectionMatrix = projectionFromColmap(scaledCam, 0.1f, 1000.0f);
    uniforms.viewProjectionMatrix = matrix_multiply(uniforms.projectionMatrix, uniforms.viewMatrix);
    uniforms.screenSize = simd_make_float2((float)actualWidth, (float)actualHeight);
    uniforms.focalLength = simd_make_float2(scaledFx, scaledFy);
    
    simd_float3x3 R;
    R.columns[0] = uniforms.viewMatrix.columns[0].xyz;
    R.columns[1] = uniforms.viewMatrix.columns[1].xyz;
    R.columns[2] = uniforms.viewMatrix.columns[2].xyz;
    uniforms.cameraPos = -matrix_multiply(simd_transpose(R), img.translation);
    
    // DEBUG: Check view-space coordinates for first Gaussian
    static bool viewDebugPrinted = false;
    if (!viewDebugPrinted) {
        Gaussian* gs = (Gaussian*)gaussianBuffer->contents();
        simd_float4 worldPos = simd_make_float4(gs[0].position, 1.0f);
        simd_float4 viewPos = matrix_multiply(uniforms.viewMatrix, worldPos);
        simd_float4 clipPos = matrix_multiply(uniforms.viewProjectionMatrix, worldPos);
        simd_float3 ndc = simd_make_float3(clipPos.x/clipPos.w, clipPos.y/clipPos.w, clipPos.z/clipPos.w);
        simd_float2 screenPos = simd_make_float2((ndc.x * 0.5f + 0.5f) * actualWidth,
                                                  (ndc.y * 0.5f + 0.5f) * actualHeight);
        std::cout << "=== View Transform Debug ===" << std::endl;
        std::cout << "World pos: (" << worldPos.x << ", " << worldPos.y << ", " << worldPos.z << ")" << std::endl;
        std::cout << "View pos: (" << viewPos.x << ", " << viewPos.y << ", " << viewPos.z << ")" << std::endl;
        std::cout << "Clip pos: (" << clipPos.x << ", " << clipPos.y << ", " << clipPos.z << ", w=" << clipPos.w << ")" << std::endl;
        std::cout << "NDC: (" << ndc.x << ", " << ndc.y << ", " << ndc.z << ")" << std::endl;
        std::cout << "Screen pos: (" << screenPos.x << ", " << screenPos.y << ")" << std::endl;
        std::cout << "Screen size: " << actualWidth << "x" << actualHeight << std::endl;
        viewDebugPrinted = true;
    }
    
    // Forward pass
    tiledRasterizer->forward(commandQueue, gaussianBuffer, gaussianCount, uniforms, renderTarget);
    
    // DEBUG: Save training render less frequently
    static int saveCounter = 0;
    if (saveCounter % 500 == 0) {
        char filename[64];
        snprintf(filename, sizeof(filename), "/tmp/train_render_%04d.ppm", saveCounter);
        saveTextureToPPM(renderTarget, metalDevice, commandQueue, filename);
        
        // Also save ground truth for comparison
        char gtFilename[64];
        snprintf(gtFilename, sizeof(gtFilename), "/tmp/ground_truth_%04d.ppm", saveCounter);
        saveTextureToPPM(img.texture, metalDevice, commandQueue, gtFilename);
    }
    saveCounter++;
    
    // Compute loss
    float loss = computeLoss(renderTarget, img.texture);
    
    // Backward pass
    tiledRasterizer->backward(commandQueue, gaussianBuffer, gaussianGradients, gaussianCount,
                              uniforms, renderTarget, img.texture);
    
    // DEBUG: Print gradient magnitudes once
    static int gradDebugCount = 0;
    if (gradDebugCount < 3) {
        GaussianGradients* grads = (GaussianGradients*)gaussianGradients->contents();
        float sumPosGrad = 0, sumOpacGrad = 0, sumSHGrad = 0, sumScaleGrad = 0;
        int nonZeroCount = 0;
        for (size_t i = 0; i < std::min(gaussianCount, (size_t)1000); i++) {
            float posGrad = fabs(grads[i].position_x) + fabs(grads[i].position_y) + fabs(grads[i].position_z);
            float scaleGrad = fabs(grads[i].scale_x) + fabs(grads[i].scale_y) + fabs(grads[i].scale_z);
            sumPosGrad += posGrad;
            sumOpacGrad += fabs(grads[i].opacity);
            sumSHGrad += fabs(grads[i].sh[0]) + fabs(grads[i].sh[4]) + fabs(grads[i].sh[8]);
            sumScaleGrad += scaleGrad;
            if (posGrad > 0.0001) nonZeroCount++;
        }
        std::cout << "\n=== Gradient Debug (iter " << gradDebugCount << ") ===" << std::endl;
        std::cout << "Avg position grad: " << sumPosGrad / 1000 << std::endl;
        std::cout << "Avg opacity grad: " << sumOpacGrad / 1000 << std::endl;
        std::cout << "Avg SH grad: " << sumSHGrad / 1000 << std::endl;
        std::cout << "Avg scale grad: " << sumScaleGrad / 1000 << std::endl;
        std::cout << "Non-zero grads in first 1000: " << nonZeroCount << std::endl;
        
        // Print first non-zero gradient
        for (size_t i = 0; i < std::min(gaussianCount, (size_t)100); i++) {
            if (fabs(grads[i].position_x) > 0.0001 || fabs(grads[i].opacity) > 0.0001) {
                std::cout << "Gaussian " << i << " grad: pos=(" << grads[i].position_x << "," 
                          << grads[i].position_y << "," << grads[i].position_z 
                          << ") opacity=" << grads[i].opacity 
                          << " sh0=" << grads[i].sh[0] << std::endl;
                break;
            }
        }
        gradDebugCount++;
    }
    
    // Accumulate for density control
    densityController->accumulateGradients(commandQueue, gaussianGradients, gaussianCount);
    
    // Optimizer step - official 3DGS learning rates
    optimizer->step(commandQueue, gaussianBuffer, gaussianGradients,
                    0.00016f,  // position lr (official default)
                    0.005f,    // scale lr (official default)
                    0.01f,      // rotation lr (OFF - debugging)
                    0.05f,     // opacity lr (ENABLED - needed to make Gaussians visible)
                    0.000f);   // sh lr (OFF for testing)
//    Gaussian* g = (Gaussian*)gaussianBuffer->contents();
//    float minSH = 1e10, maxSH = -1e10;
//    for (size_t i = 0; i < std::min(gaussianCount, (size_t)1000); i++) {
//        for (int j = 0; j < 12; j++) {
//            minSH = std::min(minSH, g[i].sh[j]);
//            maxSH = std::max(maxSH, g[i].sh[j]);
//        }
//    }
//    std::cout << "SH range: [" << minSH << ", " << maxSH << "]" << std::endl;

    Gaussian* g = (Gaussian*)gaussianBuffer->contents();
    float avgOpacity = 0, avgZ = 0;
    for (int i = 0; i < 100; i++) {
        avgOpacity += 1.0f / (1.0f + exp(-g[i].opacity));  // sigmoid
        avgZ += g[i].position.z;
    }
    std::cout << "Avg opacity: " << avgOpacity/100 << " Avg Z: " << avgZ/100 << std::endl;

    
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
    
    // Debug: Check color distribution after training
    std::cout << "\n=== Color Debug After Training ===" << std::endl;
    Gaussian* gaussians = (Gaussian*)gaussianBuffer->contents();
    
    float minSH0 = FLT_MAX, maxSH0 = -FLT_MAX;
    float minSH4 = FLT_MAX, maxSH4 = -FLT_MAX;
    float minSH8 = FLT_MAX, maxSH8 = -FLT_MAX;
    int validCount = 0;
    float sumR = 0, sumG = 0, sumB = 0;
    
    for (size_t i = 0; i < gaussianCount; i++) {
        float sh0 = gaussians[i].sh[0];
        float sh4 = gaussians[i].sh[4];
        float sh8 = gaussians[i].sh[8];
        
        if (!std::isnan(sh0) && !std::isnan(sh4) && !std::isnan(sh8)) {
            minSH0 = std::min(minSH0, sh0);
            maxSH0 = std::max(maxSH0, sh0);
            minSH4 = std::min(minSH4, sh4);
            maxSH4 = std::max(maxSH4, sh4);
            minSH8 = std::min(minSH8, sh8);
            maxSH8 = std::max(maxSH8, sh8);
            
            // Compute color: color = SH_C0 * sh_dc + 0.5
            const float SH_C0 = 0.28209479f;
            float r = SH_C0 * sh0 + 0.5f;
            float g = SH_C0 * sh4 + 0.5f;
            float b = SH_C0 * sh8 + 0.5f;
            
            sumR += std::clamp(r, 0.0f, 1.0f);
            sumG += std::clamp(g, 0.0f, 1.0f);
            sumB += std::clamp(b, 0.0f, 1.0f);
            validCount++;
        }
    }
    
    std::cout << "SH[0] (R DC) range: [" << minSH0 << ", " << maxSH0 << "]" << std::endl;
    std::cout << "SH[4] (G DC) range: [" << minSH4 << ", " << maxSH4 << "]" << std::endl;
    std::cout << "SH[8] (B DC) range: [" << minSH8 << ", " << maxSH8 << "]" << std::endl;
    std::cout << "Average color (RGB): (" << sumR/validCount << ", " << sumG/validCount << ", " << sumB/validCount << ")" << std::endl;
    
    // Print a few sample Gaussians
    std::cout << "\nSample Gaussian Colors:" << std::endl;
    for (int i = 0; i < 5 && i < (int)gaussianCount; i++) {
        const float SH_C0 = 0.28209479f;
        float r = std::clamp(SH_C0 * gaussians[i].sh[0] + 0.5f, 0.0f, 1.0f);
        float g = std::clamp(SH_C0 * gaussians[i].sh[4] + 0.5f, 0.0f, 1.0f);
        float b = std::clamp(SH_C0 * gaussians[i].sh[8] + 0.5f, 0.0f, 1.0f);
        std::cout << "  Gaussian " << i << ": SH=(" << gaussians[i].sh[0] << ", " << gaussians[i].sh[4] << ", " << gaussians[i].sh[8]
                  << ") -> Color=(" << r << ", " << g << ", " << b << ")" << std::endl;
    }
}

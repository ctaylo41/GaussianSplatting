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
#include <Foundation/NSAutoreleasePool.hpp>

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

void MTLEngine::loadGaussians(const std::vector<Gaussian>& gaussians, float sceneExtent) {
    gaussianCount = gaussians.size();
    gaussianBuffer = metalDevice->newBuffer(gaussians.data(), gaussianCount * sizeof(Gaussian),
                                            MTL::ResourceStorageModeShared);
    
    // Create position buffer for sorting
    positionBuffer = metalDevice->newBuffer(gaussianCount * sizeof(simd_float3),
                                            MTL::ResourceStorageModeShared);
    simd_float3* positions = (simd_float3*)positionBuffer->contents();
    for (size_t i = 0; i < gaussianCount; i++) {
        positions[i] = gaussians[i].position;
    }
    
    // Use the camera-based scene extent for density control
    if (sceneExtent <= 0.0f) {
        // Fallback: compute from point cloud (not ideal)
        float min_x = FLT_MAX, max_x = -FLT_MAX;
        float min_y = FLT_MAX, max_y = -FLT_MAX;
        float min_z = FLT_MAX, max_z = -FLT_MAX;
        for (const auto& g : gaussians) {
            min_x = std::min(min_x, g.position.x); max_x = std::max(max_x, g.position.x);
            min_y = std::min(min_y, g.position.y); max_y = std::max(max_y, g.position.y);
            min_z = std::min(min_z, g.position.z); max_z = std::max(max_z, g.position.z);
        }
        sceneExtent = sqrtf((max_x-min_x)*(max_x-min_x) + 
                           (max_y-min_y)*(max_y-min_y) + 
                           (max_z-min_z)*(max_z-min_z));
        std::cout << "WARNING: Using point cloud extent (fallback): " << sceneExtent << std::endl;
    }
    
    std::cout << "Density control using scene extent: " << sceneExtent << std::endl;
    DensityController::setSceneExtent(sceneExtent);
    
    gpuSort = new GPURadixSort32(metalDevice, shaderLibrary, 2000000);
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
    // GPU radix sort for depth ordering
    MTL::Buffer* sortedIndices = gpuSort->sort(commandQueue, positionBuffer, camera.get_position(), gaussianCount);
    
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
        // Projection matrix matching projectionFromColmap() for consistency
        // COLMAP Y-down + Metal texture top-left origin = NO Y flip needed
        simd_float4x4 proj = {0};
        proj.columns[0][0] = 2.0f * scaledFx / windowWidth;
        proj.columns[1][1] = 2.0f * scaledFy / windowHeight;  // POSITIVE: matches projectionFromColmap()
        proj.columns[2][0] = 2.0f * cx / windowWidth - 1.0f;  // Consistent offset formula
        proj.columns[2][1] = 2.0f * cy / windowHeight - 1.0f; // Consistent offset formula
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
    renderPassDesc->colorAttachments()->object(0)->setClearColor(MTL::ClearColor(1.0, 1.0, 1.0, 1.0));
    
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
    // Quaternion convention: float4(.x=w, .y=x, .z=y, .w=z)
    // This matches the Gaussian rotation convention used throughout the codebase
    float w = quat.x, x = quat.y, y = quat.z, z = quat.w;
    
    // COLMAP stores world-to-camera rotation R and camera-space translation t
    // Transform: p_cam = R * p_world + t
    //
    // Standard quaternion to rotation matrix (R):
    // R = | 1-2(y²+z²)   2(xy-wz)    2(xz+wy)  |
    //     | 2(xy+wz)    1-2(x²+z²)   2(yz-wx)  |
    //     | 2(xz-wy)     2(yz+wx)   1-2(x²+y²) |
    //
    // Metal uses column-major matrices and (matrix * vector) computes:
    //   result[i] = dot(row i of matrix, vector)
    // where row i of matrix = (col0[i], col1[i], col2[i], col3[i])
    //
    // For viewPos = viewMatrix * worldPos to compute R * worldPos + t:
    // We need row 0 of viewMatrix to be row 0 of [R|t] = (R[0][0], R[0][1], R[0][2], t.x)
    // In column-major storage: col0[0]=R[0][0], col1[0]=R[0][1], col2[0]=R[0][2], col3[0]=t.x
    //
    // So column j of viewMatrix should contain R[*][j] (j-th column of R) in its first 3 elements
    // Wait no - we want row i of viewMatrix = row i of [R|t]
    // Metal column major: row i is (col0[i], col1[i], col2[i], col3[i])
    // So we need: col0[i] = R[i][0], col1[i] = R[i][1], col2[i] = R[i][2], col3[i] = t[i]
    // This means: col j = (R[0][j], R[1][j], R[2][j], 0 or 1) = column j of R
    //
    // Build R (column-major, so R.columns[j] = j-th column of rotation matrix):
    matrix_float3x3 R;
    // Column 0 of R: (R[0][0], R[1][0], R[2][0])
    R.columns[0] = simd_make_float3(1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y));
    // Column 1 of R: (R[0][1], R[1][1], R[2][1])
    R.columns[1] = simd_make_float3(2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x));
    // Column 2 of R: (R[0][2], R[1][2], R[2][2])
    R.columns[2] = simd_make_float3(2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y));
    
    // Build 4x4 view matrix [R | t]
    // For (matrix * vec4(pos, 1)) = R*pos + t:
    // We need the 4x4 matrix such that row i = (R[i][0], R[i][1], R[i][2], t[i])
    // In column major: col0 = (R[0][0], R[1][0], R[2][0], 0) = (R.columns[0], 0)... NO!
    //
    // Let me think again. In column-major:
    //   viewMatrix.columns[j][i] = element at row i, col j
    //
    // We want: viewMatrix * vec4(p, 1) = vec4(R*p + t, 1)
    //   result[0] = row0 · input = col0[0]*p.x + col1[0]*p.y + col2[0]*p.z + col3[0]*1
    //             = R[0][0]*p.x + R[0][1]*p.y + R[0][2]*p.z + t.x  ✓
    //
    // So we need: col0[0] = R[0][0], col1[0] = R[0][1], col2[0] = R[0][2], col3[0] = t.x
    //             col0[1] = R[1][0], col1[1] = R[1][1], col2[1] = R[1][2], col3[1] = t.y
    //             etc.
    //
    // This means: columns[j] = (R[0][j], R[1][j], R[2][j], 0) for j=0,1,2
    //             columns[3] = (t.x, t.y, t.z, 1)
    //
    // And R.columns[j] = (R[0][j], R[1][j], R[2][j]) which IS the j-th column of R matrix!
    // So the original code was correct! Let me verify the quaternion formula...
    //
    // Standard quaternion to matrix (row-major R[row][col]):
    //   R[0][0] = 1 - 2*(y*y + z*z)
    //   R[0][1] = 2*(x*y - w*z)
    //   R[0][2] = 2*(x*z + w*y)
    //   R[1][0] = 2*(x*y + w*z)
    //   R[1][1] = 1 - 2*(x*x + z*z)
    //   R[1][2] = 2*(y*z - w*x)
    //   R[2][0] = 2*(x*z - w*y)
    //   R[2][1] = 2*(y*z + w*x)
    //   R[2][2] = 1 - 2*(x*x + y*y)
    //
    // Column 0 should be (R[0][0], R[1][0], R[2][0]) = (1-2(y²+z²), 2(xy+wz), 2(xz-wy)) ✓
    //
    // OK the rotation matrix is correct. The issue must be elsewhere!
    
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
    
    // COLMAP projection: pixel_x = fx * X/Z + cx, pixel_y = fy * Y/Z + cy
    // We want: screen_pos = (ndc * 0.5 + 0.5) * size  to equal COLMAP pixel coordinates
    // 
    // For X: ndc_x * 0.5 + 0.5 = (fx * X/Z + cx) / w
    //        ndc_x = 2 * fx * X / (w * Z) + (2*cx/w - 1)
    //
    // For Y: ndc_y * 0.5 + 0.5 = (fy * Y/Z + cy) / h
    //        ndc_y = 2 * fy * Y / (h * Z) + (2*cy/h - 1)
    //
    // Matrix form: clip = P * view, ndc = clip.xyz / clip.w
    // With clip.w = Z (for COLMAP's +Z forward convention):
    //   ndc_x = clip.x / Z = (P[0][0] * X + P[2][0] * Z) / Z = P[0][0] * X/Z + P[2][0]
    //   ndc_y = clip.y / Z = (P[1][1] * Y + P[2][1] * Z) / Z = P[1][1] * Y/Z + P[2][1]
    //
    // Therefore: P[0][0] = 2*fx/w, P[2][0] = 2*cx/w - 1
    //            P[1][1] = 2*fy/h, P[2][1] = 2*cy/h - 1
    //            (NO NEGATIVE on fy - COLMAP Y-down matches Metal texture top-left origin)
    
    simd_float4x4 proj = {0};
    proj.columns[0][0] = 2.0f * fx / w;
    proj.columns[1][1] = 2.0f * fy / h;  // POSITIVE: COLMAP Y-down, Metal texture origin top-left
    proj.columns[2][0] = 2.0f * cx / w - 1.0f;
    proj.columns[2][1] = 2.0f * cy / h - 1.0f;
    proj.columns[2][2] = farZ / (farZ - nearZ);
    proj.columns[2][3] = 1.0f;  // clip.w = view.z (COLMAP +Z forward)
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

float MTLEngine::trainStep(size_t imageIndex, 
                           float lr_position,
                           float lr_scale,
                           float lr_rotation,
                           float lr_opacity,
                           float lr_sh) {
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
    
    // DEBUG: Check view-space coordinates for first few Gaussians on CPU side
    static bool projectionDebugPrinted = false;
    if (!projectionDebugPrinted) {
        Gaussian* gaussians = (Gaussian*)gaussianBuffer->contents();
        printf("\n=== CPU Projection Debug ===\n");
        printf("uniforms.viewMatrix:\n");
        for (int r = 0; r < 4; r++) {
            printf("  [%.4f %.4f %.4f %.4f]\n",
                   uniforms.viewMatrix.columns[0][r],
                   uniforms.viewMatrix.columns[1][r],
                   uniforms.viewMatrix.columns[2][r],
                   uniforms.viewMatrix.columns[3][r]);
        }
        printf("uniforms.projectionMatrix:\n");
        for (int r = 0; r < 4; r++) {
            printf("  [%.4f %.4f %.4f %.4f]\n",
                   uniforms.projectionMatrix.columns[0][r],
                   uniforms.projectionMatrix.columns[1][r],
                   uniforms.projectionMatrix.columns[2][r],
                   uniforms.projectionMatrix.columns[3][r]);
        }
        printf("screenSize: (%.0f, %.0f)\n", uniforms.screenSize.x, uniforms.screenSize.y);
        printf("focalLength: (%.2f, %.2f)\n", uniforms.focalLength.x, uniforms.focalLength.y);
        
        printf("\nFirst 5 Gaussians:\n");
        for (int i = 0; i < 5 && i < (int)gaussianCount; i++) {
            simd_float3 pos = gaussians[i].position;
            simd_float4 worldPos = simd_make_float4(pos.x, pos.y, pos.z, 1.0f);
            simd_float4 viewPos = matrix_multiply(uniforms.viewMatrix, worldPos);
            simd_float4 clipPos = matrix_multiply(uniforms.viewProjectionMatrix, worldPos);
            
            printf("  [%d] world=(%.2f,%.2f,%.2f) view=(%.2f,%.2f,%.2f) clip.w=%.2f\n",
                   i, pos.x, pos.y, pos.z, viewPos.x, viewPos.y, viewPos.z, clipPos.w);
            
            if (clipPos.w > 0.1f && viewPos.z > 0.1f) {
                simd_float3 ndc = simd_make_float3(clipPos.x/clipPos.w, clipPos.y/clipPos.w, clipPos.z/clipPos.w);
                printf("       ndc=(%.3f,%.3f) in frustum: %s\n", 
                       ndc.x, ndc.y, 
                       (fabs(ndc.x) < 1.2f && fabs(ndc.y) < 1.2f) ? "YES" : "NO");
            } else {
                printf("       BEHIND camera or clipped\n");
            }
        }
        projectionDebugPrinted = true;
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
    
    // Accumulate for density control
    densityController->accumulateGradients(commandQueue, gaussianGradients, gaussianCount);
    
    // Optimizer step with decayed learning rates
    optimizer->step(commandQueue, gaussianBuffer, gaussianGradients,
                    lr_position,   // decayed position lr
                    lr_scale,      // scale lr (official: scaling_lr = 0.005)
                    lr_rotation,   // rotation lr (official: rotation_lr = 0.001)
                    lr_opacity,    // opacity lr (official: opacity_lr = 0.025)
                    lr_sh);        // sh lr (official: feature_lr = 0.0025)
    
    // Periodic stats every 200 images
    if (saveCounter % 200 == 0) {
        Gaussian* g = (Gaussian*)gaussianBuffer->contents();
        float avgOpacity = 0, avgScale = 0;
        const int sampleCount = std::min((size_t)100, gaussianCount);
        for (int i = 0; i < sampleCount; i++) {
            avgOpacity += 1.0f / (1.0f + exp(-g[i].opacity));
            avgScale += (exp(g[i].scale.x) + exp(g[i].scale.y) + exp(g[i].scale.z)) / 3.0f;
        }
        printf("\n[iter %d] Gaussians: %zu | Avg opacity: %.3f | Avg scale: %.4f\n",
               saveCounter, gaussianCount, avgOpacity / sampleCount, avgScale / sampleCount);
    }
    
    return loss;
}

void MTLEngine::updatePositionBuffer() {
    Gaussian* gaussians = (Gaussian*)gaussianBuffer->contents();
    simd_float3* positions = (simd_float3*)positionBuffer->contents();
    
    for (size_t i = 0; i < gaussianCount; i++) {
        positions[i] = gaussians[i].position;
    }
}

// Exponential learning rate decay (matches official 3DGS)
static float exponentialLRDecay(float lr_init, float lr_final, size_t currentIter, size_t maxIter) {
    if (currentIter >= maxIter) return lr_final;
    float t = static_cast<float>(currentIter) / static_cast<float>(maxIter);
    return lr_init * std::pow(lr_final / lr_init, t);
}

void MTLEngine::train(size_t numEpochs) {
    std::cout << "Starting training for " << numEpochs << " epochs..." << std::endl;
    std::cout << "Gaussians: " << gaussianCount << " | Images: " << trainingImages.size() << std::endl;
    
    // Training configuration (matching official 3DGS)
    const size_t OPACITY_RESET_INTERVAL = 3000;
    const size_t DENSIFY_FROM_ITER = 500;
    const size_t DENSIFY_UNTIL_ITER = 15000;
    const float OPACITY_RESET_VALUE = -4.6f;  // sigmoid^-1(0.01) ≈ -4.6
    
    // ============================================================
    // LEARNING RATE CONFIGURATION (matching official 3DGS)
    // ============================================================
    // Position LR decays exponentially from init to final
    const float POSITION_LR_INIT = 0.00016f;
    const float POSITION_LR_FINAL = 0.0000016f;  // 100x smaller at end
    const float SCALE_LR = 0.001f;  // Reduced from 0.005 to prevent scale bloating
    const float ROTATION_LR = 0.001f;
    const float OPACITY_LR = 0.05f;
    const float SH_LR = 0.0025f;
    
    // Total iterations for LR scheduling
    const size_t totalExpectedIters = numEpochs * trainingImages.size();
    
    std::cout << "Learning rate decay: position " << POSITION_LR_INIT 
              << " -> " << POSITION_LR_FINAL << " over " << totalExpectedIters << " iterations" << std::endl;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    size_t totalIterations = 0;
    
    for (size_t epoch = 0; epoch < numEpochs; epoch++) {
        float epochLoss = 0.0f;
        auto epochStart = std::chrono::high_resolution_clock::now();
        
        for (size_t imgIdx = 0; imgIdx < trainingImages.size(); imgIdx++) {
            // Autorelease pool to prevent memory buildup from temporary Metal objects
            NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
            
            // Compute decayed learning rates
            float currentPositionLR = exponentialLRDecay(POSITION_LR_INIT, POSITION_LR_FINAL, 
                                                          totalIterations, totalExpectedIters);
            
            float loss = trainStep(imgIdx, currentPositionLR, SCALE_LR, ROTATION_LR, OPACITY_LR, SH_LR);
            epochLoss += loss;
            totalIterations++;
            
            // Progress update every 20 images
            if (imgIdx % 20 == 0) {
                std::cout << "\rEpoch " << epoch << " [" << imgIdx << "/" << trainingImages.size()
                          << "] Loss: " << (epochLoss / (imgIdx + 1)) << std::flush;
            }
            
            // ============================================================
            // OPACITY RESET (Critical for preventing floaters)
            // Official 3DGS resets opacity every 3000 iterations until iter 15000
            // Uses clamping: min(current_opacity, 0.01) - keeps low opacities unchanged
            // ============================================================
            if (totalIterations % OPACITY_RESET_INTERVAL == 0 && 
                totalIterations > 0 && 
                totalIterations < DENSIFY_UNTIL_ITER) {
                std::cout << std::endl << "Resetting opacity at iteration " << totalIterations << std::endl;
                
                Gaussian* gaussians = (Gaussian*)gaussianBuffer->contents();
                for (size_t i = 0; i < gaussianCount; i++) {
                    // Clamp opacity to max 0.01: min(current, sigmoid^-1(0.01))
                    // This keeps already-low opacities unchanged but caps high ones
                    if (gaussians[i].opacity > OPACITY_RESET_VALUE) {
                        gaussians[i].opacity = OPACITY_RESET_VALUE;
                    }
                }
                
                // Reset optimizer momentum for opacity to allow fresh learning
                optimizer->resetOpacityMomentum();
            }
            
            // ============================================================
            // DENSITY CONTROL
            // Only run between iterations 500 and 15000
            // Skip 500 iterations after opacity reset to let Gaussians recover
            // ============================================================
            bool inOpacityRecoveryPeriod = (totalIterations >= OPACITY_RESET_INTERVAL &&
                                            totalIterations <= OPACITY_RESET_INTERVAL + 500);
            bool shouldDensify = (totalIterations >= DENSIFY_FROM_ITER &&
                                  totalIterations < DENSIFY_UNTIL_ITER &&
                                  totalIterations % densityControlInterval == 0 &&
                                  !inOpacityRecoveryPeriod);
            
            if (shouldDensify) {
                densityController->apply(commandQueue, gaussianBuffer, positionBuffer,
                                         nullptr, gaussianCount, totalIterations);
                
                // Resize gradient buffer if needed
                if (gaussianGradients->length() < gaussianCount * sizeof(GaussianGradients)) {
                    gaussianGradients->release();
                    gaussianGradients = metalDevice->newBuffer(gaussianCount * sizeof(GaussianGradients),
                                                               MTL::ResourceStorageModeShared);
                }
                
                // Resize optimizer buffers if needed
                if (optimizer) {
                    optimizer->resizeIfNeeded(gaussianCount);
                }
            }
            
            pool->release();
        }
        
        updatePositionBuffer();
        
        auto epochEnd = std::chrono::high_resolution_clock::now();
        auto epochDuration = std::chrono::duration_cast<std::chrono::seconds>(epochEnd - epochStart).count();
        
        // Compute current position LR for logging
        float currentLR = exponentialLRDecay(POSITION_LR_INIT, POSITION_LR_FINAL, 
                                             totalIterations, totalExpectedIters);
        
        std::cout << std::endl << "=== Epoch " << epoch << " | Loss: "
                  << (epochLoss / trainingImages.size())
                  << " | Gaussians: " << gaussianCount
                  << " | pos_lr: " << currentLR
                  << " | Time: " << epochDuration << "s ===" << std::endl;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    
    std::cout << "Training complete! Total time: " << totalDuration << "s" << std::endl;
}

void MTLEngine::exportTrainingViews(const std::string& outputFolder) {
    if (trainingImages.empty()) {
        std::cerr << "No training images to export views for!" << std::endl;
        return;
    }
    
    // Create output folder if it doesn't exist
    std::string mkdirCmd = "mkdir -p \"" + outputFolder + "\"";
    system(mkdirCmd.c_str());
    
    std::cout << "\n=== Exporting " << trainingImages.size() << " training views to " << outputFolder << " ===" << std::endl;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    for (size_t imgIdx = 0; imgIdx < trainingImages.size(); imgIdx++) {
        const TrainingImage& img = trainingImages[imgIdx];
        const ColmapCamera& cam = colmapData.cameras.at(img.cameraId);
        
        // Create render target matching the original image size
        if (!renderTarget || renderTarget->width() != cam.width || renderTarget->height() != cam.height) {
            createRenderTarget(cam.width, cam.height);
        }
        
        // Set up uniforms for this view (TiledUniforms for tiled rasterizer)
        TiledUniforms uniforms;
        uniforms.viewMatrix = viewMatrixFromColmap(img.rotation, img.translation);
        
        float near = 0.1f;
        float far = 1000.0f;
        
        // Projection matching training
        simd_float4x4 proj = {0};
        proj.columns[0][0] = 2.0f * cam.fx / cam.width;
        proj.columns[1][1] = 2.0f * cam.fy / cam.height;
        proj.columns[2][0] = 2.0f * cam.cx / cam.width - 1.0f;
        proj.columns[2][1] = 2.0f * cam.cy / cam.height - 1.0f;
        proj.columns[2][2] = far / (far - near);
        proj.columns[2][3] = 1.0f;
        proj.columns[3][2] = -(far * near) / (far - near);
        uniforms.projectionMatrix = proj;
        
        uniforms.viewProjectionMatrix = matrix_multiply(uniforms.projectionMatrix, uniforms.viewMatrix);
        uniforms.screenSize = simd_make_float2((float)cam.width, (float)cam.height);
        uniforms.focalLength = simd_make_float2(cam.fx, cam.fy);
        
        simd_float3x3 R;
        R.columns[0] = uniforms.viewMatrix.columns[0].xyz;
        R.columns[1] = uniforms.viewMatrix.columns[1].xyz;
        R.columns[2] = uniforms.viewMatrix.columns[2].xyz;
        uniforms.cameraPos = -matrix_multiply(simd_transpose(R), img.translation);
        
        // Set tile info
        uniforms.numTilesX = (cam.width + 15) / 16;
        uniforms.numTilesY = (cam.height + 15) / 16;
        uniforms.numGaussians = (uint32_t)gaussianCount;
        
        // Render
        tiledRasterizer->forward(commandQueue, gaussianBuffer, gaussianCount, uniforms, renderTarget);
        
        // Generate filename from image ID
        char filename[64];
        snprintf(filename, sizeof(filename), "image_%04u_render.ppm", img.imageId);
        
        std::string outputPath = outputFolder + "/" + filename;
        saveTextureToPPM(renderTarget, metalDevice, commandQueue, outputPath.c_str());
        
        // Progress
        if (imgIdx % 10 == 0 || imgIdx == trainingImages.size() - 1) {
            std::cout << "\rExporting views: " << (imgIdx + 1) << "/" << trainingImages.size() << std::flush;
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    
    std::cout << std::endl << "Exported " << trainingImages.size() << " views in " << duration << "s" << std::endl;
    std::cout << "Output folder: " << outputFolder << std::endl;
}

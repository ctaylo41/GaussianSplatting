//
//  mtl_engine.hpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-24.
//


#include "mtl_engine.hpp"
#include "gpu_sort.hpp"
#include <iostream>
#include "image_loader.hpp"
#include "gradients.hpp"
#include <chrono>
#include <fstream>
#include <Foundation/NSAutoreleasePool.hpp>

// For debugging save a texture to PPM file
void saveTextureToPPM(MTL::Texture* texture, MTL::Device* device, MTL::CommandQueue* queue, const char* filename) {
    uint32_t width = texture->width();
    uint32_t height = texture->height();
    // RGBA
    size_t bytesPerPixel = 4;  
    size_t bytesPerRow = width * bytesPerPixel;
    size_t totalBytes = bytesPerRow * height;
    
    // Create a shared buffer to read back the texture
    MTL::Buffer* readbackBuffer = device->newBuffer(totalBytes, MTL::ResourceStorageModeShared);
    
    // Copy texture to buffer
    MTL::CommandBuffer* cmdBuffer = queue->commandBuffer();
    MTL::BlitCommandEncoder* blit = cmdBuffer->blitCommandEncoder();
    blit->copyFromTexture(texture, 0, 0, MTL::Origin(0, 0, 0), MTL::Size(width, height, 1),
                          readbackBuffer, 0, bytesPerRow, 0);
    blit->endEncoding();
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();
    
    // Access buffer data
    uint8_t* data = (uint8_t*)readbackBuffer->contents();
    
    // Write PPM file
    std::ofstream file(filename, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    
    // Convert RGBA float to RGB bytes
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            size_t offset = y * bytesPerRow + x * bytesPerPixel;
            // R
            file.put(data[offset]);    
            // G
            file.put(data[offset + 1]); 
            // B
            file.put(data[offset + 2]); 
        }
    }
    
    // Cleanup
    file.close();
    readbackBuffer->release();
    printf("Saved render to %s (%dx%d)\n", filename, width, height);
}

// Initialize the Metal engine
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

// Initialize the Metal engine in headless mode for training
void MTLEngine::initHeadless() {
    initDevice();
    initCommandQueue();
    loadShaders();
    createPipeline();
    createLossPipeline();
    uniformBuffer = metalDevice->newBuffer(sizeof(Uniforms), MTL::ResourceStorageModeShared);
}

// Main loop to run the engine
void MTLEngine::run(Camera& camera) {
    activeCamera = &camera;
    size_t totalIterations = 0;
    
    // Window loop
    while (!glfwWindowShouldClose(glfwWindow)) {
        glfwPollEvents();
        
        // Training step
        if (isTraining && !trainingImages.empty()) {
            float loss = trainStep(currentImageIdx);
            epochLoss += loss;
            epochIterations++;
            totalIterations++;
            
            // Display progress every 10 iterations
            if (epochIterations % 10 == 0) {
                std::cout << "\rEpoch " << currentEpoch
                          << " | Image " << currentImageIdx << "/" << trainingImages.size()
                          << " | Loss: " << (epochLoss / epochIterations) << std::flush;
            }
            
            // Density control application
            if (totalIterations % densityControlInterval == 0 && totalIterations > 500) {
                // Get camera parameters for screen-space pruning
                // Use scaled intrinsics to match actual training texture resolution
                const TrainingImage& currentImg = trainingImages[currentImageIdx];
                const ColmapCamera& cam = colmapData.cameras.at(currentImg.cameraId);
                
                uint32_t actualWidth = currentImg.texture->width();
                float scaleX = (float)actualWidth / (float)cam.width;
                
                // Scale focal length to match actual training resolution
                float focalLength = cam.fx * scaleX;
                float imageWidth = (float)actualWidth;
                float avgDepth = sceneExtent;
                
                densityController->apply(commandQueue, gaussianBuffer, positionBuffer,
                                         nullptr, gaussianCount, totalIterations,
                                         0.0002f,
                                         0.003f,  // min_opacity: must be lower than reset value (0.01)
                                         0.1f * sceneExtent,
                                         focalLength,
                                         imageWidth,
                                         avgDepth);
                
                // Reallocate gradients buffer if needed
                if (gaussianGradients->length() < gaussianCount * sizeof(GaussianGradients)) {
                    gaussianGradients->release();
                    gaussianGradients = metalDevice->newBuffer(gaussianCount * sizeof(GaussianGradients),
                                                               MTL::ResourceStorageModeShared);
                }
            }
            
            // Advance to next image
            currentImageIdx++;
            // New epoch
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
        // Render current view
        render(camera);
    }
    activeCamera = nullptr;
}

// Cleanup resources
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

// Initialize Metal device
void MTLEngine::initDevice() {
    // Get the default Metal device
    metalDevice = MTL::CreateSystemDefaultDevice();
    if (!metalDevice) {
        std::cerr << "Failed to create Metal device" << std::endl;
        std::exit(1);
    }
    std::cout << "Using Metal device: " << metalDevice->name()->utf8String() << std::endl;
}

// Initialize GLFW window with Metal layer
void MTLEngine::initWindow() {
    // Initialize GLFW
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindow = glfwCreateWindow(800, 600, "Gaussian Splatting", NULL, NULL);
    if (!glfwWindow) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    // Create Metal layer
    metalWindow = glfwGetCocoaWindow(glfwWindow);
    // Set up Metal layer for rendering
    metalLayer = [CAMetalLayer layer];
    // Configure Metal layer properties
    metalLayer.device = (__bridge id<MTLDevice>)metalDevice;
    metalLayer.pixelFormat = MTLPixelFormatRGBA8Unorm;
    metalWindow.contentView.layer = metalLayer;
    metalWindow.contentView.wantsLayer = YES;
    metalLayer.frame = metalWindow.contentView.bounds;
    metalLayer.drawableSize = CGSizeMake(800, 600);
}

// Setup GLFW callbacks for input handling
void MTLEngine::setupCallbacks() {
    glfwSetWindowUserPointer(glfwWindow, this);
    glfwSetFramebufferSizeCallback(glfwWindow, framebufferSizeCallback);
    glfwSetMouseButtonCallback(glfwWindow, mouseButtonCallback);
    glfwSetCursorPosCallback(glfwWindow, cursorPosCallback);
    glfwSetScrollCallback(glfwWindow, scrollCallback);
    glfwSetKeyCallback(glfwWindow, keyCallback);
}

// GLFW callback implementations
void MTLEngine::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    MTLEngine* engine = static_cast<MTLEngine*>(glfwGetWindowUserPointer(window));
    
    // Left button for orbiting
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            engine->isDragging = true;
            glfwGetCursorPos(window, &engine->lastMouseX, &engine->lastMouseY);
        } else if (action == GLFW_RELEASE) {
            engine->isDragging = false;
        }
    }
    
    // Right or middle button for panning
    if (button == GLFW_MOUSE_BUTTON_RIGHT || button == GLFW_MOUSE_BUTTON_MIDDLE) {
        if (action == GLFW_PRESS) {
            engine->isPanning = true;
            glfwGetCursorPos(window, &engine->lastMouseX, &engine->lastMouseY);
        } else if (action == GLFW_RELEASE) {
            engine->isPanning = false;
        }
    }
}

// GLFW cursor position callback
void MTLEngine::cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    MTLEngine* engine = static_cast<MTLEngine*>(glfwGetWindowUserPointer(window));
    if (!engine->activeCamera) return;
    
    // Calculate mouse movement delta
    double deltaX = xpos - engine->lastMouseX;
    double deltaY = ypos - engine->lastMouseY;
    
    // Orbit or pan based on mouse state
    if (engine->isDragging) {
        float orbitSpeed = 0.005f;
        engine->activeCamera->orbit(-deltaX * orbitSpeed, -deltaY * orbitSpeed);
    }
    
    if (engine->isPanning) {
        engine->activeCamera->pan(deltaX, deltaY);
    }
    
    // Update last mouse position
    engine->lastMouseX = xpos;
    engine->lastMouseY = ypos;
}

// GLFW scroll callback for zooming
void MTLEngine::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    // Zoom the active camera based on scroll input
    MTLEngine* engine = static_cast<MTLEngine*>(glfwGetWindowUserPointer(window));
    if (!engine->activeCamera) return;
    
    float zoomSpeed = 0.5f;
    engine->activeCamera->zoom(-yoffset * zoomSpeed);
}

// Load Gaussians into GPU buffers
void MTLEngine::loadGaussians(const std::vector<Gaussian>& gaussians, float sceneExtent) {
    // Create Gaussian buffer
    gaussianCount = gaussians.size();
    gaussianBuffer = metalDevice->newBuffer(gaussians.data(), gaussianCount * sizeof(Gaussian),
                                            MTL::ResourceStorageModeShared);
    
    // Create position buffer for sorting
    positionBuffer = metalDevice->newBuffer(gaussianCount * sizeof(simd_float3),
                                            MTL::ResourceStorageModeShared);
    // Populate position buffer
    simd_float3* positions = (simd_float3*)positionBuffer->contents();
    for (size_t i = 0; i < gaussianCount; i++) {
        positions[i] = gaussians[i].position;
    }
    
    // Use the camera-based scene extent for density control
    if (sceneExtent <= 0.0f) {
        // Compute extent from point cloud as fallback
        float min_x = FLT_MAX, max_x = -FLT_MAX;
        float min_y = FLT_MAX, max_y = -FLT_MAX;
        float min_z = FLT_MAX, max_z = -FLT_MAX;
        // Find bounding box of Gaussians
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
    
    // Store and set scene extent for density control
    this->sceneExtent = sceneExtent;
    std::cout << "Density control using scene extent: " << sceneExtent << std::endl;
    DensityController::setSceneExtent(sceneExtent);
    
    // Initialize GPU radix sort, optimizer, density controller, and rasterizer
    gpuSort = new GPURadixSort32(metalDevice, shaderLibrary, 2000000);
    optimizer = new AdamOptimizer(metalDevice, shaderLibrary, 2000000);
    densityController = new DensityController(metalDevice, shaderLibrary);
    densityController->resetAccumulator(gaussianCount);
    
    // Create gradients buffer
    gaussianGradients = metalDevice->newBuffer(gaussianCount * sizeof(GaussianGradients),
                                               MTL::ResourceStorageModeShared);
    
    // Initialize tiled rasterizer
    tiledRasterizer = new TiledRasterizer(metalDevice, shaderLibrary, 2000000);
    
    std::cout << "Loaded " << gaussianCount << " Gaussians" << std::endl;
}

// Initialize Metal command queue
void MTLEngine::initCommandQueue() {
    commandQueue = metalDevice->newCommandQueue();
}

// Create Metal render pipeline
void MTLEngine::createPipeline() {
    // Create vertex and fragment functions
    MTL::Function* vertexShader = shaderLibrary->newFunction(NS::String::string("vertexShader", NS::ASCIIStringEncoding));
    assert(vertexShader);
    MTL::Function* fragmentShader = shaderLibrary->newFunction(NS::String::string("fragmentShader", NS::ASCIIStringEncoding));
    assert(fragmentShader);
    
    // Create render pipeline descriptor    
    MTL::RenderPipelineDescriptor* desc = MTL::RenderPipelineDescriptor::alloc()->init();
    // Set shaders
    desc->setVertexFunction(vertexShader);
    desc->setFragmentFunction(fragmentShader);
    
    // Set color attachment properties
    auto colorAttachment = desc->colorAttachments()->object(0);
    colorAttachment->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
    colorAttachment->setBlendingEnabled(true);
    colorAttachment->setSourceRGBBlendFactor(MTL::BlendFactorOne);
    colorAttachment->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
    colorAttachment->setRgbBlendOperation(MTL::BlendOperationAdd);
    colorAttachment->setSourceAlphaBlendFactor(MTL::BlendFactorOne);
    colorAttachment->setDestinationAlphaBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
    colorAttachment->setAlphaBlendOperation(MTL::BlendOperationAdd);
    
    // Create render pipeline state
    NS::Error* error;
    metalRenderPSO = metalDevice->newRenderPipelineState(desc, &error);
    
    if (metalRenderPSO == nil) {
        std::cout << "Error creating render pipeline state: " << error << std::endl;
        std::exit(0);
    }
    
    // Cleanup
    desc->release();
    vertexShader->release();
    fragmentShader->release();
}

// Load Metal shaders
void MTLEngine::loadShaders() {
    shaderLibrary = metalDevice->newDefaultLibrary();
    if (!shaderLibrary) {
        std::cerr << "Failed to load shader library" << std::endl;
        std::exit(1);
    }
}

// Create depth texture for rendering
void MTLEngine::createDepthTexture() {
    // Create depth texture descriptor
    MTL::TextureDescriptor* depthDesc = MTL::TextureDescriptor::alloc()->init();
    // Set texture properties
    depthDesc->setWidth(800);
    depthDesc->setHeight(600);
    depthDesc->setPixelFormat(MTL::PixelFormatDepth32Float);
    depthDesc->setStorageMode(MTL::StorageModePrivate);
    depthDesc->setUsage(MTL::TextureUsageRenderTarget);
    // Create depth texture
    depthTexture = metalDevice->newTexture(depthDesc);
    depthDesc->release();
}

void MTLEngine::render(Camera& camera) {
    // GPU radix sort for depth ordering
    MTL::Buffer* sortedIndices = gpuSort->sort(commandQueue, positionBuffer, camera.get_position(), gaussianCount);
    
    // Prepare uniform data
    Uniforms uniforms;
    
    // Debug printing flag
    static bool renderDebugPrinted = false;
    
    // Set camera matrices based on mode
    if (useTrainingView && !trainingImages.empty()) {
        const TrainingImage& img = trainingImages[currentTrainingIndex];
        const ColmapCamera& cam = colmapData.cameras.at(img.cameraId);
        
        // Set view matrix from COLMAP pose must be done before any transforms
        uniforms.viewMatrix = viewMatrixFromColmap(img.rotation, img.translation);

        
        // For display render to window size
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
        // Build projection matrix in column-major format directly 
        // Projection matrix matching projectionFromColmap() for consistency
        // COLMAP Y-down + Metal texture top-left origin = NO Y flip needed
        simd_float4x4 proj = {0};
        proj.columns[0][0] = 2.0f * scaledFx / windowWidth;
        proj.columns[1][1] = 2.0f * scaledFy / windowHeight; 
        proj.columns[2][0] = 2.0f * cx / windowWidth - 1.0f;  
        proj.columns[2][1] = 2.0f * cy / windowHeight - 1.0f;
        proj.columns[2][2] = far / (far - near);
        proj.columns[2][3] = 1.0f; 
        proj.columns[3][2] = -(far * near) / (far - near);
        // Set projection matrix
        uniforms.projectionMatrix = proj;
        
        // Compute view-projection matrix
        uniforms.viewProjectionMatrix = matrix_multiply(uniforms.projectionMatrix, uniforms.viewMatrix);
        uniforms.screenSize = simd_make_float2((float)windowWidth, (float)windowHeight);
        uniforms.focalLength = simd_make_float2(scaledFx, scaledFy);
        
        // compute camera position from view matrix
        simd_float3x3 R;
        R.columns[0] = uniforms.viewMatrix.columns[0].xyz;
        R.columns[1] = uniforms.viewMatrix.columns[1].xyz;
        R.columns[2] = uniforms.viewMatrix.columns[2].xyz;
        // -C^T * R^T
        uniforms.cameraPos = -matrix_multiply(simd_transpose(R), img.translation);
        
        // Debug printing
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
            // Multiply to get view and clip positions
            simd_float4 viewPos = matrix_multiply(uniforms.viewMatrix, simd_make_float4(testPos, 1.0f));
            simd_float4 clipPos = matrix_multiply(uniforms.viewProjectionMatrix, simd_make_float4(testPos, 1.0f));
            // Debug print
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
        // Free camera view
        float fovY = 45.0f * M_PI / 180.0f;
        float fy = windowHeight / (2.0f * tan(fovY / 2.0f));
        float fx = fy;
        
        // Set camera matrices
        uniforms.viewMatrix = camera.get_view_matrix();
        uniforms.projectionMatrix = camera.get_projection_matrix();
        uniforms.viewProjectionMatrix = matrix_multiply(camera.get_projection_matrix(), camera.get_view_matrix());
        uniforms.screenSize = simd_make_float2((float)windowWidth, (float)windowHeight);
        uniforms.focalLength = simd_make_float2(fx, fy);
        uniforms.cameraPos = camera.get_position();
        
        // Debug printing
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
            // Multiply to get view and clip positions
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
    // Upload uniforms to GPU
    memcpy(uniformBuffer->contents(), &uniforms, sizeof(Uniforms));
    
    // Get next drawable from Metal layer
    CA::MetalDrawable* drawable = (__bridge CA::MetalDrawable*)[metalLayer nextDrawable];
    if (!drawable) return;
    // Set up render pass descriptor
    MTL::RenderPassDescriptor* renderPassDesc = MTL::RenderPassDescriptor::alloc()->init();
    renderPassDesc->colorAttachments()->object(0)->setTexture(drawable->texture());
    renderPassDesc->colorAttachments()->object(0)->setLoadAction(MTL::LoadActionClear);
    renderPassDesc->colorAttachments()->object(0)->setStoreAction(MTL::StoreActionStore);
    renderPassDesc->colorAttachments()->object(0)->setClearColor(MTL::ClearColor(1.0, 1.0, 1.0, 1.0));
    
    // Create command buffer and render encoder
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::RenderCommandEncoder* encoder = commandBuffer->renderCommandEncoder(renderPassDesc);
    
    // Set pipeline and draw
    encoder->setRenderPipelineState(metalRenderPSO);
    // Set buffers
    encoder->setVertexBuffer(gaussianBuffer, 0, 0);
    encoder->setVertexBuffer(uniformBuffer, 0, 1);
    encoder->setVertexBuffer(sortedIndices, 0, 2);
    // Draw Gaussians as triangle strips
    encoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4), gaussianCount);
    
    // Finalize encoding and present
    encoder->endEncoding();
    commandBuffer->presentDrawable(drawable);
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    // Cleanup
    renderPassDesc->release();
}

// GLFW framebuffer size callback
void MTLEngine::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    if (width == 0 || height == 0) return;
    
    // Update engine state
    MTLEngine* engine = static_cast<MTLEngine*>(glfwGetWindowUserPointer(window));
    engine->windowWidth = width;
    engine->windowHeight = height;
    
    // Update Metal layer size
    engine->metalLayer.drawableSize = CGSizeMake(width, height);
    
    // Recreate depth texture
    if (engine->activeCamera) {
        engine->activeCamera->setAspectRatio((float)width / (float)height);
    }
}

// Load training images from COLMAP data
void MTLEngine::loadTrainingData(const ColmapData& colmap, const std::string& imagePath) {
    colmapData = colmap;
    trainingImages = loadTrainingImages(metalDevice, colmap, imagePath);
    std::cout << "Loaded " << trainingImages.size() << " training images" << std::endl;
}

// Convert COLMAP quaternion and translation to view matrix
simd_float4x4 MTLEngine::viewMatrixFromColmap(simd_float4 quat, simd_float3 translation) {
    // Quaternion convention: float4(.x=w, .y=x, .z=y, .w=z)
    float w = quat.x, x = quat.y, y = quat.z, z = quat.w;
    
    // Convert quaternion to rotation matrix R
    matrix_float3x3 R;
    // Column 0 of R: (R[0][0], R[1][0], R[2][0])
    R.columns[0] = simd_make_float3(1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y));
    // Column 1 of R: (R[0][1], R[1][1], R[2][1])
    R.columns[1] = simd_make_float3(2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x));
    // Column 2 of R: (R[0][2], R[1][2], R[2][2])
    R.columns[2] = simd_make_float3(2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y));
    
    
    // Build view matrix: [ R | t ] in column-major order
    simd_float4x4 view;
    view.columns[0] = simd_make_float4(R.columns[0], 0);
    view.columns[1] = simd_make_float4(R.columns[1], 0);
    view.columns[2] = simd_make_float4(R.columns[2], 0);
    view.columns[3] = simd_make_float4(translation, 1);
    
    return view;
}

// Create projection matrix from COLMAP camera parameters
simd_float4x4 MTLEngine::projectionFromColmap(const ColmapCamera& cam, float nearZ, float farZ) {
    // Extract parameters
    float fx = cam.fx;
    float fy = cam.fy;
    float cx = cam.cx;
    float cy = cam.cy;
    float w = (float)cam.width;
    float h = (float)cam.height;
    
    // Build projection matrix in column-major format directly
    simd_float4x4 proj = {0};
    proj.columns[0][0] = 2.0f * fx / w;
    proj.columns[1][1] = 2.0f * fy / h; 
    proj.columns[2][0] = 2.0f * cx / w - 1.0f;
    proj.columns[2][1] = 2.0f * cy / h - 1.0f;
    proj.columns[2][2] = farZ / (farZ - nearZ);
    proj.columns[2][3] = 1.0f;
    proj.columns[3][2] = -(farZ * nearZ) / (farZ - nearZ);
    
    return proj;
}

// GLFW key callback for input handling
void MTLEngine::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    MTLEngine* engine = static_cast<MTLEngine*>(glfwGetWindowUserPointer(window));
    
    // Toggle training view with T
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_T) {
            engine->useTrainingView = !engine->useTrainingView;
            std::cout << "Training view: " << (engine->useTrainingView ? "ON" : "OFF") << std::endl;
        }
        
        // Toggle training with Space
        if (key == GLFW_KEY_SPACE) {
            engine->isTraining = !engine->isTraining;
            std::cout << "Training: " << (engine->isTraining ? "STARTED" : "PAUSED") << std::endl;
        }
        
        // Navigate training images with Left/Right arrows
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

// Create render target texture
void MTLEngine::createRenderTarget(uint32_t width, uint32_t height) {
    if (renderTarget) renderTarget->release();
    
    // Create texture descriptor
    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::alloc()->init();
    desc->setWidth(width);
    desc->setHeight(height);
    desc->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
    desc->setStorageMode(MTL::StorageModeShared);
    desc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    renderTarget = metalDevice->newTexture(desc);
    desc->release();
}

// Create loss computation pipelines
void MTLEngine::createLossPipeline() {
    NS::Error* error = nullptr;
    
    // L1 loss pipeline
    MTL::Function* lossFunc = shaderLibrary->newFunction(NS::String::string("computeL1Loss", NS::ASCIIStringEncoding));
    lossComputePSO = metalDevice->newComputePipelineState(lossFunc, &error);
    if (!lossComputePSO) {
        std::cerr << "Failed to create L1 loss pipeline: " << error->localizedDescription()->utf8String() << std::endl;
    }
    lossFunc->release();
    
    // SSIM loss pipeline
    MTL::Function* ssimFunc = shaderLibrary->newFunction(NS::String::string("computeSSIM", NS::ASCIIStringEncoding));
    ssimComputePSO = metalDevice->newComputePipelineState(ssimFunc, &error);
    if (!ssimComputePSO) {
        std::cerr << "Failed to create SSIM pipeline: " << error->localizedDescription()->utf8String() << std::endl;
    }
    ssimFunc->release();
    
    // Combined loss pipeline  
    MTL::Function* combinedFunc = shaderLibrary->newFunction(NS::String::string("computeCombinedLoss", NS::ASCIIStringEncoding));
    combinedLossPSO = metalDevice->newComputePipelineState(combinedFunc, &error);
    if (!combinedLossPSO) {
        std::cerr << "Failed to create combined loss pipeline: " << error->localizedDescription()->utf8String() << std::endl;
    }
    combinedFunc->release();
    
    // Reduction pipeline
    MTL::Function* reduceFunc = shaderLibrary->newFunction(NS::String::string("reduceLoss", NS::ASCIIStringEncoding));
    reductionPSO = metalDevice->newComputePipelineState(reduceFunc, &error);
    if (!reductionPSO) {
        std::cerr << "Failed to create reduction pipeline: " << error->localizedDescription()->utf8String() << std::endl;
    }
    reduceFunc->release();
    
    std::cout << "Loss pipelines created (L1 + D-SSIM with lambda=" << lambdaDSSIM << ")" << std::endl;
}

// Compute combined loss (L1 + D-SSIM) between rendered and ground truth textures
float MTLEngine::computeLoss(MTL::Texture* rendered, MTL::Texture* groundTruth) {
    uint32_t width = rendered->width();
    uint32_t height = rendered->height();
    uint32_t pixelCount = width * height;
    
    // Allocate buffers for L1, SSIM, and combined losses
    if (!lossBuffer || lossBuffer->length() < pixelCount * sizeof(float)) {
        if (lossBuffer) lossBuffer->release();
        lossBuffer = metalDevice->newBuffer(pixelCount * sizeof(float), MTL::ResourceStorageModeShared);
    }
    
    // SSIM buffer
    if (!ssimBuffer || ssimBuffer->length() < pixelCount * sizeof(float)) {
        if (ssimBuffer) ssimBuffer->release();
        ssimBuffer = metalDevice->newBuffer(pixelCount * sizeof(float), MTL::ResourceStorageModeShared);
    }
    
    // Combined loss buffer
    if (!combinedLossBuffer || combinedLossBuffer->length() < pixelCount * sizeof(float)) {
        if (combinedLossBuffer) combinedLossBuffer->release();
        combinedLossBuffer = metalDevice->newBuffer(pixelCount * sizeof(float), MTL::ResourceStorageModeShared);
    }
    
    if (!totalLossBuffer) {
        totalLossBuffer = metalDevice->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
    }
    
    // Initialize total loss to zero
    float zero = 0.0f;
    memcpy(totalLossBuffer->contents(), &zero, sizeof(float));
    
    // Create command buffer and encoder
    MTL::CommandBuffer* cmdBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = cmdBuffer->computeCommandEncoder();
    
    // Define threadgroups
    MTL::Size gridSize = MTL::Size(width, height, 1);
    MTL::Size threadGroupSize = MTL::Size(16, 16, 1);
    
    // Step 1: Compute L1 loss per pixel
    encoder->setComputePipelineState(lossComputePSO);
    encoder->setTexture(rendered, 0);
    encoder->setTexture(groundTruth, 1);
    encoder->setBuffer(lossBuffer, 0, 0);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    
    // Step 2: Compute SSIM (D-SSIM) per pixel
    encoder->setComputePipelineState(ssimComputePSO);
    encoder->setTexture(rendered, 0);
    encoder->setTexture(groundTruth, 1);
    encoder->setBuffer(ssimBuffer, 0, 0);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    
    // Step 3: Combine losses: (1-lambda)*L1 + lambda*D-SSIM
    encoder->setComputePipelineState(combinedLossPSO);
    encoder->setTexture(rendered, 0);
    encoder->setTexture(groundTruth, 1);
    // L1 losses
    encoder->setBuffer(lossBuffer, 0, 0);      
    // SSIM losses       
    encoder->setBuffer(ssimBuffer, 0, 1);      
    // Output combined     
    encoder->setBuffer(combinedLossBuffer, 0, 2);   
    encoder->setBytes(&lambdaDSSIM, sizeof(float), 3);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    
    // Step 4: Reduce combined loss to single value
    encoder->setComputePipelineState(reductionPSO);
    encoder->setBuffer(combinedLossBuffer, 0, 0);
    encoder->setBuffer(totalLossBuffer, 0, 1);
    encoder->setBytes(&pixelCount, sizeof(uint32_t), 2);
    
    // Launch reduction with fixed number of threads
    uint32_t reductionThreads = 1024;
    encoder->dispatchThreads(MTL::Size(reductionThreads, 1, 1), MTL::Size(64, 1, 1));
    
    // Finalize encoding
    encoder->endEncoding();
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();
    
    // Retrieve total loss
    float totalLoss = *(float*)totalLossBuffer->contents();
    return totalLoss / pixelCount;
}

// Perform a single training step on a specified training image
float MTLEngine::trainStep(size_t imageIndex, 
                           float lr_position,
                           float lr_scale,
                           float lr_rotation,
                           float lr_opacity,
                           float lr_sh) {
    // Validate image index
    if (imageIndex >= trainingImages.size()) return 0.0f;
    
    // Prepare uniforms for the specified training image
    const TrainingImage& img = trainingImages[imageIndex];
    const ColmapCamera& cam = colmapData.cameras.at(img.cameraId);
    
    // Get image texture size
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
    
    // Ensure render target matches image size
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
    
    // Set up uniforms
    TiledUniforms uniforms;
    uniforms.viewMatrix = viewMatrixFromColmap(img.rotation, img.translation);
    uniforms.projectionMatrix = projectionFromColmap(scaledCam, 0.1f, 1000.0f);
    uniforms.viewProjectionMatrix = matrix_multiply(uniforms.projectionMatrix, uniforms.viewMatrix);
    uniforms.screenSize = simd_make_float2((float)actualWidth, (float)actualHeight);
    uniforms.focalLength = simd_make_float2(scaledFx, scaledFy);
    
    // Compute camera position from view matrix
    simd_float3x3 R;
    R.columns[0] = uniforms.viewMatrix.columns[0].xyz;
    R.columns[1] = uniforms.viewMatrix.columns[1].xyz;
    R.columns[2] = uniforms.viewMatrix.columns[2].xyz;
    uniforms.cameraPos = -matrix_multiply(simd_transpose(R), img.translation);
    
    // DEBUG Check view-space coordinates for first few Gaussians on CPU side
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
    
    // Save training render less frequently
    static int saveCounter = 0;
    if (saveCounter % 500 == 0) {
        // Save rendered image
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
        // Print average opacity and scale
        Gaussian* g = (Gaussian*)gaussianBuffer->contents();
        // Calculate average opacity and scale for a sample of Gaussians
        float avgOpacity = 0, avgScale = 0;
        // Sample up to 100 Gaussians for stats
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

// Update position buffer from Gaussian buffer for external use
void MTLEngine::updatePositionBuffer() {
    // Copy positions from Gaussian buffer to position buffer
    Gaussian* gaussians = (Gaussian*)gaussianBuffer->contents();
    simd_float3* positions = (simd_float3*)positionBuffer->contents();
    
    for (size_t i = 0; i < gaussianCount; i++) {
        positions[i] = gaussians[i].position;
    }
}

// Exponential learning rate decay with matches 3DGS official schedule
static float exponentialLRDecay(float lr_init, float lr_final, size_t currentIter, size_t maxIter) {
    // Clamp to final learning rate
    if (currentIter >= maxIter) return lr_final;
    // Compute exponential decay
    float t = static_cast<float>(currentIter) / static_cast<float>(maxIter);
    return lr_init * std::pow(lr_final / lr_init, t);
}

void MTLEngine::train(size_t numEpochs) {
    // Training loop
    std::cout << "Starting training for " << numEpochs << " epochs..." << std::endl;
    std::cout << "Gaussians: " << gaussianCount << " | Images: " << trainingImages.size() << std::endl;
    
    // Training configuration which matches 3DGS official settings
    const size_t OPACITY_RESET_INTERVAL = 3000;
    const size_t DENSIFY_FROM_ITER = 500;
    const size_t DENSIFY_UNTIL_ITER = 15000;
    const float OPACITY_RESET_VALUE = -4.6f;  // sigmoid^-1(0.01) ≈ -4.6
    
    // Position LR decays exponentially from init to final
    const float POSITION_LR_INIT = 0.00016f;
    // 100x smaller at end
    const float POSITION_LR_FINAL = 0.0000016f;  
    // Back to official value - need Gaussians to shrink for sharpness
    const float SCALE_LR = 0.005f;  
    const float ROTATION_LR = 0.001f;
    const float OPACITY_LR = 0.05f;
    const float SH_LR = 0.0025f;
    
    // Total iterations for LR scheduling
    const size_t totalExpectedIters = numEpochs * trainingImages.size();
    
    std::cout << "Learning rate decay: position " << POSITION_LR_INIT 
              << " -> " << POSITION_LR_FINAL << " over " << totalExpectedIters << " iterations" << std::endl;
    
    // Start training timer
    auto startTime = std::chrono::high_resolution_clock::now();
    size_t totalIterations = 0;
    
    // Training loop
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
            
            // ============================================
            // DENSITY CONTROL (must happen BEFORE opacity reset - matching official order!)
            // ============================================
            // Official logic: size_threshold = 20 if iteration > opacity_reset_interval else None
            // Note: uses > not >= so at iter 3000, size_threshold is still None
            bool enableScreenPruning = (totalIterations > OPACITY_RESET_INTERVAL);
            
            // Density control condition (matching official train.py)
            // Official: if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0
            bool shouldDensify = (totalIterations > DENSIFY_FROM_ITER &&
                                  totalIterations < DENSIFY_UNTIL_ITER &&
                                  totalIterations % densityControlInterval == 0);
            
            // Apply density control if needed
            if (shouldDensify) {
                // Get camera parameters for screen-space pruning
                const TrainingImage& currentImg = trainingImages[imgIdx];
                const ColmapCamera& cam = colmapData.cameras.at(currentImg.cameraId);
                
                uint32_t actualWidth = currentImg.texture->width();
                float scaleX = (float)actualWidth / (float)cam.width;
                float focalLength = cam.fx * scaleX;
                float imageWidth = (float)actualWidth;
                // Use conservative avgDepth to prevent over-aggressive screen pruning
                float avgDepth = 2.0f * sceneExtent;

                if (totalIterations % 1000 == 0) {
                    std::cout << "Density params: fx=" << focalLength 
                            << " width=" << imageWidth 
                            << " avgDepth=" << avgDepth 
                            << " sceneExtent=" << sceneExtent 
                            << " screenPrune=" << (enableScreenPruning ? "ON" : "OFF") << std::endl;
                }
                
                // Pass enableScreenPruning to density controller
                // Official uses size_threshold=20 pixels when enabled
                densityController->apply(commandQueue, gaussianBuffer, positionBuffer,
                                         nullptr, gaussianCount, totalIterations,
                                         0.0002f,
                                         0.005f,  // min_opacity: official uses 0.005
                                         0.1f * sceneExtent,
                                         focalLength,
                                         imageWidth,
                                         avgDepth);
                
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
            
            // ============================================
            // OPACITY RESET (must happen AFTER density control - matching official order!)
            // ============================================
            if (totalIterations % OPACITY_RESET_INTERVAL == 0 && 
                totalIterations > 0 && 
                totalIterations < DENSIFY_UNTIL_ITER) {
                std::cout << std::endl << "Resetting opacity at iteration " << totalIterations << std::endl;
                
                // Clamp opacities in Gaussian buffer
                Gaussian* gaussians = (Gaussian*)gaussianBuffer->contents();
                for (size_t i = 0; i < gaussianCount; i++) {
                    // Clamp opacity to max 0.01: min(current, sigmoid^-1(0.01))
                    if (gaussians[i].opacity > OPACITY_RESET_VALUE) {
                        gaussians[i].opacity = OPACITY_RESET_VALUE;
                    }
                }
                
                // Reset optimizer momentum for opacity to allow fresh learning
                optimizer->resetOpacityMomentum();
                
                // Reset gradient accumulators after opacity reset
                // This ensures next densification uses fresh post-reset gradients
                densityController->resetAccumulator(gaussianCount);
            }
            
            pool->release();
        }
        
        // Update position buffer at end of epoch
        updatePositionBuffer();
        
        // Epoch timing
        auto epochEnd = std::chrono::high_resolution_clock::now();
        auto epochDuration = std::chrono::duration_cast<std::chrono::seconds>(epochEnd - epochStart).count();
        
        // Compute current position LR for logging
        float currentLR = exponentialLRDecay(POSITION_LR_INIT, POSITION_LR_FINAL, 
                                             totalIterations, totalExpectedIters);
        
        // Epoch summary
        std::cout << std::endl << "=== Epoch " << epoch << " | Loss: "
                  << (epochLoss / trainingImages.size())
                  << " | Gaussians: " << gaussianCount
                  << " | pos_lr: " << currentLR
                  << " | Time: " << epochDuration << "s ===" << std::endl;
    }
    
    // Total training time
    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    
    std::cout << "Training complete! Total time: " << totalDuration << "s" << std::endl;
}

// Export rendered views for all training images to specified folder
void MTLEngine::exportTrainingViews(const std::string& outputFolder) {
    if (trainingImages.empty()) {
        std::cerr << "No training images to export views for!" << std::endl;
        return;
    }
    
    // Create output folder if it doesn't exist
    std::string mkdirCmd = "mkdir -p \"" + outputFolder + "\"";
    system(mkdirCmd.c_str());
    
    std::cout << "\n=== Exporting " << trainingImages.size() << " training views to " << outputFolder << " ===" << std::endl;
    
    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Loop over training images
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
        
        // Compute view-projection matrix
        uniforms.viewProjectionMatrix = matrix_multiply(uniforms.projectionMatrix, uniforms.viewMatrix);
        uniforms.screenSize = simd_make_float2((float)cam.width, (float)cam.height);
        uniforms.focalLength = simd_make_float2(cam.fx, cam.fy);
        
        // Compute camera position from view matrix
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
    
    // Final timing
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    
    std::cout << std::endl << "Exported " << trainingImages.size() << " views in " << duration << "s" << std::endl;
    std::cout << "Output folder: " << outputFolder << std::endl;
}

//
//  mtl_engine.hpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-24.
//


#include "mtl_engine.hpp"
#include "gpu_sort.hpp"
#include <iostream>
#include <iomanip>
#include "image_loader.hpp"
#include "gradients.hpp"
#include <chrono>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>
#include <Foundation/NSAutoreleasePool.hpp>

// Helper to convert half float to float
static float halfToFloat(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        // Denormalized
        float f = mant / 1024.0f;
        return sign ? -f * powf(2.0f, -14.0f) : f * powf(2.0f, -14.0f);
    } else if (exp == 31) {
        return mant ? NAN : (sign ? -INFINITY : INFINITY);
    }

    float f = (1.0f + mant / 1024.0f) * powf(2.0f, (float)exp - 15.0f);
    return sign ? -f : f;
}

// For debugging save a texture to PPM file
void saveTextureToPPM(MTL::Texture* texture, MTL::Device* device, MTL::CommandQueue* queue, const char* filename) {
    uint32_t width = texture->width();
    uint32_t height = texture->height();

    // Detect texture format
    bool isFloat16 = (texture->pixelFormat() == MTL::PixelFormatRGBA16Float);
    size_t bytesPerPixel = isFloat16 ? 8 : 4;
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

    // Convert to RGB bytes
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            size_t offset = y * bytesPerRow + x * bytesPerPixel;

            if (isFloat16) {
                // RGBA16Float: 4 half-floats (2 bytes each)
                uint16_t* halfData = (uint16_t*)(data + offset);
                float r = halfToFloat(halfData[0]);
                float g = halfToFloat(halfData[1]);
                float b = halfToFloat(halfData[2]);

                // Clamp and convert to 8-bit
                file.put((uint8_t)(fminf(fmaxf(r, 0.0f), 1.0f) * 255.0f));
                file.put((uint8_t)(fminf(fmaxf(g, 0.0f), 1.0f) * 255.0f));
                file.put((uint8_t)(fminf(fmaxf(b, 0.0f), 1.0f) * 255.0f));
            } else {
                // RGBA8Unorm: 4 uint8_t
                file.put(data[offset]);
                file.put(data[offset + 1]);
                file.put(data[offset + 2]);
            }
        }
    }

    // Cleanup
    file.close();
    readbackBuffer->release();
    printf("Saved render to %s (%dx%d)\n", filename, width, height);
}

// Analyze rendered texture for brightness statistics (debug)
void analyzeRenderBrightness(MTL::Texture* texture, MTL::Device* device, MTL::CommandQueue* queue) {
    uint32_t width = texture->width();
    uint32_t height = texture->height();

    // Detect texture format
    bool isFloat16 = (texture->pixelFormat() == MTL::PixelFormatRGBA16Float);
    size_t bytesPerPixel = isFloat16 ? 8 : 4;
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

    uint8_t* data = (uint8_t*)readbackBuffer->contents();

    // Analyze pixel brightness
    int overBrightPixels = 0;      // Any channel > 1.0
    int nearWhitePixels = 0;       // All channels > 0.95 (high background bleed)
    int veryBrightPixels = 0;      // Max channel > 1.5
    float maxBrightness = 0.0f;
    float totalBrightness = 0.0f;
    uint32_t totalPixels = width * height;

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            size_t offset = y * bytesPerRow + x * bytesPerPixel;
            float r, g, b;

            if (isFloat16) {
                uint16_t* halfData = (uint16_t*)(data + offset);
                r = halfToFloat(halfData[0]);
                g = halfToFloat(halfData[1]);
                b = halfToFloat(halfData[2]);
            } else {
                r = data[offset] / 255.0f;
                g = data[offset + 1] / 255.0f;
                b = data[offset + 2] / 255.0f;
            }

            float brightness = (r + g + b) / 3.0f;
            float maxChannel = fmaxf(fmaxf(r, g), b);
            totalBrightness += brightness;
            if (maxChannel > maxBrightness) maxBrightness = maxChannel;

            if (r > 1.0f || g > 1.0f || b > 1.0f) overBrightPixels++;
            if (r > 0.95f && g > 0.95f && b > 0.95f) nearWhitePixels++;
            if (maxChannel > 1.5f) veryBrightPixels++;
        }
    }

    float avgBrightness = totalBrightness / totalPixels;
    float nearWhitePct = 100.0f * nearWhitePixels / totalPixels;

    printf("[Render] over-bright: %d | near-white(>0.95): %d (%.1f%%) | very-bright(>1.5): %d | max: %.2f | avg: %.3f\n",
           overBrightPixels, nearWhitePixels, nearWhitePct, veryBrightPixels, maxBrightness, avgBrightness);

    readbackBuffer->release();
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
            float loss = trainStep(currentImageIdx, 0.00016f * sceneExtent);
            epochLoss += loss;
            epochIterations++;
            totalIterations++;
            
            // Display progress every 10 iterations
            if (epochIterations % 10 == 0) {
                std::cout << "\rEpoch " << currentEpoch
                          << " | Image " << currentImageIdx << "/" << trainingImages.size()
                          << " | Loss: " << std::fixed << std::setprecision(6) << (epochLoss / epochIterations) << std::flush;
            }
            
            // Density control application
            // Stop densification at 15k iterations 
            const size_t DENSIFY_UNTIL_ITER = 15000;
            if (totalIterations % densityControlInterval == 0 &&
                totalIterations > 500 &&
                totalIterations < DENSIFY_UNTIL_ITER) {
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

                size_t oldCount = gaussianCount;
                DensityStats stats = densityController->apply(commandQueue, gaussianBuffer, positionBuffer,
                                         nullptr, gaussianCount, totalIterations,
                                         0.0002f,
                                         0.005f,  // min_opacity (matches OPACITY_PRUNE_THRESHOLD)
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

                // Remap optimizer momentum for surviving Gaussians
                // Must call remapMomentum whenever changes occur, even if indexMapping is empty
                // (e.g., all Gaussians split - children need zero momentum, not inherited)
                bool hadChanges = (stats.numPruned > 0 || stats.numCloned > 0 || stats.numSplit > 0);
                if (optimizer && hadChanges) {
                    optimizer->remapMomentum(stats.indexMapping, gaussianCount);
                }
            }
            
            // Advance to next image
            currentImageIdx++;
            // New epoch
            if (currentImageIdx >= trainingImages.size()) {
                std::cout << std::endl << "=== Epoch " << currentEpoch << " complete | Avg Loss: "
                          << std::fixed << std::setprecision(6) << (epochLoss / epochIterations) << " ===" << std::endl;
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
    if (cpuSortedIndices) cpuSortedIndices->release();
    if (gpuSort) delete gpuSort;
    if (optimizer) delete optimizer;
    if (densityController) delete densityController;
    if (tiledRasterizer) delete tiledRasterizer;
    if (gaussianGradients) gaussianGradients->release();
    if (viewerRenderTarget) viewerRenderTarget->release();
    if (blitPSO) blitPSO->release();
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

    // Feature probing: this tells us if the runtime exposes Metal 3 vs Metal 4 families.
    // (This is OS/runtime-gated; Xcode alone won't change it.)
    auto printFamily = [&](MTL::GPUFamily family, const char* label) {
        bool supported = metalDevice->supportsFamily(family);
        std::cout << "  supportsFamily(" << label << "): " << (supported ? "YES" : "NO") << std::endl;
    };

    std::cout << "Metal family support:" << std::endl;
    printFamily(MTL::GPUFamilyMetal3, "Metal3");
    printFamily(MTL::GPUFamilyMetal4, "Metal4");
    // Useful context for Apple Silicon
    printFamily(MTL::GPUFamilyMac2, "Mac2");
    printFamily(MTL::GPUFamilyApple7, "Apple7");
    printFamily(MTL::GPUFamilyApple8, "Apple8");
    printFamily(MTL::GPUFamilyApple9, "Apple9");
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

    // The drawable must be sized in backing pixels, not layout points. Hardcoding the
    // 800x600 window size renders at half resolution on a Retina display and upscales 2x;
    // glfwGetFramebufferSize reports the real pixel count.
    glfwGetFramebufferSize(glfwWindow, &windowWidth, &windowHeight);
    metalLayer.contentsScale = metalWindow.backingScaleFactor;
    metalLayer.drawableSize = CGSizeMake(windowWidth, windowHeight);
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
    
    // Verify struct sizes match Metal shader layout
    static_assert(sizeof(Gaussian) == 112, "Gaussian struct must be 112 bytes to match Metal layout!");
    static_assert(sizeof(GaussianGradients) == 112, "GaussianGradients struct must be 112 bytes to match Metal layout!");
    
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

    // Create blit pipeline for copying float texture to drawable
    MTL::Function* blitVertex = shaderLibrary->newFunction(NS::String::string("blitVertexShader", NS::ASCIIStringEncoding));
    MTL::Function* blitFragment = shaderLibrary->newFunction(NS::String::string("blitFragmentShader", NS::ASCIIStringEncoding));

    // Create blit pipeline state
    if (blitVertex && blitFragment) {
        MTL::RenderPipelineDescriptor* blitDesc = MTL::RenderPipelineDescriptor::alloc()->init();
        blitDesc->setVertexFunction(blitVertex);
        blitDesc->setFragmentFunction(blitFragment);
        blitDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatRGBA8Unorm);

        blitPSO = metalDevice->newRenderPipelineState(blitDesc, &error);
        if (!blitPSO) {
            std::cerr << "Failed to create blit pipeline: " << error->localizedDescription()->utf8String() << std::endl;
        }

        blitDesc->release();
        blitVertex->release();
        blitFragment->release();
    } else {
        std::cerr << "Failed to load blit shaders" << std::endl;
    }
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

// CPU sort by depth (back-to-front for proper alpha blending)
void MTLEngine::cpuSortByDepth(simd_float3 cameraPos) {
    if (!gaussianBuffer || gaussianCount == 0) return;

    // Allocate CPU sorted indices buffer if needed
    if (!cpuSortedIndices || cpuSortedIndices->length() < gaussianCount * sizeof(uint32_t)) {
        if (cpuSortedIndices) cpuSortedIndices->release();
        cpuSortedIndices = metalDevice->newBuffer(gaussianCount * sizeof(uint32_t),
                                                   MTL::ResourceStorageModeShared);
    }

    // Get Gaussian positions
    Gaussian* gaussians = (Gaussian*)gaussianBuffer->contents();
    uint32_t* indices = (uint32_t*)cpuSortedIndices->contents();

    // Create distance-index pairs for sorting
    std::vector<std::pair<float, uint32_t>> distanceIndex(gaussianCount);

    for (size_t i = 0; i < gaussianCount; i++) {
        simd_float3 pos = gaussians[i].position;
        // Skip invalid positions
        if (std::isnan(pos.x) || std::isnan(pos.y) || std::isnan(pos.z)) {
            distanceIndex[i] = {-1e30f, (uint32_t)i};  // Put invalid at front (will be clipped)
            continue;
        }
        // Compute squared distance from camera
        simd_float3 diff = pos - cameraPos;
        float distSq = simd_dot(diff, diff);
        distanceIndex[i] = {distSq, (uint32_t)i};
    }

    // Sort by distance descending (far to near = back-to-front)
    std::sort(distanceIndex.begin(), distanceIndex.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Copy sorted indices to buffer
    for (size_t i = 0; i < gaussianCount; i++) {
        indices[i] = distanceIndex[i].second;
    }
}

void MTLEngine::render(Camera& camera) {
    // Debug printing flag
    static bool renderDebugPrinted = false;

    // Build TiledUniforms for the tiled rasterizer which is the same format used in training
    TiledUniforms tiledUniforms;

    // Set camera matrices based on mode
    if (useTrainingView && !trainingImages.empty()) {
        const TrainingImage& img = trainingImages[currentTrainingIndex];
        const ColmapCamera& cam = colmapData.cameras.at(img.cameraId);

        // Set view matrix from COLMAP pose
        tiledUniforms.viewMatrix = viewMatrixFromColmap(img.rotation, img.translation);

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

        // Build projection matrix in column-major format
        simd_float4x4 proj = {0};
        proj.columns[0][0] = 2.0f * scaledFx / windowWidth;
        proj.columns[1][1] = 2.0f * scaledFy / windowHeight;
        proj.columns[2][0] = 2.0f * cx / windowWidth - 1.0f;
        proj.columns[2][1] = 2.0f * cy / windowHeight - 1.0f;
        proj.columns[2][2] = far / (far - near);
        proj.columns[2][3] = 1.0f;
        proj.columns[3][2] = -(far * near) / (far - near);
        tiledUniforms.projectionMatrix = proj;

        // Compute view-projection matrix
        tiledUniforms.viewProjectionMatrix = matrix_multiply(tiledUniforms.projectionMatrix, tiledUniforms.viewMatrix);
        tiledUniforms.screenSize = simd_make_float2((float)windowWidth, (float)windowHeight);
        tiledUniforms.focalLength = simd_make_float2(scaledFx, scaledFy);

        // Compute camera position from view matrix
        simd_float3x3 R;
        R.columns[0] = tiledUniforms.viewMatrix.columns[0].xyz;
        R.columns[1] = tiledUniforms.viewMatrix.columns[1].xyz;
        R.columns[2] = tiledUniforms.viewMatrix.columns[2].xyz;
        tiledUniforms.cameraPos = -matrix_multiply(simd_transpose(R), img.translation);

        // Debug printing
        if (!renderDebugPrinted) {
            printf("\n=== RENDER DEBUG (Training View - Tiled) ===\n");
            printf("Window: %dx%d, COLMAP: %dx%d\n", windowWidth, windowHeight, cam.width, cam.height);
            printf("Focal: scaled fx=%.2f fy=%.2f\n", scaledFx, scaledFy);
            printf("Camera position: (%.3f, %.3f, %.3f)\n",
                   tiledUniforms.cameraPos.x, tiledUniforms.cameraPos.y, tiledUniforms.cameraPos.z);
            renderDebugPrinted = true;
        }
    } else {
        // Free camera view convert to COLMAP-style matrices for consistency
        float fovY = 45.0f * M_PI / 180.0f;
        float fy = windowHeight / (2.0f * tan(fovY / 2.0f));
        float fx = fy;

        // Get camera position and compute look-at direction
        simd_float3 camPos = camera.get_position();

        // Use the Camera's view matrix but ensure it uses left-hand convention
        tiledUniforms.viewMatrix = camera.get_view_matrix();

        // Build projection matrix matching COLMAP style for tiled rasterizer
        float near = 0.1f;
        float far = 10000.0f;
        float cx = windowWidth / 2.0f;
        float cy = windowHeight / 2.0f;

        simd_float4x4 proj = {0};
        proj.columns[0][0] = 2.0f * fx / windowWidth;
        proj.columns[1][1] = 2.0f * fy / windowHeight;
        proj.columns[2][0] = 2.0f * cx / windowWidth - 1.0f;
        proj.columns[2][1] = 2.0f * cy / windowHeight - 1.0f;
        proj.columns[2][2] = far / (far - near);
        proj.columns[2][3] = 1.0f;
        proj.columns[3][2] = -(far * near) / (far - near);
        tiledUniforms.projectionMatrix = proj;

        tiledUniforms.viewProjectionMatrix = matrix_multiply(tiledUniforms.projectionMatrix, tiledUniforms.viewMatrix);
        tiledUniforms.screenSize = simd_make_float2((float)windowWidth, (float)windowHeight);
        tiledUniforms.focalLength = simd_make_float2(fx, fy);
        tiledUniforms.cameraPos = camPos;

        // Debug printing
        if (!renderDebugPrinted && gaussianBuffer) {
            printf("\n=== RENDER DEBUG (Free Camera - Tiled) ===\n");
            printf("Window: %dx%d\n", windowWidth, windowHeight);
            printf("Focal: fx=%.2f fy=%.2f\n", fx, fy);
            printf("Camera position: (%.3f, %.3f, %.3f)\n",
                   tiledUniforms.cameraPos.x, tiledUniforms.cameraPos.y, tiledUniforms.cameraPos.z);
            printf("View matrix:\n");
            for (int r = 0; r < 4; r++) {
                printf("  [%.4f %.4f %.4f %.4f]\n",
                       tiledUniforms.viewMatrix.columns[0][r],
                       tiledUniforms.viewMatrix.columns[1][r],
                       tiledUniforms.viewMatrix.columns[2][r],
                       tiledUniforms.viewMatrix.columns[3][r]);
            }

            // Test transform a sample point
            Gaussian* gaussians = (Gaussian*)gaussianBuffer->contents();
            simd_float3 testPos = gaussians[0].position;
            simd_float4 viewPos = matrix_multiply(tiledUniforms.viewMatrix, simd_make_float4(testPos, 1.0f));
            simd_float4 clipPos = matrix_multiply(tiledUniforms.viewProjectionMatrix, simd_make_float4(testPos, 1.0f));
            printf("Sample Gaussian 0:\n");
            printf("  world pos: (%.3f, %.3f, %.3f)\n", testPos.x, testPos.y, testPos.z);
            printf("  view pos: (%.3f, %.3f, %.3f, %.3f)\n", viewPos.x, viewPos.y, viewPos.z, viewPos.w);
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

    // Set tile info for tiled rasterizer
    tiledUniforms.numTilesX = (windowWidth + 15) / 16;
    tiledUniforms.numTilesY = (windowHeight + 15) / 16;
    tiledUniforms.numGaussians = (uint32_t)gaussianCount;

    // Get next drawable from Metal layer
    CA::MetalDrawable* drawable = (__bridge CA::MetalDrawable*)[metalLayer nextDrawable];
    if (!drawable) return;

    // Create a render target texture matching window size for tiled rasterizer
    // The tiled rasterizer writes to a texture, then we blit to the drawable
    if (!viewerRenderTarget ||
        viewerRenderTarget->width() != windowWidth ||
        viewerRenderTarget->height() != windowHeight) {
        if (viewerRenderTarget) viewerRenderTarget->release();

        MTL::TextureDescriptor* desc = MTL::TextureDescriptor::alloc()->init();
        desc->setWidth(windowWidth);
        desc->setHeight(windowHeight);
        desc->setPixelFormat(MTL::PixelFormatRGBA16Float);
        desc->setStorageMode(MTL::StorageModeShared);
        desc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
        viewerRenderTarget = metalDevice->newTexture(desc);
        desc->release();
    }

    // Use tiled rasterizer for rendering (same as training for consistency)
    tiledRasterizer->forward(commandQueue, gaussianBuffer, gaussianCount, tiledUniforms, viewerRenderTarget);

    // Render the texture to the drawable using a full-screen pass
    // (Can't blit directly due to pixel format mismatch)
    MTL::RenderPassDescriptor* renderPassDesc = MTL::RenderPassDescriptor::alloc()->init();
    renderPassDesc->colorAttachments()->object(0)->setTexture(drawable->texture());
    renderPassDesc->colorAttachments()->object(0)->setLoadAction(MTL::LoadActionDontCare);
    renderPassDesc->colorAttachments()->object(0)->setStoreAction(MTL::StoreActionStore);

    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::RenderCommandEncoder* encoder = commandBuffer->renderCommandEncoder(renderPassDesc);

    // Use the blit pipeline to copy the texture
    encoder->setRenderPipelineState(blitPSO);
    encoder->setFragmentTexture(viewerRenderTarget, 0);
    encoder->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));

    encoder->endEncoding();
    commandBuffer->presentDrawable(drawable);
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    renderPassDesc->release();
}

// GLFW framebuffer size callback
void MTLEngine::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    if (width == 0 || height == 0) return;
    
    // Update engine state
    MTLEngine* engine = static_cast<MTLEngine*>(glfwGetWindowUserPointer(window));
    engine->windowWidth = width;
    engine->windowHeight = height;

    // Update Metal layer size (width/height are already backing pixels from GLFW)
    engine->metalLayer.contentsScale = engine->metalWindow.backingScaleFactor;
    engine->metalLayer.frame = engine->metalWindow.contentView.bounds;
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
    // Use RGBA16Float to store actual HDR color values without clamping
    // This allows the loss to see true color values and provide proper gradient feedback
    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::alloc()->init();
    desc->setWidth(width);
    desc->setHeight(height);
    desc->setPixelFormat(MTL::PixelFormatRGBA16Float);
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

// Compute combined loss L1 and D-SSIM between rendered and ground truth textures
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
    
    // Compute L1 loss per pixel
    encoder->setComputePipelineState(lossComputePSO);
    encoder->setTexture(rendered, 0);
    encoder->setTexture(groundTruth, 1);
    encoder->setBuffer(lossBuffer, 0, 0);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    
    // Compute SSIM (D-SSIM) per pixel
    encoder->setComputePipelineState(ssimComputePSO);
    encoder->setTexture(rendered, 0);
    encoder->setTexture(groundTruth, 1);
    encoder->setBuffer(ssimBuffer, 0, 0);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    
    // Combine losses: (1-lambda)*L1 + lambda*D-SSIM
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
    
    // Reduce combined loss to single value
    encoder->setComputePipelineState(reductionPSO);
    encoder->setBuffer(combinedLossBuffer, 0, 0);
    encoder->setBuffer(totalLossBuffer, 0, 1);
    encoder->setBytes(&pixelCount, sizeof(uint32_t), 2);
    
    // reduceLoss is implemented as a single-threadgroup reduction
    // Dispatch exactly one 256-thread group.
    uint32_t reductionThreads = 256;
    encoder->dispatchThreads(MTL::Size(reductionThreads, 1, 1), MTL::Size(reductionThreads, 1, 1));
    
    // Finalize encoding
    encoder->endEncoding();
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();
    
    // Retrieve total loss
    float totalLoss = *(float*)totalLossBuffer->contents();
    return totalLoss / pixelCount;
}

// Encode loss computation without waiting call readLossValue() after a later GPU wait
void MTLEngine::encodeLossAsync(MTL::Texture* rendered, MTL::Texture* groundTruth) {
    uint32_t width = rendered->width();
    uint32_t height = rendered->height();
    uint32_t pixelCount = width * height;
    lastLossPixelCount = pixelCount;

    // Allocate buffers if needed
    if (!lossBuffer || lossBuffer->length() < pixelCount * sizeof(float)) {
        if (lossBuffer) lossBuffer->release();
        lossBuffer = metalDevice->newBuffer(pixelCount * sizeof(float), MTL::ResourceStorageModeShared);
    }
    if (!ssimBuffer || ssimBuffer->length() < pixelCount * sizeof(float)) {
        if (ssimBuffer) ssimBuffer->release();
        ssimBuffer = metalDevice->newBuffer(pixelCount * sizeof(float), MTL::ResourceStorageModeShared);
    }
    if (!combinedLossBuffer || combinedLossBuffer->length() < pixelCount * sizeof(float)) {
        if (combinedLossBuffer) combinedLossBuffer->release();
        combinedLossBuffer = metalDevice->newBuffer(pixelCount * sizeof(float), MTL::ResourceStorageModeShared);
    }
    if (!totalLossBuffer) {
        totalLossBuffer = metalDevice->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
    }

    float zero = 0.0f;
    memcpy(totalLossBuffer->contents(), &zero, sizeof(float));

    MTL::CommandBuffer* cmdBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = cmdBuffer->computeCommandEncoder();

    MTL::Size gridSize = MTL::Size(width, height, 1);
    MTL::Size threadGroupSize = MTL::Size(16, 16, 1);

    encoder->setComputePipelineState(lossComputePSO);
    encoder->setTexture(rendered, 0);
    encoder->setTexture(groundTruth, 1);
    encoder->setBuffer(lossBuffer, 0, 0);
    encoder->dispatchThreads(gridSize, threadGroupSize);

    encoder->setComputePipelineState(ssimComputePSO);
    encoder->setTexture(rendered, 0);
    encoder->setTexture(groundTruth, 1);
    encoder->setBuffer(ssimBuffer, 0, 0);
    encoder->dispatchThreads(gridSize, threadGroupSize);

    encoder->setComputePipelineState(combinedLossPSO);
    encoder->setTexture(rendered, 0);
    encoder->setTexture(groundTruth, 1);
    encoder->setBuffer(lossBuffer, 0, 0);
    encoder->setBuffer(ssimBuffer, 0, 1);
    encoder->setBuffer(combinedLossBuffer, 0, 2);
    encoder->setBytes(&lambdaDSSIM, sizeof(float), 3);
    encoder->dispatchThreads(gridSize, threadGroupSize);

    encoder->setComputePipelineState(reductionPSO);
    encoder->setBuffer(combinedLossBuffer, 0, 0);
    encoder->setBuffer(totalLossBuffer, 0, 1);
    encoder->setBytes(&pixelCount, sizeof(uint32_t), 2);
    // reduceLoss is implemented as a single-threadgroup reduction (no atomics)
    // Dispatch exactly one 256-thread group.
    uint32_t reductionThreads = 256;
    encoder->dispatchThreads(MTL::Size(reductionThreads, 1, 1), MTL::Size(reductionThreads, 1, 1));

    encoder->endEncoding();
    cmdBuffer->commit();
    // No wait - loss result readable after next GPU sync on same queue
}

// Read the loss value after a GPU wait has ensured the loss computation is complete
float MTLEngine::readLossValue() {
    float totalLoss = *(float*)totalLossBuffer->contents();
    return totalLoss / lastLossPixelCount;
}

// Perform a single training step on a specified training image
float MTLEngine::trainStep(size_t imageIndex,
                           float lr_position,
                           float lr_scale,
                           float lr_rotation,
                           float lr_opacity,
                           float lr_sh,
                           float lr_sh_rest) {
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
    
    // Print uniforms once
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
    
    // Print view-space coordinates for first few Gaussians on CPU side
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

        // Analyze render brightness for white spot debugging
        analyzeRenderBrightness(renderTarget, metalDevice, commandQueue);
    }
    saveCounter++;
    
    // Encode loss computation (async - no GPU wait, will complete before backward)
    encodeLossAsync(renderTarget, img.texture);

    // Backward pass combined L1 + D-SSIM pixel gradients; waits for loss + backward completion
    tiledRasterizer->backward(commandQueue, gaussianBuffer, gaussianGradients, gaussianCount,
                              uniforms, renderTarget, img.texture);

    // Read loss value (safe - backward wait guarantees loss command buffer completed)
    float loss = readLossValue();

    // If either pass hit a GPU abort, gaussianGradients holds partial or stale data.
    // Stepping Adam on it applies a garbage update to every Gaussian, and because the
    // damage is baked into the parameters it never washes out — repeated aborts are what
    // drive the loss back up late in training. Skip the update and drop the iteration.
    if (tiledRasterizer->lastPassFailed()) {
        std::cerr << "[GPU ABORT] skipping optimizer step for image " << imageIndex
                  << " (gradients unreliable); total aborts so far: "
                  << tiledRasterizer->getAbortCount() << std::endl;
        tiledRasterizer->clearPassFailed();
        return loss;
    }

    // Set actual Gaussian count before optimizer step
    optimizer->setGaussianCount(gaussianCount);

    // Start optimizer on GPU
    // Safe because optimizer writes to gaussianBuffer/Adam state,
    // while density accumulation reads gradientBuffer/projectedGaussians
    optimizer->step(commandQueue, gaussianBuffer, gaussianGradients,
                    lr_position,
                    lr_scale,
                    lr_rotation,
                    lr_opacity,
                    lr_sh,
                    lr_sh_rest,
                    /*maxLogScaleTrain=*/logf(fmaxf(1e-6f, 0.1f * sceneExtent)),
                    /*wait=*/false);

    // CPU density accumulation runs while optimizer GPU work executes
    densityController->accumulateGradients(commandQueue, gaussianGradients, gaussianCount);
    densityController->accumulateRadii(tiledRasterizer->getProjectedGaussians(), gaussianCount);

    // Wait for optimizer before reading Gaussian buffer
    optimizer->waitForLastStep();

    // Periodic stats every 100 images
    if (saveCounter % 100 == 0) {
        // Print average opacity and scale
        Gaussian* g = (Gaussian*)gaussianBuffer->contents();
        // Calculate average opacity and scale for a sample of Gaussians
        float avgOpacity = 0, avgScale = 0, maxScale = 0;
        int opAbove90 = 0, opAbove95 = 0, opAbove99 = 0;
        int scaleExploded = 0;
        // Sample up to 1000 Gaussians for stats
        const int sampleCount = std::min((size_t)1000, gaussianCount);
        for (int i = 0; i < sampleCount; i++) {
            float sigOp = 1.0f / (1.0f + exp(-g[i].opacity));
            avgOpacity += sigOp;
            float scale = (exp(g[i].scale.x) + exp(g[i].scale.y) + exp(g[i].scale.z)) / 3.0f;
            avgScale += scale;
            if (scale > maxScale) maxScale = scale;
            if (scale > 10.0f) scaleExploded++;

            // Count opacity distribution
            if (sigOp > 0.90f) opAbove90++;
            if (sigOp > 0.95f) opAbove95++;
            if (sigOp > 0.99f) opAbove99++;
        }
        printf("\n[iter %d] Gaussians: %zu | Avg opacity: %.3f | Avg scale: %.4f | Max scale: %.4f\n",
               saveCounter, gaussianCount, avgOpacity / sampleCount, avgScale / sampleCount, maxScale);
        printf("[Opacity] above 0.90: %d | above 0.95: %d | above 0.99: %d (of %d sampled)\n",
               opAbove90, opAbove95, opAbove99, sampleCount);

        // Scale explosion warning
        if (scaleExploded > 0 || maxScale > 20.0f) {
            printf("*** WARNING: %d Gaussians have scale > 10.0 (max: %.2f) ***\n",
                   scaleExploded, maxScale);
        }

        // Full-population health scan. The sampled stats above only look at the first
        // 1000 entries, which is a biased sample: density control reorders the array, so
        // the head skews toward long-lived Gaussians. These counters scan everything and
        // are the ones to watch across the 15000-iteration boundary, where density control
        // stops and nothing repairs damaged Gaussians any more.
        size_t nScaleFloor = 0, nOpacityRail = 0, nDead = 0, nNonFinite = 0;
        for (size_t i = 0; i < gaussianCount; i++) {
            const Gaussian& gi = g[i];

            if (!std::isfinite(gi.position.x) || !std::isfinite(gi.position.y) ||
                !std::isfinite(gi.position.z) || !std::isfinite(gi.opacity) ||
                !std::isfinite(gi.scale.x) || !std::isfinite(gi.scale.y) ||
                !std::isfinite(gi.scale.z)) {
                nNonFinite++;
                continue;
            }

            // Collapsed below exp(-6): the log-scale gradient is multiplied by scale, so
            // these have an effectively zero size gradient and can never grow back. The
            // +0.3 low-pass filter still renders them as a fixed ~0.55px screen dot whose
            // footprint does NOT shrink with distance, so they fit one view and are wrong
            // in every other. Monotonic growth here indicts MIN_SCALE_TRAIN being too low.
            float maxLogScale = std::max({gi.scale.x, gi.scale.y, gi.scale.z});
            if (maxLogScale < -6.0f) nScaleFloor++;

            // Pinned at the Adam opacity clamp.
            if (std::fabs(gi.opacity) >= 7.99f) nOpacityRail++;

            // Below the prune threshold but never pruned (density control ended).
            if (1.0f / (1.0f + std::exp(-gi.opacity)) < 0.005f) nDead++;
        }

        printf("[health] scale-floor: %zu (%.2f%%) | opacity-rail: %zu | dead-opacity: %zu | non-finite: %zu\n",
               nScaleFloor, 100.0 * nScaleFloor / std::max<size_t>(1, gaussianCount),
               nOpacityRail, nDead, nNonFinite);
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
    const float OPACITY_RESET_VALUE = -4.6f;
    
    const float POSITION_LR_INIT = 0.00016f * sceneExtent;
    const float POSITION_LR_FINAL = 0.0000016f * sceneExtent;
    // Scale/opacity now decay like position. Rationale: Adam gives every Gaussian a step
    // of magnitude ~lr regardless of how noisy its gradient is, so a constant scale_lr is
    // a fixed-size random walk in LOG-scale space for the many under-constrained (near-dead,
    // dilation-saturated) Gaussians. exp() of that walk drifts upward (E[exp(walk)]=exp(σ²/2)),
    // which is the scale bloat measured after iter 15000 once split/prune stop masking it.
    // Decaying the LR collapses the walk variance late in training; converged Gaussians have
    // tiny gradients anyway so nothing useful is lost. Start scale_lr at half the old value
    // (0.0025) because walk variance scales with the initial rate, not just the schedule.
    const float SCALE_LR_INIT = 0.0025f;
    const float SCALE_LR_FINAL = 0.00025f;
    const float ROTATION_LR = 0.001f;
    const float OPACITY_LR_INIT = 0.05f;
    const float OPACITY_LR_FINAL = 0.005f;
    const float SH_LR = 0.0025f;
    const float SH_REST_LR = SH_LR / 20.0f;  // 0.000125
    const size_t SH_DEGREE1_ACTIVATE_ITER = 1000;
    
    // Total iterations for LR scheduling
    const size_t totalExpectedIters = numEpochs * trainingImages.size();
    
    std::cout << "Learning rate decay over " << totalExpectedIters << " iterations:" << std::endl;
    std::cout << "  position " << POSITION_LR_INIT << " -> " << POSITION_LR_FINAL << std::endl;
    std::cout << "  scale    " << SCALE_LR_INIT << " -> " << SCALE_LR_FINAL << std::endl;
    std::cout << "  opacity  " << OPACITY_LR_INIT << " -> " << OPACITY_LR_FINAL << std::endl;
    
    // Start training timer
    auto startTime = std::chrono::high_resolution_clock::now();
    size_t totalIterations = 0;

    // Official 3DGS no opacity LR boost after reset Adam's adaptive rate
    // naturally handles the small sigmoid derivative at -4.6

    // Image order shuffling
    std::vector<size_t> imageOrder(trainingImages.size());
    std::iota(imageOrder.begin(), imageOrder.end(), 0);
    std::mt19937 rng(42);  // Fixed seed for reproducibility

    // Training loop
    for (size_t epoch = 0; epoch < numEpochs; epoch++) {
        float epochLoss = 0.0f;
        auto epochStart = std::chrono::high_resolution_clock::now();

        // Shuffle image order each epoch
        std::shuffle(imageOrder.begin(), imageOrder.end(), rng);

        for (size_t imgIdx = 0; imgIdx < trainingImages.size(); imgIdx++) {
            size_t shuffledIdx = imageOrder[imgIdx];
            // Autorelease pool to prevent memory buildup from temporary Metal objects
            NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
            
            // Compute decayed learning rates
            float currentPositionLR = exponentialLRDecay(POSITION_LR_INIT, POSITION_LR_FINAL,
                                                          totalIterations, totalExpectedIters);
            // Scale/opacity decay on the same exponential schedule to damp the late-training
            // Adam random walk that inflates scale and drains opacity (see LR constants above).
            float currentScaleLR = exponentialLRDecay(SCALE_LR_INIT, SCALE_LR_FINAL,
                                                      totalIterations, totalExpectedIters);
            float currentOpacityLR = exponentialLRDecay(OPACITY_LR_INIT, OPACITY_LR_FINAL,
                                                        totalIterations, totalExpectedIters);

            // Activate degree-1 SH at iter 1000 (official 3DGS schedule)
            float currentSHRestLR = (totalIterations >= SH_DEGREE1_ACTIVATE_ITER) ? SH_REST_LR : 0.0f;
            float loss = trainStep(shuffledIdx, currentPositionLR, currentScaleLR, ROTATION_LR, currentOpacityLR, SH_LR, currentSHRestLR);
            epochLoss += loss;
            totalIterations++;
            
            // Progress update every 20 images
            if (imgIdx % 20 == 0) {
                std::cout << "\rEpoch " << epoch << " [" << imgIdx << "/" << trainingImages.size()
                          << "] Iter: " << totalIterations 
                          << " | Loss: " << std::fixed << std::setprecision(4) << (epochLoss / (imgIdx + 1)) 
                          << std::flush;
            }
            
            // Detailed progress every 100 iterations
            if (totalIterations % 100 == 0) {
                std::cout << "\n[Iter " << totalIterations << "] "
                          << "Loss: " << std::fixed << std::setprecision(4) << loss
                          << " | Gaussians: " << gaussianCount 
                          << " | Position LR: " << std::scientific << std::setprecision(2) << currentPositionLR
                          << std::endl;
            }
            
            // Enable screen-space pruning after OPACITY_RESET_INTERVAL iterations
            bool enableScreenPruning = (totalIterations > OPACITY_RESET_INTERVAL);

            // Density control condition - match official 3DGS
            // Only start density control AFTER densify_from_iter, not before
            // Skip on opacity reset iterations to avoid wasteful clone-then-reset
            bool isOpacityResetIter = (totalIterations % OPACITY_RESET_INTERVAL == 0 &&
                                       totalIterations > 0 &&
                                       totalIterations < DENSIFY_UNTIL_ITER);

            bool shouldDensify = (totalIterations >= DENSIFY_FROM_ITER &&
                                  totalIterations < DENSIFY_UNTIL_ITER &&
                                  totalIterations % densityControlInterval == 0 &&
                                  !isOpacityResetIter);
            
            // Apply density control if needed
            if (shouldDensify) {
                // Get camera parameters for screen-space pruning
                const TrainingImage& currentImg = trainingImages[shuffledIdx];
                const ColmapCamera& cam = colmapData.cameras.at(currentImg.cameraId);
                
                uint32_t actualWidth = currentImg.texture->width();
                float scaleX = (float)actualWidth / (float)cam.width;
                float focalLength = cam.fx * scaleX;
                float imageWidth = (float)actualWidth;
                // Approximate scene extent from camera positions
                float avgDepth = 4.0f * sceneExtent;

                if (totalIterations % 1000 == 0) {
                    std::cout << "Density params: fx=" << focalLength
                            << " width=" << imageWidth
                            << " avgDepth=" << avgDepth
                            << " sceneExtent=" << sceneExtent
                            << " screenPrune=" << (enableScreenPruning ? "ON" : "OFF") << std::endl;
                }
                
                // Save count before density control to detect new Gaussians
                size_t oldCount = gaussianCount;
                
                // Pass enableScreenPruning to density controller
                // Official uses size_threshold=20 pixels when enabled
                // FIX #5: Use consistent opacity threshold (0.005) across all checks
                DensityStats stats = densityController->apply(commandQueue, gaussianBuffer, positionBuffer,
                                         nullptr, gaussianCount, totalIterations,
                                         0.0002f,  // Fixed threshold (official 3DGS)
                                         0.005f,   // min_opacity (matches OPACITY_PRUNE_THRESHOLD)
                                         0.1f * sceneExtent,  // max_scale
                                         focalLength,
                                         imageWidth,
                                         avgDepth);

                // Resize gradient buffer if needed
                if (gaussianGradients->length() < gaussianCount * sizeof(GaussianGradients)) {
                    gaussianGradients->release();
                    gaussianGradients = metalDevice->newBuffer(gaussianCount * sizeof(GaussianGradients),
                                                               MTL::ResourceStorageModeShared);
                }

                // Remap optimizer momentum - must handle the case where indexMapping is empty
                // (e.g., all Gaussians split). In that case, remapMomentum creates zeroed buffers.
                // The old fallback was WRONG: it assumed new Gaussians are at the end, but after
                // density control they're interleaved throughout the array.
                bool hadChanges = (stats.numPruned > 0 || stats.numCloned > 0 || stats.numSplit > 0);
                if (optimizer && hadChanges) {
                    optimizer->remapMomentum(stats.indexMapping, gaussianCount);
                }
            }
            
            // Opacity reset condition
            if (isOpacityResetIter) {
                std::cout << std::endl << "Resetting opacity at iteration " << totalIterations << std::endl;

                // Clamp opacities in Gaussian buffer
                Gaussian* gaussians = (Gaussian*)gaussianBuffer->contents();
                for (size_t i = 0; i < gaussianCount; i++) {
                    // Clamp opacity to max OPACITY_RESET_VALUE
                    if (gaussians[i].opacity > OPACITY_RESET_VALUE) {
                        gaussians[i].opacity = OPACITY_RESET_VALUE;
                    }
                }

                // Reset opacity momentum only — matches official 3DGS behavior.
                // replace_tensor_to_optimizer zeros exp_avg and exp_avg_sq for opacity only.
                // All other Adam state (position, scale, rotation, SH) is preserved.
                optimizer->resetOpacityMomentum();

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
    
    // Final prune. Density control only runs while iteration < DENSIFY_UNTIL_ITER (15000),
    // so Gaussians that fade below the opacity threshold during the remaining ~15000
    // iterations are never removed and ship in the PLY as invisible dead weight.
    {
        Gaussian* gaussians = (Gaussian*)gaussianBuffer->contents();
        const float minOpacity = 0.005f;

        size_t writeIdx = 0;
        for (size_t i = 0; i < gaussianCount; i++) {
            const Gaussian& g = gaussians[i];

            bool invalid = !std::isfinite(g.position.x) || !std::isfinite(g.position.y) ||
                           !std::isfinite(g.position.z) || !std::isfinite(g.opacity) ||
                           !std::isfinite(g.scale.x) || !std::isfinite(g.scale.y) ||
                           !std::isfinite(g.scale.z);

            float opacity = 1.0f / (1.0f + expf(-g.opacity));
            if (invalid || opacity < minOpacity) continue;

            gaussians[writeIdx++] = g;
        }

        size_t removed = gaussianCount - writeIdx;
        std::cout << "Final prune: removed " << removed << " Gaussians below opacity "
                  << minOpacity << " (" << gaussianCount << " -> " << writeIdx << ")" << std::endl;
        gaussianCount = writeIdx;
        updatePositionBuffer();
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
        
        // Get actual image texture size (matching training behavior)
        uint32_t actualWidth = img.texture->width();
        uint32_t actualHeight = img.texture->height();

        // Scale factor from COLMAP resolution to actual image resolution
        float scaleX = (float)actualWidth / (float)cam.width;
        float scaleY = (float)actualHeight / (float)cam.height;

        // Scale camera intrinsics to match actual image size (matching training)
        float scaledFx = cam.fx * scaleX;
        float scaledFy = cam.fy * scaleY;
        float scaledCx = cam.cx * scaleX;
        float scaledCy = cam.cy * scaleY;

        // Create render target matching the actual image size
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
        
        // Set up uniforms for this view
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

        // Set tile info using actual image dimensions
        uniforms.numTilesX = (actualWidth + 15) / 16;
        uniforms.numTilesY = (actualHeight + 15) / 16;
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

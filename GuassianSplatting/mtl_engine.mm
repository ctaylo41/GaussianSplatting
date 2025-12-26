//
//  mtl_engine.mm
//  Metal-Guide
//

#include "mtl_engine.hpp"
#include <iostream>

void MTLEngine::init() {
    initDevice();
    initCommandQueue();
    initWindow();
    createDepthTexture();
    loadShaders();
    createPipeline();
    uniformBuffer = metalDevice->newBuffer(sizeof(Uniforms), MTL::ResourceStorageModeShared);
}

void MTLEngine::run(Camera& camera) {
    while (!glfwWindowShouldClose(glfwWindow)) {
        glfwPollEvents();
        render(camera);
    }
}

void MTLEngine::cleanup() {
    glfwTerminate();
    if(gaussianBuffer) {
        gaussianBuffer->release();
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

void MTLEngine::loadGaussians(const std::vector<Gaussian>& gaussians) {
    gaussianCount = gaussians.size();
    gaussianBuffer = metalDevice->newBuffer(gaussians.data(), gaussianCount*sizeof(gaussians[0]),MTL::ResourceStorageModeShared);
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
    renderPipelineDescriptor->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    renderPipelineDescriptor->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);
    
    
    NS::Error* error;
    metalRenderPSO = metalDevice->newRenderPipelineState(renderPipelineDescriptor, &error);
    
    if(metalRenderPSO == nil) {
        std::cout << "Error creating render pipeline state: " << error << std::endl;
        std::exit(0);
    }
    
    MTL::DepthStencilDescriptor* depthStencilDescriptor = MTL::DepthStencilDescriptor::alloc()->init();
    depthStencilDescriptor->setDepthCompareFunction(MTL::CompareFunctionLessEqual);
    depthStencilDescriptor->setDepthWriteEnabled(true);
    depthStencilState = metalDevice->newDepthStencilState(depthStencilDescriptor);
    
    depthStencilDescriptor->release();
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
    Uniforms uniforms;
    uniforms.viewProjection = matrix_multiply(camera.get_projection_matrix(), camera.get_view_matrix());
    memcpy(uniformBuffer->contents(), &uniforms, sizeof(Uniforms));
    
    CA::MetalDrawable* drawable = (__bridge CA::MetalDrawable*)[metalLayer nextDrawable];
    if (!drawable) return;
    
    MTL::RenderPassDescriptor* renderPassDesc = MTL::RenderPassDescriptor::alloc()->init();
    renderPassDesc->colorAttachments()->object(0)->setTexture(drawable->texture());
    renderPassDesc->colorAttachments()->object(0)->setLoadAction(MTL::LoadActionClear);
    renderPassDesc->colorAttachments()->object(0)->setStoreAction(MTL::StoreActionStore);
    renderPassDesc->colorAttachments()->object(0)->setClearColor(MTL::ClearColor(1.0, 0.0, 0.0, 1.0));

    
    renderPassDesc->depthAttachment()->setTexture(depthTexture);
    renderPassDesc->depthAttachment()->setLoadAction(MTL::LoadActionClear);
    renderPassDesc->depthAttachment()->setStoreAction(MTL::StoreActionDontCare);
    renderPassDesc->depthAttachment()->setClearDepth(1.0);
    
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::RenderCommandEncoder* encoder = commandBuffer->renderCommandEncoder(renderPassDesc);
    
    encoder->setRenderPipelineState(metalRenderPSO);
    encoder->setDepthStencilState(depthStencilState);
    encoder->setVertexBuffer(gaussianBuffer,0, 0);
    encoder->setVertexBuffer(uniformBuffer, 0, 1);
    encoder->drawPrimitives(MTL::PrimitiveTypePoint, NS::UInteger(0), gaussianCount);
    
    encoder->endEncoding();
    commandBuffer->presentDrawable(drawable);
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    renderPassDesc->release();
    
}

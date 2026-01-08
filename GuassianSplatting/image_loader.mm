//
//  image_loader.mm
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-27.
//

#include "image_loader.hpp"
#import "stbi_image.h"
#include <iostream>

// Load an image from disk and create a Metal texture
MTL::Texture* loadImageAsTexture(MTL::Device* device, const std::string& path) {
    int width, height, channels;
    
    // Load image using stb_image
    unsigned char* image = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    assert(image!=NULL);
    
    size_t bytesPerRow = 4 * width;
    // Create Metal texture     
    MTL::TextureDescriptor* textureDescriptor = MTL::TextureDescriptor::alloc()->init();
    // Set texture properties
    textureDescriptor->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
    textureDescriptor->setWidth(width);
    textureDescriptor->setHeight(height);
    // Set Usage to Shader Read
    textureDescriptor->setUsage(MTL::TextureUsageShaderRead);
    // Create texture
    MTL::Texture* texture = device->newTexture(textureDescriptor);
    
    // Copy image data into texture
    MTL::Region region = MTL::Region(0,0,0, width, height, 1);
    texture->replaceRegion(region, 0, image, bytesPerRow);
    
    // Release descriptor and free image memory
    textureDescriptor->release();
    stbi_image_free(image);
    
    return texture;
}

// Load all training images based on COLMAP data
std::vector<TrainingImage> loadTrainingImages(MTL::Device* device, const ColmapData& colmap, const std::string& imagePath) {
    std::vector<TrainingImage> trainingImages;
    trainingImages.reserve(colmap.images.size());
    
    // Iterate through COLMAP images and load each one
    for (const auto& img : colmap.images) {
        std::string fullPath = imagePath + "/" + img.filename;
        
        // Load image as Metal texture
        MTL::Texture* texture = loadImageAsTexture(device, fullPath);
        if(!texture) {
            std::cerr << "Failed to load: " << fullPath << std::endl;
            continue;
        }
        
        // Create TrainingImage struct
        TrainingImage ti;
        ti.imageId = img.id;
        ti.cameraId = img.cameraId;
        ti.texture = texture;
        ti.rotation = img.rotation;
        ti.translation = img.translation;
        
        trainingImages.push_back(ti);
    }
    
    std::cout << "Loaded " << trainingImages.size() << " training images" << std::endl;
    
    return trainingImages;
}

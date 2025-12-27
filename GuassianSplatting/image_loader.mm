//
//  image_loader.mm
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-27.
//

#include "image_loader.hpp"
#import "stbi_image.h"
#include <iostream>

MTL::Texture* loadImageAsTexture(MTL::Device* device, const std::string& path) {
    int width, height, channels;
    
    unsigned char* image = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    assert(image!=NULL);
    
    MTL::TextureDescriptor* textureDescriptor = MTL::TextureDescriptor::alloc()->init();
    textureDescriptor->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
    textureDescriptor->setWidth(width);
    textureDescriptor->setHeight(height);
    textureDescriptor->setUsage(MTL::TextureUsageShaderRead);
    
    MTL::Texture* texture = device->newTexture(textureDescriptor);
    
    MTL::Region region = MTL::Region(0,0,0, width, height, 1);
    NS::UInteger bytesPerRow = 4 * width;
    texture->replaceRegion(region, 0, image, bytesPerRow);
    
    textureDescriptor->release();
    
    stbi_image_free(image);
    
    return texture;
}

std::vector<TrainingImage> loadTrainingImages(MTL::Device* device, const ColmapData& colmap, const std::string& imagePath) {
    std::vector<TrainingImage> trainingImages;
    trainingImages.reserve(colmap.images.size());
    
    for (const auto& img : colmap.images) {
        std::string fullPath = imagePath + "/" + img.filename;
        
        MTL::Texture* texture = loadImageAsTexture(device, fullPath);
        if(!texture) {
            std::cerr << "Failed to load: " << fullPath << std::endl;
            continue;
        }
        
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

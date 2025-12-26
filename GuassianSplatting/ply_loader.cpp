//
//  ply_loader.cpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-26.
//


#include "ply_loader.hpp"
#include "tinyply.h"
#include <fstream>
#include <iostream>

std::vector<Gaussian> load_ply(const std::string& file_path) {
    std::vector<Gaussian> gaussians;
    
    try {
        std::ifstream file(file_path, std::ios::binary);
        if(!file.is_open()) {
            std::cerr << "Failed to open file: " << file_path << std::endl;
            return gaussians;
        }
        
        tinyply::PlyFile ply_file;
        ply_file.parse_header(file);
        
        std::shared_ptr<tinyply::PlyData> positions, scales, rotations, opacities, sh_dcs;
        
        try {
            positions = ply_file.request_properties_from_element("vertex", {"x","y","z"});
        } catch (...) {
            std::cerr << "Failed to load positions" << std::endl;
            return gaussians;
        }
        
        try {
            scales = ply_file.request_properties_from_element("vertex", {"scale_0", "scale_1","scale_2"});
        } catch (...) {
            std::cerr << "Failed to load scales" << std::endl;
            return gaussians;
        }
        
        try {
            rotations = ply_file.request_properties_from_element("vertex", {"rot_0", "rot_1", "rot_2", "rot_3"});
        } catch (...) {
            std::cerr << "Failed to load rotations" << std::endl;
            return gaussians;
        }
        
        try {
            opacities = ply_file.request_properties_from_element("vertex", {"opacity"});
        } catch (...) {
            std::cerr << "Failed to load opacities" << std::endl;
            return gaussians;
        }
        
        try {
            sh_dcs = ply_file.request_properties_from_element("vertex", {"f_dc_0","f_dc_1","f_dc_2"});
        } catch (...) {
            std::cerr << "Failed to load SH DC coefficients" << std::endl;
            return gaussians;
        }
        
        ply_file.read(file);
        
        size_t vertex_count = positions->count;
        std::cout << "Loading " << vertex_count << " gaussians..." << std::endl;
        
        const float* pos_data = reinterpret_cast<const float*>(positions->buffer.get());
        const float* scale_data = reinterpret_cast<const float*>(scales->buffer.get());
        const float* rot_data = reinterpret_cast<const float*>(rotations->buffer.get());
        const float* opacity_data = reinterpret_cast<const float*>(opacities->buffer.get());
        const float* sh_dc_data = reinterpret_cast<const float*>(sh_dcs->buffer.get());
        
        gaussians.reserve(vertex_count);
        
        for(size_t i=0;i<vertex_count;i++) {
            Gaussian g;
            
            g.position = simd_make_float3(pos_data[i*3 + 0],
                                          pos_data[i*3 + 1],
                                          pos_data[i*3 + 2]);
            
            g.scale = simd_make_float3(std::exp(scale_data[i*3 + 0]),
                                       std::exp(scale_data[i*3 + 1]),
                                       std::exp(scale_data[i*3 + 2]));
            
            float qx = rot_data[i*4 + 0];
            float qy = rot_data[i*4 + 1];
            float qz = rot_data[i*4 + 2];
            float qw = rot_data[i*4 + 3];
            
            float q_len = std::sqrt(qx*qx + qy*qy + qz*qz + qw*qw);
            if (q_len > 0.0f) {
                qx /= q_len;
                qy /= q_len;
                qz /= q_len;
                qw /= q_len;
            }
            g.rotation = simd_make_float4(qx,qy,qz,qw);
            
            float raw_opacity = opacity_data[i];
            g.opacity = 1.0f / (1.0f + std::exp(-raw_opacity));
            
            const float SH_CO = 0.2820948f;
            g.sh_dc = simd_make_float3(sh_dc_data[i*3 + 0] * SH_CO + 0.5f,
                                       sh_dc_data[i*3 + 1] * SH_CO + 0.5f,
                                       sh_dc_data[i*3 + 2] * SH_CO + 0.5f);
            
            gaussians.push_back(g);
            
        }
        
        std::cout << "Loaded " << gaussians.size() << " gaussians successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading ply: " << e.what() << std::endl;
    }
    return gaussians;
}

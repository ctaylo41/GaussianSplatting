//
//  ply_loader.cpp
//  GuassianSplatting
//
//  CRITICAL: This loader must match the internal representation expected by shaders:
//  - Scale: stored in LOG space (shader applies exp())
//  - Opacity: stored as RAW pre-sigmoid value (shader applies sigmoid())
//  - Rotation: stored as (w,x,y,z) in float4 where .x=w, .y=x, .z=y, .w=z
//
//  External PLY Compatibility:
//  - Standard 3DGS PLY files store scales already in log space
//  - Standard 3DGS PLY files store opacity in raw (pre-sigmoid) space
//  - Rotation stored as (rot_0=w, rot_1=x, rot_2=y, rot_3=z)
//

#include "ply_loader.hpp"
#include "tinyply.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <float.h>
#include <algorithm>

// Detect if scales appear to be in linear space rather than log space
// Log-space scales from 3DGS are typically in range [-10, 2]
// Linear scales would be small positive numbers like [0.001, 1.0]
//
// Standard 3DGS PLY files store scales in LOG SPACE.
// The log scale typically ranges from -8 to 0, giving linear scales of 0.0003 to 1.0
// If we see scales that are all positive but less than 1, they might be LINEAR (not log)
// If we see scales with negative values, they are LOG SPACE (standard)
// If we see scales with positive values > 2, they are LOG SPACE (would give huge linear scales)
bool detectLinearScales(const float* scale_data, size_t count, float& outMinVal, float& outMaxVal) {
    if (count == 0) return false;
    
    int positiveCount = 0;
    int negativeCount = 0;
    float maxVal = -FLT_MAX;
    float minVal = FLT_MAX;
    
    size_t sampleSize = std::min(count, (size_t)1000);
    for (size_t i = 0; i < sampleSize; i++) {
        for (int j = 0; j < 3; j++) {
            float val = scale_data[i * 3 + j];
            if (val > 0) positiveCount++;
            if (val < 0) negativeCount++;
            maxVal = std::max(maxVal, val);
            minVal = std::min(minVal, val);
        }
    }
    
    outMinVal = minVal;
    outMaxVal = maxVal;
    
    std::cout << "Scale analysis: min=" << minVal << " max=" << maxVal 
              << " positive=" << positiveCount << " negative=" << negativeCount << std::endl;
    
    // If we have any negative values, it's definitely log space
    if (negativeCount > 0) {
        return false; // Log space
    }
    
    // All positive: check the range
    // Log scales > 2 would give exp() > 7.4 which is large
    // Linear scales are typically small (< 1)
    // If max is small and positive, likely linear
    if (maxVal <= 1.0f && minVal > 0.0f) {
        return true;  // Likely linear
    }
    
    // Otherwise, assume log space (positive log scales are valid)
    return false;
}

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
        
        std::shared_ptr<tinyply::PlyData> sh_rest;
        
        try {
            sh_rest = ply_file.request_properties_from_element("vertex", {
                "f_rest_0", "f_rest_1", "f_rest_2",
                "f_rest_3", "f_rest_4", "f_rest_5",
                "f_rest_6", "f_rest_7", "f_rest_8"
            });
        } catch(...) {
            std::cout << "No SH rest coefficients, using DC only" << std::endl;
            sh_rest = nullptr;
        }
        
        ply_file.read(file);
        
        size_t vertex_count = positions->count;
        std::cout << "Loading " << vertex_count << " gaussians..." << std::endl;
        
        const float* pos_data = reinterpret_cast<const float*>(positions->buffer.get());
        const float* scale_data = reinterpret_cast<const float*>(scales->buffer.get());
        const float* rot_data = reinterpret_cast<const float*>(rotations->buffer.get());
        const float* opacity_data = reinterpret_cast<const float*>(opacities->buffer.get());
        const float* sh_dc_data = reinterpret_cast<const float*>(sh_dcs->buffer.get());
        const float* sh_rest_data = sh_rest ? reinterpret_cast<const float*>(sh_rest->buffer.get()) : nullptr;

        // Auto-detect scale format
        float scaleMin, scaleMax;
        bool isLinearScale = detectLinearScales(scale_data, vertex_count, scaleMin, scaleMax);
        if (isLinearScale) {
            std::cout << "DETECTED: Linear scale format - will convert to log space" << std::endl;
        } else {
            std::cout << "DETECTED: Log scale format (standard 3DGS)" << std::endl;
            std::cout << "  Log scale range: [" << scaleMin << ", " << scaleMax << "]" << std::endl;
            std::cout << "  Linear scale range: [" << std::exp(scaleMin) << ", " << std::exp(scaleMax) << "]" << std::endl;
        }

        gaussians.reserve(vertex_count);
        
        int numSkipped = 0;
        
        for(size_t i = 0; i < vertex_count; i++) {
            Gaussian g;
            
            // ===== POSITION: Direct copy =====
            g.position = simd_make_float3(pos_data[i*3 + 0],
                                          pos_data[i*3 + 1],
                                          pos_data[i*3 + 2]);
            
            // Skip NaN/Inf positions
            if (std::isnan(g.position.x) || std::isnan(g.position.y) || std::isnan(g.position.z) ||
                std::isinf(g.position.x) || std::isinf(g.position.y) || std::isinf(g.position.z) ||
                std::abs(g.position.x) > 1e6 || std::abs(g.position.y) > 1e6 || std::abs(g.position.z) > 1e6) {
                numSkipped++;
                continue;
            }
            
            // ===== SCALE: Handle both linear and log formats =====
            float s0 = scale_data[i*3 + 0];
            float s1 = scale_data[i*3 + 1];
            float s2 = scale_data[i*3 + 2];
            
            if (isLinearScale) {
                // Convert linear scales to log space
                // Add small epsilon to avoid log(0)
                s0 = std::log(std::max(s0, 1e-8f));
                s1 = std::log(std::max(s1, 1e-8f));
                s2 = std::log(std::max(s2, 1e-8f));
            }
            
            // Clamp scales to reasonable range for viewing
            // Higher limit for external PLY compatibility (training uses stricter 4.0)
            const float MAX_LOG_SCALE = 8.0f;
            s0 = std::clamp(s0, -MAX_LOG_SCALE, MAX_LOG_SCALE);
            s1 = std::clamp(s1, -MAX_LOG_SCALE, MAX_LOG_SCALE);
            s2 = std::clamp(s2, -MAX_LOG_SCALE, MAX_LOG_SCALE);
            
            g.scale = simd_make_float3(s0, s1, s2);
            
            // ===== ROTATION: PLY format is (rot_0=w, rot_1=x, rot_2=y, rot_3=z) =====
            float qw = rot_data[i*4 + 0];  // rot_0 = w
            float qx = rot_data[i*4 + 1];  // rot_1 = x
            float qy = rot_data[i*4 + 2];  // rot_2 = y
            float qz = rot_data[i*4 + 3];  // rot_3 = z
            
            // Normalize quaternion
            float q_len = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
            if (q_len > 0.0001f) {
                qw /= q_len; qx /= q_len; qy /= q_len; qz /= q_len;
            } else {
                qw = 1.0f; qx = 0.0f; qy = 0.0f; qz = 0.0f;
            }
            
            // Store as (w, x, y, z) - shader expects q.x=w, q.y=x, q.z=y, q.w=z
            g.rotation = simd_make_float4(qw, qx, qy, qz);
            
            // ===== OPACITY: Keep as RAW - DO NOT apply sigmoid()! =====
            g.opacity = opacity_data[i];
            
            // ===== SH COEFFICIENTS =====
            g.sh[0] = sh_dc_data[i * 3 + 0];  // R DC
            g.sh[4] = sh_dc_data[i * 3 + 1];  // G DC
            g.sh[8] = sh_dc_data[i * 3 + 2];  // B DC
            
            if (sh_rest_data) {
                g.sh[1] = sh_rest_data[i * 9 + 0];
                g.sh[5] = sh_rest_data[i * 9 + 1];
                g.sh[9] = sh_rest_data[i * 9 + 2];
                g.sh[2] = sh_rest_data[i * 9 + 3];
                g.sh[6] = sh_rest_data[i * 9 + 4];
                g.sh[10] = sh_rest_data[i * 9 + 5];
                g.sh[3] = sh_rest_data[i * 9 + 6];
                g.sh[7] = sh_rest_data[i * 9 + 7];
                g.sh[11] = sh_rest_data[i * 9 + 8];
            } else {
                for (int j = 1; j < 4; j++) {
                    g.sh[j] = 0; g.sh[j+4] = 0; g.sh[j+8] = 0;
                }
            }
            
            gaussians.push_back(g);
        }
        
        std::cout << "Loaded " << gaussians.size() << " gaussians successfully" << std::endl;
        if (numSkipped > 0) {
            std::cout << "Skipped " << numSkipped << " invalid gaussians" << std::endl;
        }
        
        // Debug first Gaussian with more detail
        if (!gaussians.empty()) {
            const Gaussian& g = gaussians[0];
            std::cout << "First Gaussian:" << std::endl;
            std::cout << "  Position: (" << g.position.x << ", " << g.position.y << ", " << g.position.z << ")" << std::endl;
            std::cout << "  Scale (log): (" << g.scale.x << ", " << g.scale.y << ", " << g.scale.z << ")" << std::endl;
            std::cout << "  Scale (exp): (" << std::exp(g.scale.x) << ", " << std::exp(g.scale.y) << ", " << std::exp(g.scale.z) << ")" << std::endl;
            std::cout << "  Rotation (wxyz): (" << g.rotation.x << ", " << g.rotation.y << ", " << g.rotation.z << ", " << g.rotation.w << ")" << std::endl;
            std::cout << "  Opacity (raw): " << g.opacity << std::endl;
            std::cout << "  Opacity (sigmoid): " << (1.0f / (1.0f + std::exp(-g.opacity))) << std::endl;
            
            // Show SH DC coefficients and expected color
            const float SH_C0 = 0.28209479177387814f;
            std::cout << "  SH DC (indices 0,4,8): (" << g.sh[0] << ", " << g.sh[4] << ", " << g.sh[8] << ")" << std::endl;
            float r = SH_C0 * g.sh[0] + 0.5f;
            float gr = SH_C0 * g.sh[4] + 0.5f;
            float b = SH_C0 * g.sh[8] + 0.5f;
            std::cout << "  Expected color: (" << r << ", " << gr << ", " << b << ")" << std::endl;
        }
        
        // Also show stats across all Gaussians
        if (gaussians.size() > 1) {
            float minScale = FLT_MAX, maxScale = -FLT_MAX;
            float minOpacity = FLT_MAX, maxOpacity = -FLT_MAX;
            for (const auto& g : gaussians) {
                minScale = std::min(minScale, std::min({g.scale.x, g.scale.y, g.scale.z}));
                maxScale = std::max(maxScale, std::max({g.scale.x, g.scale.y, g.scale.z}));
                minOpacity = std::min(minOpacity, g.opacity);
                maxOpacity = std::max(maxOpacity, g.opacity);
            }
            std::cout << "Scale (log) range: [" << minScale << ", " << maxScale << "]" << std::endl;
            std::cout << "Opacity (raw) range: [" << minOpacity << ", " << maxOpacity << "]" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading ply: " << e.what() << std::endl;
    }
    return gaussians;
}

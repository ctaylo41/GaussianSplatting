//
//  ply_exporter.hpp
//  GaussianSplatting
//
//  Export trained Gaussians to PLY format compatible with standard 3DGS viewers
//
//  IMPORTANT: Export format must match what loader expects:
//  - Scale: LOG space (stored as-is from internal representation)
//  - Opacity: RAW pre-sigmoid (stored as-is)
//  - Rotation: (rot_0=w, rot_1=x, rot_2=y, rot_3=z)
//

#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include "ply_loader.hpp"

// PLY Exporter class
class PLYExporter {
public:
    // Export Gaussians to PLY file
    static bool exportPLY(const std::string& filename,
                          const Gaussian* gaussians,
                          size_t count) {
        
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return false;
        }
        
        // Count valid Gaussians
        size_t validCount = 0;
        for (size_t i = 0; i < count; i++) {
            const Gaussian& g = gaussians[i];
            if (!std::isnan(g.position.x) && !std::isinf(g.position.x) &&
                std::abs(g.position.x) < 1e6) {
                validCount++;
            }
        }
        
        // Write PLY header
        file << "ply\n";
        file << "format binary_little_endian 1.0\n";
        file << "element vertex " << validCount << "\n";
        
        // Position
        file << "property float x\n";
        file << "property float y\n";
        file << "property float z\n";
        
        // Normals (unused but expected)
        file << "property float nx\n";
        file << "property float ny\n";
        file << "property float nz\n";
        
        // SH DC terms
        file << "property float f_dc_0\n";
        file << "property float f_dc_1\n";
        file << "property float f_dc_2\n";
        
        // SH rest (degree 1)
        for (int i = 0; i < 9; i++) {
            file << "property float f_rest_" << i << "\n";
        }
        
        // Opacity (raw)
        file << "property float opacity\n";
        
        // Scale (log)
        file << "property float scale_0\n";
        file << "property float scale_1\n";
        file << "property float scale_2\n";
        
        // Rotation (quaternion w,x,y,z)
        file << "property float rot_0\n";
        file << "property float rot_1\n";
        file << "property float rot_2\n";
        file << "property float rot_3\n";
        
        file << "end_header\n";
        
        // Write binary data
        float zero = 0.0f;
        
        for (size_t i = 0; i < count; i++) {
            const Gaussian& g = gaussians[i];
            
            // Skip invalid
            if (std::isnan(g.position.x) || std::isinf(g.position.x) ||
                std::abs(g.position.x) > 1e6) {
                continue;
            }
            
            // Position
            float pos_x = g.position.x;
            float pos_y = g.position.y;
            float pos_z = g.position.z;
            file.write(reinterpret_cast<const char*>(&pos_x), sizeof(float));
            file.write(reinterpret_cast<const char*>(&pos_y), sizeof(float));
            file.write(reinterpret_cast<const char*>(&pos_z), sizeof(float));
            
            // Normals (zeros)
            file.write(reinterpret_cast<const char*>(&zero), sizeof(float));
            file.write(reinterpret_cast<const char*>(&zero), sizeof(float));
            file.write(reinterpret_cast<const char*>(&zero), sizeof(float));
            
            // SH DC terms 
            float sh_dc_0 = g.sh[0];
            float sh_dc_1 = g.sh[4];
            float sh_dc_2 = g.sh[8];
            file.write(reinterpret_cast<const char*>(&sh_dc_0), sizeof(float));
            file.write(reinterpret_cast<const char*>(&sh_dc_1), sizeof(float));
            file.write(reinterpret_cast<const char*>(&sh_dc_2), sizeof(float));
            
            // SH rest interleaved by coefficient
            float sh_rest[9] = {
                // coef 1: R, G, B
                g.sh[1], g.sh[5], g.sh[9],    
                // coef 2: R, G, B
                g.sh[2], g.sh[6], g.sh[10],   
                // coef 3: R, G, B
                g.sh[3], g.sh[7], g.sh[11]    
            };
            for (int j = 0; j < 9; j++) {
                file.write(reinterpret_cast<const char*>(&sh_rest[j]), sizeof(float));
            }
            
            // Opacity (raw, NOT sigmoid)
            float opacity = g.opacity;
            file.write(reinterpret_cast<const char*>(&opacity), sizeof(float));
            
            // Scale (log, NOT exp)
            float scale_x = g.scale.x;
            float scale_y = g.scale.y;
            float scale_z = g.scale.z;
            file.write(reinterpret_cast<const char*>(&scale_x), sizeof(float));
            file.write(reinterpret_cast<const char*>(&scale_y), sizeof(float));
            file.write(reinterpret_cast<const char*>(&scale_z), sizeof(float));
            
            // Rotation: internal (.x=w, .y=x, .z=y, .w=z) -> PLY (rot_0=w, rot_1=x, rot_2=y, rot_3=z)
            float rot_w = g.rotation.x; 
            float rot_x = g.rotation.y; 
            float rot_y = g.rotation.z;
            float rot_z = g.rotation.w;
            file.write(reinterpret_cast<const char*>(&rot_w), sizeof(float));
            file.write(reinterpret_cast<const char*>(&rot_x), sizeof(float));
            file.write(reinterpret_cast<const char*>(&rot_y), sizeof(float));
            file.write(reinterpret_cast<const char*>(&rot_z), sizeof(float));
        }
        
        file.close();
        
        std::cout << "Exported " << validCount << " Gaussians to " << filename << std::endl;
        
        // Debug show first exported Gaussian
        if (count > 0) {
            const Gaussian& g = gaussians[0];
            std::cout << "First exported Gaussian:" << std::endl;
            std::cout << "  Position: (" << g.position.x << ", " << g.position.y << ", " << g.position.z << ")" << std::endl;
            std::cout << "  Scale (log): (" << g.scale.x << ", " << g.scale.y << ", " << g.scale.z << ")" << std::endl;
            std::cout << "  Rotation: w=" << g.rotation.x << " x=" << g.rotation.y << " y=" << g.rotation.z << " z=" << g.rotation.w << std::endl;
            std::cout << "  Opacity (raw): " << g.opacity << std::endl;
        }
        
        return true;
    }
};

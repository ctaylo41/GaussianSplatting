//
//  ply_exporter.hpp
//  GaussianSplatting
//
//  Export trained Gaussians to PLY format compatible with standard 3DGS viewers
//

#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include "ply_loader.hpp"

class PLYExporter {
public:
    
    // Export Gaussians to PLY file
    // Format matches the original 3DGS output for compatibility with viewers
    static bool exportPLY(const std::string& filename,
                          const Gaussian* gaussians,
                          size_t count) {
        
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return false;
        }
        
        // Write PLY header
        file << "ply\n";
        file << "format binary_little_endian 1.0\n";
        file << "element vertex " << count << "\n";
        
        // Position
        file << "property float x\n";
        file << "property float y\n";
        file << "property float z\n";
        
        // Normals (unused but expected by some viewers)
        file << "property float nx\n";
        file << "property float ny\n";
        file << "property float nz\n";
        
        // SH coefficients - DC terms (degree 0)
        file << "property float f_dc_0\n";
        file << "property float f_dc_1\n";
        file << "property float f_dc_2\n";
        
        // SH coefficients - higher order (degree 1)
        for (int i = 0; i < 9; i++) {
            file << "property float f_rest_" << i << "\n";
        }
        
        // Opacity (raw, before sigmoid)
        file << "property float opacity\n";
        
        // Scale (log scale)
        file << "property float scale_0\n";
        file << "property float scale_1\n";
        file << "property float scale_2\n";
        
        // Rotation (quaternion wxyz)
        file << "property float rot_0\n";
        file << "property float rot_1\n";
        file << "property float rot_2\n";
        file << "property float rot_3\n";
        
        file << "end_header\n";
        
        // Write binary data
        for (size_t i = 0; i < count; i++) {
            const Gaussian& g = gaussians[i];
            
            // Copy simd types to plain floats to avoid address-of issues
            float pos_x = g.position.x;
            float pos_y = g.position.y;
            float pos_z = g.position.z;
            
            float scale_x = g.scale.x;
            float scale_y = g.scale.y;
            float scale_z = g.scale.z;
            
            float rot_w = g.rotation.x;  // w stored in x
            float rot_x = g.rotation.y;  // x stored in y
            float rot_y = g.rotation.z;  // y stored in z
            float rot_z = g.rotation.w;  // z stored in w
            
            float opacity = g.opacity;
            
            float zero = 0.0f;
            
            // Position
            file.write(reinterpret_cast<const char*>(&pos_x), sizeof(float));
            file.write(reinterpret_cast<const char*>(&pos_y), sizeof(float));
            file.write(reinterpret_cast<const char*>(&pos_z), sizeof(float));
            
            // Normals (zeros)
            file.write(reinterpret_cast<const char*>(&zero), sizeof(float));
            file.write(reinterpret_cast<const char*>(&zero), sizeof(float));
            file.write(reinterpret_cast<const char*>(&zero), sizeof(float));
            
            // SH DC terms (indices 0, 4, 8 in our layout)
            float sh_dc_0 = g.sh[0];  // R DC
            float sh_dc_1 = g.sh[4];  // G DC
            float sh_dc_2 = g.sh[8];  // B DC
            file.write(reinterpret_cast<const char*>(&sh_dc_0), sizeof(float));
            file.write(reinterpret_cast<const char*>(&sh_dc_1), sizeof(float));
            file.write(reinterpret_cast<const char*>(&sh_dc_2), sizeof(float));
            
            // SH rest (degree 1 coefficients)
            // Our layout: [R0,R1,R2,R3, G0,G1,G2,G3, B0,B1,B2,B3]
            // Standard expects: interleaved by coefficient
            float sh_rest[9] = {
                g.sh[1], g.sh[5], g.sh[9],   // R1, G1, B1
                g.sh[2], g.sh[6], g.sh[10],  // R2, G2, B2
                g.sh[3], g.sh[7], g.sh[11]   // R3, G3, B3
            };
            for (int j = 0; j < 9; j++) {
                file.write(reinterpret_cast<const char*>(&sh_rest[j]), sizeof(float));
            }
            
            // Opacity (raw)
            file.write(reinterpret_cast<const char*>(&opacity), sizeof(float));
            
            // Scale (log)
            file.write(reinterpret_cast<const char*>(&scale_x), sizeof(float));
            file.write(reinterpret_cast<const char*>(&scale_y), sizeof(float));
            file.write(reinterpret_cast<const char*>(&scale_z), sizeof(float));
            
            // Rotation (quaternion)
            file.write(reinterpret_cast<const char*>(&rot_w), sizeof(float));
            file.write(reinterpret_cast<const char*>(&rot_x), sizeof(float));
            file.write(reinterpret_cast<const char*>(&rot_y), sizeof(float));
            file.write(reinterpret_cast<const char*>(&rot_z), sizeof(float));
        }
        
        file.close();
        
        std::cout << "Exported " << count << " Gaussians to " << filename << std::endl;
        
        // Print file size
        std::ifstream checkFile(filename, std::ios::binary | std::ios::ate);
        if (checkFile.is_open()) {
            size_t fileSize = checkFile.tellg();
            std::cout << "File size: " << (fileSize / (1024.0 * 1024.0)) << " MB" << std::endl;
        }
        
        return true;
    }
    
    // Export with automatic filename based on iteration/epoch
    static bool exportCheckpoint(const std::string& baseDir,
                                  const Gaussian* gaussians,
                                  size_t count,
                                  size_t epoch,
                                  float loss) {
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/checkpoint_epoch%03zu_loss%.4f.ply",
                 baseDir.c_str(), epoch, loss);
        return exportPLY(filename, gaussians, count);
    }
};

//
//  camera.hpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-26.
//

#include <simd/simd.h>
#pragma once


class Camera {
public:
    Camera(simd_float3 target, float azimuth, float elevation, float distance_from_target, float fov, float aspect_ratio, float near_plane, float far_plane);
    
    // Getters for view and projection matrices and camera position for shader use
    const simd_float4x4 get_view_matrix() { return view_matrix; };
    const simd_float4x4 get_projection_matrix() {return projection_matrix; };
    const simd_float3 get_position() { return position; };
    
    // Camera manipulation methods
    void orbit(float deltaAzimuth, float deltaElevation);
    void zoom(float deltaDistance);
    void pan(float deltaX, float deltaY);
    void setAspectRatio(float aspect);
    
private:
    // Helper method to update matrices
    void updateMatrices();
    
    // Camera parameters
    simd_float3 position;
    simd_float3 target;
    simd_float3 up;
    
    // Spherical coordinates
    float azimuth;
    float elevation;
    float distance_from_target;

    // Projection parameters
    float fov;
    float aspect_ratio;
    float near_plane;
    float far_plane;
    
    // Matrices
    simd_float4x4 view_matrix;
    simd_float4x4 projection_matrix;
};

//
//  camera.cpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-26.
//

#include "camera.hpp"
#include "AAPLMathUtilities.h"
#include <algorithm>

// Camera Constructor, initializes camera parameters and computes initial matrices
Camera::Camera(simd_float3 target, float azimuth, float elevation, float distance_from_target, float fov, float aspect_ratio, float near_plane, float far_plane)
        :target(target),
        azimuth(azimuth),
        elevation(elevation),
        distance_from_target(distance_from_target),
        fov(fov),aspect_ratio(aspect_ratio),
        near_plane(near_plane),
        far_plane(far_plane)
{
    
    up = simd_make_float3(0.0f,1.0f,0.0f);
    updateMatrices();
}

// Update view and projection matrices based on current camera parameters
void Camera::updateMatrices() {
    float x  = distance_from_target * cos(elevation) * sin(azimuth);
    float y = distance_from_target * sin(elevation);
    float z = distance_from_target * cos(elevation) * cos(azimuth);
    
    position = simd_make_float3(target.x + x, target.y + y, target.z + z);
    
    // Use left-hand coordinate system to match COLMAP convention positive Z
    
    view_matrix = matrix_look_at_left_hand(position.x, position.y, position.z, target.x, target.y, target.z, up.x, up.y, up.z);
    
    projection_matrix = matrix_perspective_left_hand(fov, aspect_ratio, near_plane, far_plane);
}

// Orbit the camera around the target point
void Camera::orbit(float deltaAzimuth, float deltaElevation) {
    azimuth += deltaAzimuth;
    elevation += deltaElevation;
    
    const float maxElevation = M_PI / 2.0f - 0.01f;
    const float minElevation = -M_PI / 2.0f + 0.01f;
    elevation = std::clamp(elevation, minElevation, maxElevation);
    
    updateMatrices();
}

// Zoom the camera in and out by adjusting the distance from the target
void Camera::zoom(float deltaDistance) {
    distance_from_target += deltaDistance;
    
    distance_from_target = std::max(distance_from_target, 0.1f);
    
    updateMatrices();
}

// Pan the camera by moving the target point
void Camera::pan(float deltaX, float deltaY) {
    simd_float3 forward = simd_normalize(target - position);
    simd_float3 right  = simd_normalize(simd_cross(forward, up));
    simd_float3 camUp = simd_cross(right, forward);
    
    float panScale = distance_from_target * 0.002f;
    target = target + right * (-deltaX * panScale) + camUp * (deltaY * panScale);
    
    updateMatrices();
}

// Set a new aspect ratio and update the projection matrix
void Camera::setAspectRatio(float aspect) {
    aspect_ratio = aspect;
    updateMatrices();
}

//
//  camera.cpp
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-26.
//

#include "camera.hpp"
#include "AAPLMathUtilities.h"

Camera::Camera(simd_float3 target, float azimuth, float elevation, float distance_from_target, float fov, float aspect_ratio, float near_plane, float far_plane):target(target),azimuth(azimuth),elevation(elevation),distance_from_target(distance_from_target),fov(fov),aspect_ratio(aspect_ratio),near_plane(near_plane),far_plane(far_plane) {
    float x  = distance_from_target * cos(elevation) * sin(azimuth);
    float y = distance_from_target * sin(elevation);
    float z = distance_from_target * cos(elevation) * cos(azimuth);
    
    position = simd_make_float3(target.x + x, target.y + y, target.z + z);
    
    up = simd_make_float3(0.0f,1.0f,0.0f);
    
    view_matrix = matrix_look_at_right_hand(position.x, position.y, position.z, target.x, target.y, target.z, up.x, up.y, up.z);
    
    projection_matrix = matrix_perspective_right_hand(fov, aspect_ratio, near_plane, far_plane);
}

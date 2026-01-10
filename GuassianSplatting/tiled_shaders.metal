//
//  tiled_shaders.metal
//  GuassianSplatting
//
//  Created by Colin Taylor Taylor on 2025-12-28.
//

#include <metal_stdlib>
using namespace metal;

struct Gaussian {
    packed_float3 position; // offset 0, 12 bytes
    float _pad0;            // offset 12, 4 bytes padding (to align scale to 16)
    packed_float3 scale;    // offset 16, LOG scale, 12 bytes
    float _pad1;            // offset 28, 4 bytes padding (to align rotation to 32)
    float4 rotation;        // offset 32, (w,x,y,z) as (.x=w, .y=x, .z=y, .w=z)
    float opacity;          // offset 48, RAW pre-sigmoid
    float sh[12];           // offset 52, 48 bytes -> ends at 100
    float _pad2;            // offset 100, 4 bytes
    float _pad3;            // offset 104, 4 bytes
    float _pad4;            // offset 108, 4 bytes
};

// Projected Gaussian data for tiled rendering
// Use packed_float3 to match C++ memory layout 12 bytes, not 16
// Total 88 bytes
struct ProjectedGaussian {
    float2 screenPos;       
    packed_float3 conic;    
    float depth;            
    float opacity;          
    packed_float3 color;    
    float radius;           
    uint tileMinX;          
    uint tileMinY;          
    uint tileMaxX;          
    uint tileMaxY;          
    float _pad1;            
    float2 viewPos_xy;      
    packed_float3 cov2D;    
    float _pad2;            
};  

// Tile range structure
struct TileRange {
    uint start;
    uint count;
};

// Uniforms for tiled rendering
struct TiledUniforms {
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4x4 viewProjectionMatrix;
    float2 screenSize;
    float2 focalLength;
    float3 cameraPos;        
    uint numTilesX;
    uint numTilesY;
    uint numGaussians;
    uint _pad2;
};

// Gradients for Gaussians
struct GaussianGradients {
    float position_x;
    float position_y;
    float position_z;
    float opacity;
    float scale_x;
    float scale_y;
    float scale_z;
    float _pad1;
    float4 rotation;
    float sh[12];    
    float viewspace_grad_x;
    float viewspace_grad_y;
    float _pad2;
    float _pad3;
};

// Constants
constant float SH_C0 = 0.28209479177387814f;
constant uint TILE_SIZE = 16;
constant float MAX_RADIUS = 512.0f;
// exp(5) ≈ 148 reasonable max scale
constant float MAX_SCALE = 5.0f;  

// Quaternion to rotation matrix
// q.x=w, q.y=x, q.z=y, q.w=z
float3x3 quatToMat(float4 q) {
    float w = q.x, x = q.y, y = q.z, z = q.w;
    // Metal float3x3 constructor takes columns
    return float3x3(
        float3(1.0 - 2.0*(y*y + z*z), 2.0*(x*y + w*z), 2.0*(x*z - w*y)),
        float3(2.0*(x*y - w*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z + w*x)),
        float3(2.0*(x*z + w*y), 2.0*(y*z - w*x), 1.0 - 2.0*(x*x + y*y))
    );
}

// Project Gaussians to screen space and compute projected parameters
kernel void projectGaussians(
    device const Gaussian* gaussians [[buffer(0)]],
    device ProjectedGaussian* projected [[buffer(1)]],
    constant TiledUniforms& uniforms [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= uniforms.numGaussians) return;
    
    // Fetch Gaussian
    Gaussian g = gaussians[tid];
    ProjectedGaussian proj = {};
    proj.radius = 0;
    proj.tileMinX = UINT_MAX;
    proj.tileMaxX = 0;
    proj.tileMinY = UINT_MAX;
    proj.tileMaxY = 0;
    
    // Skip invalid Gaussians
    if (isnan(g.position.x) || isnan(g.position.y) || isnan(g.position.z) ||
        isnan(g.scale.x) || isnan(g.scale.y) || isnan(g.scale.z) ||
        abs(g.position.x) > 1e6 || abs(g.position.y) > 1e6 || abs(g.position.z) > 1e6) {
        projected[tid] = proj;
        return;
    }

    // Transform to clip space
    float4 worldPos = float4(g.position, 1.0);
    float4 viewPos = uniforms.viewMatrix * worldPos;
    float4 clipPos = uniforms.viewProjectionMatrix * worldPos;
    
    // COLMAP uses opencv convention camera looks down +z axis
    // Objects in front of camera have positive view z
    // clipPos.w = viewZ with our projection, so should be positive for visible objects
    if (clipPos.w <= 0.1 || viewPos.z <= 0.1) {
        projected[tid] = proj;
        return;
    }
    
    // Normalized Device Coordinates
    float3 ndc = clipPos.xyz / clipPos.w;
    
    // Outside frustum
    if (abs(ndc.x) > 1.2 || abs(ndc.y) > 1.2) {
        projected[tid] = proj;
        return;
    }
    
    // Screen position from NDC
    proj.screenPos = float2(
        (ndc.x * 0.5 + 0.5) * uniforms.screenSize.x,
        (ndc.y * 0.5 + 0.5) * uniforms.screenSize.y 
    );

    // Store positive depth
    proj.depth = viewPos.z;  
    proj.viewPos_xy = viewPos.xy;
    
    // scale is stored in LOG space
    float3 logScale = clamp(g.scale, -MAX_SCALE, MAX_SCALE);
    float3 scale = exp(logScale);
    
    // Prevent extremely elongated Gaussians (max 20:1 aspect ratio)
    float maxScale = max(max(scale.x, scale.y), scale.z);
    float minScale = min(min(scale.x, scale.y), scale.z);
    if (maxScale > 20.0f * minScale) {
        // Clamp the max scale to prevent extreme elongation
        float targetMax = 20.0f * minScale;
        scale = scale * (targetMax / maxScale);
    }
    
    // Normalize quaternion
    float4 q = g.rotation;
    float qLen = length(q);
    q = (qLen > 0.001) ? (q / qLen) : float4(1, 0, 0, 0);
    
    // Build 3D covariance using official 3DGS convention:
    // M = S * R, Sigma = M^T * M = R^T * S^2 * R
    // This is mathematically equivalent to R * S^2 * R^T but matches official backward pass
    float3x3 R = quatToMat(q);
    // Diagonal scale matrix
    float3x3 S = float3x3(
        float3(scale.x, 0, 0),
        float3(0, scale.y, 0),
        float3(0, 0, scale.z)
    );
    // Official 3DGS convention: M = S * R
    float3x3 M = S * R;  
    // Sigma = M^T * M = (S * R)^T * (S * R) = R^T * S^2 * R
    float3x3 Sigma3D = transpose(M) * M;  
    
    // View space projection using same approach as official 3DGS
    float z_cam = viewPos.z;
    float fx = uniforms.focalLength.x;
    float fy = uniforms.focalLength.y;
    
    // Clamp to avoid numerical issues at edges
    float limx = 1.3 * fx / z_cam;
    float limy = 1.3 * fy / z_cam;
    float txtz = clamp(viewPos.x / z_cam, -limx, limx);
    float tytz = clamp(viewPos.y / z_cam, -limy, limy);
    
    // Jacobian of projection perspective projection derivative
    // Maps 3D view space to 2D screen space
    float J00 = fx / z_cam;
    float J02 = -fx * txtz / z_cam;
    float J11 = fy / z_cam;
    float J12 = -fy * tytz / z_cam;
    
    // Jacobian matrix
    float3x3 J = float3x3(
        float3(J00, 0, 0),     
        float3(0, J11, 0),  
        float3(J02, J12, 0)
    );
    
    // View matrix rotation world-to-view extract 3x3 rotation
    float3x3 W = float3x3(uniforms.viewMatrix[0].xyz,
                          uniforms.viewMatrix[1].xyz,
                          uniforms.viewMatrix[2].xyz);
    
    // Combined transform T = W * J (official 3DGS convention)
    float3x3 T = W * J;
    
    // Project 3D covariance to 2D cov2D = T^T * Sigma3D * T (official formula)
    float3x3 cov2D_mat = transpose(T) * Sigma3D * T;
    
    // Extract 2D covariance components
    float a = cov2D_mat[0][0];  
    float b = cov2D_mat[1][0];
    float c = cov2D_mat[1][1];
    
    // Low-pass filter add before storing for backward pass
    a += 0.3;
    c += 0.3;
    
    // Store cov2D for backward pass after low-pass filter
    proj.cov2D = float3(a, b, c);
    
    // Compute determinant
    float det = a * c - b * b;
    if (det < 0.0001) {
        projected[tid] = proj;
        return;
    }
    
    // Conic
    float inv_det = 1.0 / det;
    proj.conic = float3(c * inv_det, -b * inv_det, a * inv_det);
    
    // Compute radius from eigenvalues
    float mid = 0.5 * (a + c);
    float disc = mid * mid - det;
    float l1 = mid + sqrt(max(0.1f, disc));
    float rawRadius = 3.0 * sqrt(l1);
    proj.radius = min(ceil(rawRadius), MAX_RADIUS);
    
    // Projected radius zero means skip
    if (proj.radius <= 0) {
        projected[tid] = proj;
        return;
    }
    
    // Tile bounds
    float r = proj.radius;
    int minX = max(0, int(proj.screenPos.x - r));
    int minY = max(0, int(proj.screenPos.y - r));
    int maxX = min(int(uniforms.screenSize.x) - 1, int(proj.screenPos.x + r));
    int maxY = min(int(uniforms.screenSize.y) - 1, int(proj.screenPos.y + r));
    
    // No tile coverage
    if (minX > maxX || minY > maxY) {
        proj.radius = 0;
        projected[tid] = proj;
        return;
    }
    
    // Tile bounds
    proj.tileMinX = uint(minX) / TILE_SIZE;
    proj.tileMinY = uint(minY) / TILE_SIZE;
    proj.tileMaxX = min(uint(maxX) / TILE_SIZE, uniforms.numTilesX - 1);
    proj.tileMaxY = min(uint(maxY) / TILE_SIZE, uniforms.numTilesY - 1);
    
    // Limit tile coverage increased from 64 to allow larger Gaussians
    uint tilesX = proj.tileMaxX - proj.tileMinX + 1;
    uint tilesY = proj.tileMaxY - proj.tileMinY + 1;
    if (tilesX * tilesY > 256) {
        proj.radius = 0;
        projected[tid] = proj;
        return;
    }
    
    // Apply sigmoid to opacity
    float rawOpacity = clamp(g.opacity, -8.0f, 8.0f);
    proj.opacity = 1.0 / (1.0 + exp(-rawOpacity));
    
    // Color from SH DC terms
    proj.color = clamp(float3(
        SH_C0 * g.sh[0] + 0.5f,
        SH_C0 * g.sh[4] + 0.5f,
        SH_C0 * g.sh[8] + 0.5f
    ), 0.0f, 1.0f);
    
    projected[tid] = proj;
}

// Tiled forward rendering kernel
kernel void tiledForward(
    device const Gaussian* gaussians [[buffer(0)]],
    device const ProjectedGaussian* projected [[buffer(1)]],
    device const uint* sortedIndices [[buffer(2)]],
    device const TileRange* tileRanges [[buffer(3)]],
    constant TiledUniforms& uniforms [[buffer(4)]],
    device uint* lastContribIdx [[buffer(5)]],
    texture2d<float, access::write> output [[texture(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= uint(uniforms.screenSize.x) || gid.y >= uint(uniforms.screenSize.y)) return;

    // Determine tile index   
    uint tileX = gid.x / TILE_SIZE;
    uint tileY = gid.y / TILE_SIZE;
    uint tileIdx = tileY * uniforms.numTilesX + tileX;
    TileRange range = tileRanges[tileIdx];
    
    // Use half precision for color accumulation to increase throughput
    half3 color = half3(0);
    half T = 1.0h;
    float2 pixelPos = float2(gid) + 0.5;
    
    uint lastIdx = 0;
    bool hasContrib = false;
    
    // Rely purely on T termination instead of artificial cap
    for (uint i = 0; i < range.count && T > 0.0001h; i++) {
        uint sortIdx = range.start + i;
        uint gIdx = sortedIndices[sortIdx];
        
        if (gIdx >= uniforms.numGaussians) continue;
        
        // Fetch projected Gaussian
        ProjectedGaussian p = projected[gIdx];
        
        // Skip invalid Gaussians
        if (p.radius <= 0) continue;
        
        // Compute offset from Gaussian center
        float2 d = pixelPos - p.screenPos;
        
        // Check if conic is valid
        float conicMag = abs(p.conic.x) + abs(p.conic.y) + abs(p.conic.z);
        if (conicMag < 0.0001) continue;
        
        // Gaussian evaluation in half precision
        half power = half(-0.5 * (p.conic.x * d.x * d.x +
                                   2.0 * p.conic.y * d.x * d.y +
                                   p.conic.z * d.y * d.y));
        
        // Early skip for negligible contribution
        if (power > 0.0h || power < -4.5h) continue;
        
        // Compute Gaussian weight and alpha
        half G = exp(power);
        half alpha = min(half(p.opacity) * G, 0.99h);
        
        // Skip negligible alpha
        if (alpha < half(1.0 / 255.0)) continue;
        
        // Accumulate color using alpha blending
        color += half3(p.color) * alpha * T;
        T *= (1.0h - alpha);
        
        lastIdx = sortIdx;
        hasContrib = true;
    }
    
    // Blend with white background using remaining transmittance
    half3 bgColor = half3(1.0h, 1.0h, 1.0h);
    color = color + bgColor * T;
    
    // Store last contributing index for backward pass
    uint pixelIdx = gid.y * uint(uniforms.screenSize.x) + gid.x;
    lastContribIdx[pixelIdx] = hasContrib ? lastIdx : UINT_MAX;
    
    output.write(float4(float3(color), 1.0), gid);
}

// Tiled backward rendering kernel
kernel void tiledBackward(
    device const Gaussian* gaussians [[buffer(0)]],
    device GaussianGradients* gradients [[buffer(1)]],
    device const ProjectedGaussian* projected [[buffer(2)]],
    device const uint* sortedIndices [[buffer(3)]],
    device const TileRange* tileRanges [[buffer(4)]],
    constant TiledUniforms& uniforms [[buffer(5)]],
    device const uint* lastContribIdx [[buffer(6)]],
    texture2d<float, access::read> rendered [[texture(0)]],
    texture2d<float, access::read> groundTruth [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    // Bounds check
    if (gid.x >= uint(uniforms.screenSize.x) || gid.y >= uint(uniforms.screenSize.y)) return;
    
    // Find last contributing Gaussian index for this pixel
    uint pixelIdx = gid.y * uint(uniforms.screenSize.x) + gid.x;
    uint lastIdx = lastContribIdx[pixelIdx];
    
    if (lastIdx == UINT_MAX) return;
    
    // Determine tile index
    uint tileX = gid.x / TILE_SIZE;
    uint tileY = gid.y / TILE_SIZE;
    uint tileIdx = tileY * uniforms.numTilesX + tileX;
    TileRange range = tileRanges[tileIdx];
    
    // Pixel position
    float2 pixelPos = float2(gid) + 0.5;
    
    float4 r_pix = rendered.read(gid);
    float4 gt_pix = groundTruth.read(gid);
    
    // L1 gradient
    float3 diff = r_pix.rgb - gt_pix.rgb;
    float3 dL_dPixel = sign(diff) / 3.0;
    
    // Traverse back to front for correct gradients
    // Find how many Gaussians contribute and compute T_final
    uint endIdx = min(lastIdx + 1, range.start + range.count);
    
    // Pre-compute T transmittance for the forward pass to get T_final
    float T_final = 1.0;
    // Traverse front to back to compute T_final
    for (uint sortIdx = range.start; sortIdx < endIdx; sortIdx++) {
        // Fetch Gaussian index
        uint gIdx = sortedIndices[sortIdx];
        if (gIdx >= uniforms.numGaussians) continue;
        
        ProjectedGaussian p = projected[gIdx];
        if (p.radius <= 0) continue;
        
        // Compute offset from Gaussian center
        float2 d = pixelPos - p.screenPos;
        float power = -0.5 * (p.conic.x * d.x * d.x +
                              2.0 * p.conic.y * d.x * d.y +
                              p.conic.z * d.y * d.y);
        
        // Early skip for negligible contribution
        if (power > 0.0 || power < -4.5) continue;
        
        // Compute Gaussian weight and alpha
        float G = exp(power);
        float alpha = min(p.opacity * G, 0.99f);
        
        // Skip negligible alpha
        if (alpha < 1.0 / 255.0) continue;
        
        // Update T
        float test_T = T_final * (1.0 - alpha);
        if (test_T < 0.0001) break;
        T_final = test_T;
    }
    
    // Traverse back to front for gradients
    // T starts at T_final and we go backwards by dividing by (1-alpha)
    float T = T_final;
    
    // Initialize accum_rec to the background color
    // White background
    float3 bgColor = float3(1.0);  
     // Accumulated color from Gaussians behind current starts with background
    float3 accum_rec = bgColor; 
    
    // Iterate in reverse order back to front
    for (int sortIdx = int(endIdx) - 1; sortIdx >= int(range.start); sortIdx--) {
        uint gIdx = sortedIndices[uint(sortIdx)];
        
        if (gIdx >= uniforms.numGaussians) continue;
        
        ProjectedGaussian p = projected[gIdx];
        
        if (p.radius <= 0) continue;
        
        // Compute offset from Gaussian center
        float2 d = pixelPos - p.screenPos;
        float power = -0.5 * (p.conic.x * d.x * d.x +
                              2.0 * p.conic.y * d.x * d.y +
                              p.conic.z * d.y * d.y);
        
        // Early skip for negligible contribution
        if (power > 0.0 || power < -4.5) continue;
        
        // Compute Gaussian weight and alpha
        float G = exp(power);
        float alpha = min(p.opacity * G, 0.99f);
        
        if (alpha < 1.0 / 255.0) continue;
        
        // Recover T at this position by undoing the alpha blend
        T = T / max(1.0 - alpha, 0.0001);
        
        float weight = alpha * T;
        
        // Color gradient
        float3 dL_dColor = dL_dPixel * weight;
        // if color is near 0 or 1 zero out gradient to prevent pushing out of bounds
        if (p.color.r <= 0.01 || p.color.r >= 0.99) dL_dColor.r = 0;
        if (p.color.g <= 0.01 || p.color.g >= 0.99) dL_dColor.g = 0;
        if (p.color.b <= 0.01 || p.color.b >= 0.99) dL_dColor.b = 0;
        
        // Accumulate color gradient
        float dL_dAlpha = T * dot(dL_dPixel, p.color - accum_rec);
        
        // Update accum_rec to include current Gaussian for the next Gaussian
        // The contribution from current Gaussian + attenuated contribution from behind
        accum_rec = alpha * p.color + (1.0 - alpha) * accum_rec;
        
        // Opacity gradient
        float sig = p.opacity;
        float dAlpha_dRawOp = sig * (1.0 - sig) * G;
        float dL_dRawOpacity = dL_dAlpha * dAlpha_dRawOp;
        
        // Gaussian gradient
        // For position/scale gradients, we need dL/dG
        float dL_dG = dL_dAlpha * sig;
        
        // Screen position gradient
       
        // Precompute gdx, gdy
        float gdx = G * d.x;
        float gdy = G * d.y;
        
        // Screen position gradient:
        float dG_ddelx = -gdx * p.conic.x - gdy * p.conic.y;
        float dG_ddely = -gdy * p.conic.z - gdx * p.conic.y;
        
        // Negate because the d convention is opposite of official
        float2 dL_dScreenPos = dL_dG * float2(-dG_ddelx, -dG_ddely);
        
        // World position gradient
        // Chain rule dL/dViewPos = dL/dScreenPos * dScreenPos/dViewPos
        float z = p.depth;
        float fx = uniforms.focalLength.x;
        float fy = uniforms.focalLength.y;
        
        // Compute normalized view ratios
        // Same as forward pass Jacobian
        float txtz = p.viewPos_xy.x / z;
        float tytz = p.viewPos_xy.y / z;
        
        // dL/dViewPos
        float3 dL_dViewPos;
        dL_dViewPos.x = dL_dScreenPos.x * fx / z;
        dL_dViewPos.y = dL_dScreenPos.y * fy / z;
        dL_dViewPos.z = -dL_dScreenPos.x * fx * txtz / z
                        -dL_dScreenPos.y * fy * tytz / z;
        
        // Extract view rotation
        // viewMatrix is world-to-view, so transpose gives view-to-world
        float3x3 viewRot = float3x3(
            uniforms.viewMatrix[0].xyz,
            uniforms.viewMatrix[1].xyz,
            uniforms.viewMatrix[2].xyz
        );

        // viewRot already contains columns of rotation, transpose gives view-to-world
        float3 dL_dWorldPos = transpose(viewRot) * dL_dViewPos;
        
        // Conic gradient
        // power = -0.5 * (conic.x * dx^2 + 2 * conic.y * dx*dy + conic.z * dy^2)
        // dL/dconic = dL/dG * dG/dpower * dpower/dconic = dL_dG * G * (-0.5) * (dx^2, 2*dx*dy, dy^2)
        float3 dL_dConic;
        dL_dConic.x = -0.5f * dL_dG * G * d.x * d.x;
        dL_dConic.y = -0.5f * dL_dG * G * 2.0f * d.x * d.y;
        dL_dConic.z = -0.5f * dL_dG * G * d.y * d.y;
        
        // Cov2D gradient
        // conic = inverse(cov2D), dL/dCov2D requires derivative of matrix inverse
        float cov_a = p.cov2D.x;
        float cov_b = p.cov2D.y;
        float cov_c = p.cov2D.z;
        
        // Denominator and its inverse squared
        float denom = cov_a * cov_c - cov_b * cov_b;
        float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);
        
        float3 dL_dCov2D;
        
        // Using formula for derivative of inverse of 2x2 matrix
        dL_dCov2D.x = denom2inv * (-cov_c * cov_c * dL_dConic.x
                                   + 2.0f * cov_b * cov_c * dL_dConic.y
                                   + (denom - cov_a * cov_c) * dL_dConic.z);
        dL_dCov2D.z = denom2inv * (-cov_a * cov_a * dL_dConic.z
                                   + 2.0f * cov_a * cov_b * dL_dConic.y
                                   + (denom - cov_a * cov_c) * dL_dConic.x);
        dL_dCov2D.y = denom2inv * 2.0f * (cov_b * cov_c * dL_dConic.x
                                          - (denom + 2.0f * cov_b * cov_b) * dL_dConic.y
                                          + cov_a * cov_b * dL_dConic.z);
        
        // Cov3D gradient
        // Forward: cov2D = T^T * Sigma3D * T where T = W * J
        // Gradient: for Y = A^T * X * A, dL/dX = A * dL/dY * A^T
        // So: dL/dSigma3D = T * dL/dCov2D * T^T
        float3 t_cam = float3(p.viewPos_xy, p.depth);
        txtz = t_cam.x / t_cam.z;
        tytz = t_cam.y / t_cam.z;
        
        // Jacobian of projection (must match forward pass exactly!)
        float J00 = fx / t_cam.z;
        float J02 = -fx * txtz / t_cam.z;
        float J11 = fy / t_cam.z;
        float J12 = -fy * tytz / t_cam.z;
        
        // Jacobian matrix
        float3x3 J = float3x3(
            float3(J00, 0, 0),
            float3(0, J11, 0),
            float3(J02, J12, 0)
        );
        
        // T = W * J
        float3x3 T_mat = viewRot * J;
        
        // dL/dCov2D as 2x2 matrix embedded in 3x3
        // Extend to 3x3 with zeros for the third row/col
        float3x3 dL_dCov2D_mat = float3x3(
            float3(dL_dCov2D.x, dL_dCov2D.y, 0),
            float3(dL_dCov2D.y, dL_dCov2D.z, 0),
            float3(0, 0, 0)
        );
        
        // For cov2D = T^T * Sigma3D * T, gradient is dL/dSigma3D = T * dL/dCov2D * T^T
        float3x3 dL_dCov3D = T_mat * dL_dCov2D_mat * transpose(T_mat);
        
        // Scale and Rotation gradients
        // Following official 3DGS: M = S * R, Sigma = M^T * M
        // This is equivalent to our R * S convention with Sigma = M * M^T
        Gaussian g_orig = gaussians[gIdx];
        float3 scale = exp(clamp(g_orig.scale, -MAX_SCALE, MAX_SCALE));
        
        // Quaternion components: q = (w, x, y, z) stored as (q.x, q.y, q.z, q.w)
        float4 q = g_orig.rotation;
        float r = q.x;
        float x_q = q.y;
        float y_q = q.z;
        float z_q = q.w;
        
        // Build rotation matrix
        float3x3 R = quatToMat(q);
        
        // Using M = S * R convention from official 3DGS
        // S is diagonal scale matrix
        float3x3 S = float3x3(
            float3(scale.x, 0, 0),
            float3(0, scale.y, 0),
            float3(0, 0, scale.z)
        );
        float3x3 M = S * R; 
        
        // For Sigma = M^T * M dL/dM = 2 * M * dL/dSigma
        float3x3 dL_dM = 2.0f * M * dL_dCov3D;
        
        // Transpose for easier gradient computation
        float3x3 Rt = transpose(R);
        float3x3 dL_dMt = transpose(dL_dM);
        
        // For M = S * R dL/dS diagonal = R^T row i dot dL/dM^T column i
        float3 dL_dScale_val = float3(
            dot(Rt[0], dL_dMt[0]),
            dot(Rt[1], dL_dMt[1]),
            dot(Rt[2], dL_dMt[2])
        );
        
        // Convert to log scale gradient dL/d(log_s) = dL/ds * s
        float3 dL_dLogScale = dL_dScale_val * scale;
        
        // For M = S * R dL/dR = S * dL/dM multiply rows by scale
        // Prepare dL_dMt scaled by s for quaternion gradient
        // dL_dMt[col] *= s[col]
        float3x3 dL_dMt_scaled = float3x3(
            dL_dMt[0] * scale.x,
            dL_dMt[1] * scale.y,
            dL_dMt[2] * scale.z
        );
        
        // Quaternion gradient using 3DGS formulas
        float4 dL_dq;
        dL_dq.x = 2.0f * (z_q * (dL_dMt_scaled[0][1] - dL_dMt_scaled[1][0]) +
                         y_q * (dL_dMt_scaled[2][0] - dL_dMt_scaled[0][2]) +
                         x_q * (dL_dMt_scaled[1][2] - dL_dMt_scaled[2][1]));
        
        dL_dq.y = 2.0f * (y_q * (dL_dMt_scaled[1][0] + dL_dMt_scaled[0][1]) +
                         z_q * (dL_dMt_scaled[2][0] + dL_dMt_scaled[0][2]) +
                         r * (dL_dMt_scaled[1][2] - dL_dMt_scaled[2][1]) -
                         2.0f * x_q * (dL_dMt_scaled[2][2] + dL_dMt_scaled[1][1]));
        
        dL_dq.z = 2.0f * (x_q * (dL_dMt_scaled[1][0] + dL_dMt_scaled[0][1]) +
                         r * (dL_dMt_scaled[2][0] - dL_dMt_scaled[0][2]) +
                         z_q * (dL_dMt_scaled[1][2] + dL_dMt_scaled[2][1]) -
                         2.0f * y_q * (dL_dMt_scaled[2][2] + dL_dMt_scaled[0][0]));
        
        dL_dq.w = 2.0f * (r * (dL_dMt_scaled[0][1] - dL_dMt_scaled[1][0]) +
                         x_q * (dL_dMt_scaled[2][0] + dL_dMt_scaled[0][2]) +
                         y_q * (dL_dMt_scaled[1][2] + dL_dMt_scaled[2][1]) -
                         2.0f * z_q * (dL_dMt_scaled[1][1] + dL_dMt_scaled[0][0]));
        
        // Atomic accumulation
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].sh[0],
            dL_dColor.r * SH_C0, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].sh[4],
            dL_dColor.g * SH_C0, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].sh[8],
            dL_dColor.b * SH_C0, memory_order_relaxed);
        
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].opacity,
            dL_dRawOpacity, memory_order_relaxed);
        
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].position_x,
            dL_dWorldPos.x, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].position_y,
            dL_dWorldPos.y, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].position_z,
            dL_dWorldPos.z, memory_order_relaxed);
        
        // Viewspace screen-space gradients for density control
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].viewspace_grad_x,
            dL_dScreenPos.x, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].viewspace_grad_y,
            dL_dScreenPos.y, memory_order_relaxed);
        
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].scale_x,
            dL_dLogScale.x, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].scale_y,
            dL_dLogScale.y, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&gradients[gIdx].scale_z,
            dL_dLogScale.z, memory_order_relaxed);
        
        atomic_fetch_add_explicit(((device atomic_float*)&gradients[gIdx].rotation) + 0,
            dL_dq.x, memory_order_relaxed);
        atomic_fetch_add_explicit(((device atomic_float*)&gradients[gIdx].rotation) + 1,
            dL_dq.y, memory_order_relaxed);
        atomic_fetch_add_explicit(((device atomic_float*)&gradients[gIdx].rotation) + 2,
            dL_dq.z, memory_order_relaxed);
        atomic_fetch_add_explicit(((device atomic_float*)&gradients[gIdx].rotation) + 3,
            dL_dq.w, memory_order_relaxed);
    }
}

// GPU Pair Generation
// Each thread handles one Gaussian and writes all its tile-pairs atomically
constant float GPU_MIN_OPACITY = 0.005f;
constant uint GPU_MAX_TILES_PER_GAUSSIAN = 256u;

kernel void generateTilePairs(
    device const ProjectedGaussian* projected [[buffer(0)]],
    device ulong* pairKeys [[buffer(1)]],
    device uint* pairValues [[buffer(2)]],
    device atomic_uint* writeCounter [[buffer(3)]],
    constant uint& numGaussians [[buffer(4)]],
    constant uint& numTilesX [[buffer(5)]],
    constant uint& maxPairs [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= numGaussians) return;
    
    ProjectedGaussian p = projected[tid];
    
    // Skip invalid Gaussians
    if (p.radius <= 0) return;
    if (p.tileMinX > p.tileMaxX || p.tileMinY > p.tileMaxY) return;
    if (p.opacity < GPU_MIN_OPACITY) return;
    if (p.tileMinX > 10000 || p.tileMaxX > 10000 || p.tileMinY > 10000 || p.tileMaxY > 10000) return;
    
    // Compute number of tiles covered
    uint tilesX = p.tileMaxX - p.tileMinX + 1;
    uint tilesY = p.tileMaxY - p.tileMinY + 1;
    uint tileCount = tilesX * tilesY;
    
    if (tileCount > GPU_MAX_TILES_PER_GAUSSIAN) return;
    
    // Create depth key for sorting (IEEE float to sortable uint)
    uint depthKey = as_type<uint>(p.depth);
    depthKey = (depthKey & 0x80000000u) ? ~depthKey : (depthKey | 0x80000000u);
    
    // Reserve write positions atomically
    uint writePos = atomic_fetch_add_explicit(writeCounter, tileCount, memory_order_relaxed);
    
    // Check buffer bounds
    if (writePos + tileCount > maxPairs) return;
    
    // Write pairs for all tiles this Gaussian touches
    uint idx = 0;
    for (uint ty = p.tileMinY; ty <= p.tileMaxY; ty++) {
        for (uint tx = p.tileMinX; tx <= p.tileMaxX; tx++) {
            uint tileIdx = ty * numTilesX + tx;
            ulong key = (ulong(tileIdx) << 32) | ulong(depthKey);
            
            pairKeys[writePos + idx] = key;
            pairValues[writePos + idx] = tid;
            idx++;
        }
    }
}

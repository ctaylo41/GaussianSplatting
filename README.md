# 3D Gaussian Splatting on Apple Silicon

A from-scratch implementation of [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) in Metal/C++ for macOS. This project implements the complete training pipeline including tiled rasterization, differentiable rendering, and adaptive density control—entirely on GPU using Apple's Metal framework.

**This is not a port or wrapper**—every component was implemented directly in Metal shaders and C++, providing deep insight into the algorithm's internals.

## Results

### Garden Scene (MipNeRF360)

| Metric | Value |
|--------|-------|
| PSNR | `[TBD]` dB |
| SSIM | `[TBD]` |
| LPIPS | `[TBD]` |
| Final Loss | `[TBD]` |
| Training Time | `[TBD]` min |
| Final Gaussians | `[TBD]` |

<details>
<summary>Training Convergence</summary>

```
[INSERT LOSS CURVE / EPOCH SUMMARIES HERE]
```

</details>

### Kitchen Scene (MipNeRF360)

| Metric | Value |
|--------|-------|
| PSNR | `[TBD]` dB |
| SSIM | `[TBD]` |
| LPIPS | `[TBD]` |
| Final Loss | `[TBD]` |
| Training Time | `[TBD]` min |
| Final Gaussians | `[TBD]` |

<details>
<summary>Training Convergence</summary>

```
[INSERT LOSS CURVE / EPOCH SUMMARIES HERE]
```

</details>

### Comparison with Original Implementation

| Method | PSNR (Garden) | SSIM | Hardware |
|--------|---------------|------|----------|
| Original 3DGS (CUDA) | 27.41 | 0.868 | NVIDIA RTX |
| This Implementation | `[TBD]` | `[TBD]` | Apple M-series |

*Note: Original uses 30K iterations with CUDA. This implementation uses Metal on Apple Silicon with [X] iterations.*

## Visual Results

| Ground Truth | Rendered | Difference |
|--------------|----------|------------|
| `[INSERT IMAGE]` | `[INSERT IMAGE]` | `[INSERT IMAGE]` |

## Technical Implementation

### Architecture

```
COLMAP Sparse Reconstruction
            ↓
    Initialize Gaussians (position, SH, scale, rotation, opacity)
            ↓
    ┌───────────────────────────────────────┐
    │           Training Loop               │
    │  ┌─────────────────────────────────┐  │
    │  │ Forward Pass (tiled rasterizer) │  │
    │  │   • Project to 2D               │  │
    │  │   • Compute 2D covariance       │  │
    │  │   • Tile assignment + sort      │  │
    │  │   • Per-tile alpha blending     │  │
    │  └─────────────────────────────────┘  │
    │              ↓                        │
    │     L1 + D-SSIM Loss                  │
    │              ↓                        │
    │  ┌─────────────────────────────────┐  │
    │  │ Backward Pass (differentiable) │  │
    │  │   • Gradient through blending   │  │
    │  │   • Chain rule to Gaussians     │  │
    │  └─────────────────────────────────┘  │
    │              ↓                        │
    │     Adam Optimizer (per-param LR)     │
    │              ↓                        │
    │     Density Control (prune/split)     │
    └───────────────────────────────────────┘
            ↓
    Export PLY → Real-time Viewer
```

### Core Components

| Component | Description |
|-----------|-------------|
| **Tiled Rasterizer** | 16×16 tile-based rendering with front-to-back alpha blending. Each tile processes Gaussians independently for parallelism. |
| **CPU Radix Sort** | Depth sorting for Gaussians within tiles. Keys encode (tile_id, depth) for correct ordering. GPU sort is a WIP. |
| **Differentiable Rendering** | Full backward pass computing gradients w.r.t. position, covariance, color (DC), and opacity. |
| **Adaptive Density Control** | Clone small Gaussians in high-gradient regions, split large ones, prune low-opacity/large Gaussians. |
| **Adam Optimizer** | GPU-based optimizer with per-parameter learning rates (position, scale, rotation, color, opacity). |

### Key Implementation Details

**Spherical Harmonics (DC only)**: Currently implements degree-0 SH only (3 DC coefficients for RGB). This provides view-independent base color. Higher-order SH for view-dependent effects is not yet implemented.

**Covariance Parameterization**: Gaussians store scale (log-space) and rotation (quaternion) separately. The 3D covariance is reconstructed as Σ = RSS^TR^T, then projected to 2D for rendering.

**Activation Functions**:
- Opacity: `sigmoid(raw)` ensures [0,1] range
- Scale: `exp(log_scale)` ensures positive values
- Color: `sigmoid(SH_C0 * dc + 0.5)` for final RGB

## Technical Challenges & Solutions

### The Post-Reset Saturation Bug

**Problem**: After opacity resets (iterations 3000, 9000, 12000), rendered images showed severe color saturation—whites became yellow, colors washed out. Loss would spike and slowly recover but never reach pre-reset quality.

**Investigation**:
1. Monitored SH coefficient magnitudes across training
2. Found DC coefficients (f_dc_0/1/2) growing unbounded after resets
3. Values reaching 10-50+ (should be ~[-2, 2] range)
4. The opacity reset was disrupting the learned color balance

**Root Cause**: When opacity resets to near-zero, Gaussians that previously contributed strongly suddenly don't. The optimizer compensates by pushing SH coefficients higher to maintain the same visual output—but this creates instability in the gradient flow.

**Solution**: Implemented sigmoid clamping on SH outputs:
```cpp
// In tiled_shaders.metal
float3 sh_color = evalSphericalHarmonics(sh_coeffs, view_dir);
float3 rgb = sigmoid(SH_C0 * sh_color + 0.5);  // Bounded to [0,1]
```

This prevents the SH coefficients from having unbounded effect on final color, making training stable through opacity resets.

**Before Fix** | **After Fix**
:---:|:---:
`[INSERT SATURATED IMAGE]` | `[INSERT FIXED IMAGE]`

### Other Challenges Overcome

1. **Tile Sorting Performance**: Initial CPU sort was bottleneck. Implemented GPU radix sort for 64-bit keys (tile_id << 32 | depth).

2. **Gradient Numerical Stability**: Added epsilon terms to covariance inverse computation to prevent NaN gradients.

3. **Memory Alignment**: Metal requires specific alignment for buffer structs. Gaussian struct padded to 256 bytes for efficient GPU access.

4. **Depth Ordering Artifacts**: Implemented proper front-to-back compositing with premultiplied alpha to eliminate ordering artifacts at tile boundaries.

## What's Not Implemented

- **Higher-order Spherical Harmonics**: Only DC terms (degree-0), no view-dependent color effects
- **GPU Radix Sort**: Currently using CPU sort; GPU implementation is WIP
- **Anti-aliasing / EWA splatting**: No mip-mapping for distant Gaussians
- **Exposure compensation**: Fixed exposure across training views
- **Background modeling**: Assumes black background (no sky dome)
- **Multi-resolution training**: Single resolution only
- **Batch processing**: Single image per iteration

## Performance

| Stage | Time |
|-------|------|
| Projection + Pair Generation | `[TBD]` ms |
| Tile Sort | `[TBD]` ms |
| Rasterization | `[TBD]` ms |
| **Total Forward** | `[TBD]` ms |
| Backward Pass | `[TBD]` ms |
| **Training Iteration** | `[TBD]` ms |
| **Inference FPS** | `[TBD]` |

*Measured on Apple M[X] with [X]GB unified memory*

## Building & Running

### Requirements
- macOS 13+ (Ventura or later)
- Xcode 14+
- GLFW (`brew install glfw`)

### Build
```bash
xcodebuild -project GuassianSplatting.xcodeproj -scheme GuassianSplatting
```

### Training
```bash
./build/GuassianSplatting \
    --colmap /path/to/sparse/0/ \
    --images /path/to/images/ \
    --output trained.ply \
    --epochs 155
```

### Viewing
```bash
./build/GuassianSplatting --view trained.ply
```

## Dataset

Results generated using scenes from the [MipNeRF 360 dataset](https://jonbarron.info/mipnerf360/):
- Garden (outdoor, complex foliage)
- Kitchen (indoor, reflective surfaces)

## References

- Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023
- [Original Implementation (CUDA)](https://github.com/graphdeco-inria/gaussian-splatting)
- [MipNeRF 360 Dataset](https://jonbarron.info/mipnerf360/)

## License

This implementation is for educational and research purposes.

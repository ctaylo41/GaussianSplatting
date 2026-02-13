# 3D Gaussian Splatting Evaluation Results

## Scene: kitchen

### Rendering Quality

| Metric | Mean | Best | Std |
|--------|------|------|-----|
| PSNR (dB) ↑ | 19.72 | 27.34 | 5.98 |
| SSIM ↑ | 0.4881 | 0.8257 | 0.2085 |
| LPIPS ↓ | 0.5676 | 0.2829 | - |

### Training Convergence

| Metric | Value |
|--------|-------|
| Gaussian Count | 241,367 → 818,463 |
| Loss Reduction | 0.1579 → 0.0741 (53.1%) |
| Training Time | 141.9 minutes |
| Opacity Resets | 4 |

### Performance

| Metric | Value |
|--------|-------|
| Training Speed | 283.0 ms/iteration |
| Inference FPS | 0.0 |
| Sort Bottleneck | 0.0% of frame time |

### Comparison with Original Paper (MipNeRF360 Bicycle)

| Method | PSNR | SSIM | Training Time |
|--------|------|------|---------------|
| Original 3DGS | 25.25 | 0.771 | 6 min |
| This Implementation | 19.72 | 0.488 | 142 min |

*Note: Original results use 30K iterations. This implementation uses Metal on Apple Silicon.*

# 3D Gaussian Splatting Evaluation Results

## Scene: kitchen

### Rendering Quality

| Metric | Mean | Best | Std |
|--------|------|------|-----|
| PSNR (dB) ↑ | 29.50 | 33.06 | 2.67 |
| SSIM ↑ | 0.9096 | 0.9495 | 0.0933 |

### Training Convergence

| Metric | Value |
|--------|-------|
| Gaussian Count | 241,367 → 1,029,781 |
| Loss Reduction | 0.1051 → 0.0246 (76.6%) |
| Training Time | 138.0 minutes |
| Opacity Resets | 4 |

### Performance

| Metric | Value |
|--------|-------|
| Training Speed | 275.1 ms/iteration |
| Inference FPS | 19.8 |
| Sort Bottleneck | 57.5% of frame time |

### Comparison with Original Paper (MipNeRF360 Bicycle)

| Method | PSNR | SSIM | Training Time |
|--------|------|------|---------------|
| Original 3DGS | 25.25 | 0.771 | 6 min |
| This Implementation | 29.50 | 0.910 | 138 min |

*Note: Original results use 30K iterations. This implementation uses Metal on Apple Silicon.*

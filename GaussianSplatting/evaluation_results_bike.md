# 3D Gaussian Splatting Evaluation Results

## Scene: bicycle

### Rendering Quality

| Metric | Mean | Best | Std |
|--------|------|------|-----|
| PSNR (dB) ↑ | 21.83 | 26.20 | 2.49 |
| SSIM ↑ | 0.6362 | 0.7925 | 0.0570 |

### Training Convergence

| Metric | Value |
|--------|-------|
| Gaussian Count | 54,275 → 1,888,200 |
| Loss Reduction | 0.1852 → 0.0767 (58.6%) |
| Training Time | 168.1 minutes |
| Opacity Resets | 4 |

### Performance

| Metric | Value |
|--------|-------|
| Training Speed | 336.1 ms/iteration |
| Inference FPS | 24.7 |
| Sort Bottleneck | 52.9% of frame time |

### Comparison with Original Paper (MipNeRF360 Bicycle)

| Method | PSNR | SSIM | Training Time |
|--------|------|------|---------------|
| Original 3DGS | 25.25 | 0.771 | 6 min |
| This Implementation | 21.83 | 0.636 | 168 min |

*Note: Original results use 30K iterations. This implementation uses Metal on Apple Silicon.*

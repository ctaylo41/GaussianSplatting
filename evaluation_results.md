# 3D Gaussian Splatting Evaluation Results

## Scene: bike

### Rendering Quality

| Metric | Mean | Best | Std |
|--------|------|------|-----|
| PSNR (dB) ↑ | 14.15 | 21.34 | 5.04 |
| SSIM ↑ | 0.2801 | 0.5373 | 0.1046 |
| LPIPS ↓ | 0.8652 | 0.5789 | - |

### Training Convergence

| Metric | Value |
|--------|-------|
| Gaussian Count | 54,275 → 1,000,000 |
| Loss Reduction | 0.2628 → 0.1276 (51.4%) |
| Training Time | 509.6 minutes |
| Opacity Resets | 4 |

### Performance

| Metric | Value |
|--------|-------|
| Training Speed | 1019.1 ms/iteration |
| Inference FPS | 0.0 |
| Sort Bottleneck | 0.0% of frame time |

### Comparison with Original Paper (MipNeRF360 Bicycle)

| Method | PSNR | SSIM | Training Time |
|--------|------|------|---------------|
| Original 3DGS | 25.25 | 0.771 | 6 min |
| This Implementation | 14.15 | 0.280 | 510 min |

*Note: Original results use 30K iterations. This implementation uses Metal on Apple Silicon.*

# 3D Gaussian Splatting Evaluation Results

## Scene: bike

### Rendering Quality

| Metric | Mean | Best | Std |
|--------|------|------|-----|
| PSNR (dB) ↑ | 14.96 | 22.92 | 5.97 |
| SSIM ↑ | 0.3236 | 0.7000 | 0.1813 |
| LPIPS ↓ | 0.7530 | 0.3946 | - |

### Training Convergence

| Metric | Value |
|--------|-------|
| Gaussian Count | 54,275 → 2,983,570 |
| Loss Reduction | 0.2430 → 0.0984 (59.5%) |
| Training Time | 173.9 minutes |
| Opacity Resets | 4 |

### Performance

| Metric | Value |
|--------|-------|
| Training Speed | 347.7 ms/iteration |
| Inference FPS | 0.0 |
| Sort Bottleneck | 0.0% of frame time |

### Comparison with Original Paper (MipNeRF360 Bicycle)

| Method | PSNR | SSIM | Training Time |
|--------|------|------|---------------|
| Original 3DGS | 25.25 | 0.771 | 6 min |
| This Implementation | 14.96 | 0.324 | 174 min |

*Note: Original results use 30K iterations. This implementation uses Metal on Apple Silicon.*

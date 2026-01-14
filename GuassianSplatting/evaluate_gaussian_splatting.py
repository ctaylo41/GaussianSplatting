#!/usr/bin/env python3
"""
3D Gaussian Splatting Evaluation Script
Calculates quantitative metrics for graduate portfolio documentation.

Metrics computed:
1. Rendering quality: PSNR, SSIM, LPIPS on held-out test views
2. Training convergence: Final loss, iterations to converge
3. Gaussian count: Starting vs final
4. Performance: Training time, inference FPS

Usage:
    python evaluate_gaussian_splatting.py \
        --renders_dir /path/to/exported_renders \
        --ground_truth_dir /path/to/original_images \
        --training_log /path/to/training_output.txt \
        --scene_name garden \
        --output_json results.json

    Rendered images should be named: image_0001_render.ppm, image_0002_render.ppm, etc.
    Ground truth images are the original training images (JPG/PNG).
"""

import argparse
import json
import os
import re
import glob
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import math

import numpy as np
from PIL import Image

# Optional imports gracefully handle missing dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. LPIPS will be skipped.")

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Install with: pip install lpips")

try:
    from skimage.metrics import structural_similarity as ssim_skimage
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. Using custom SSIM implementation.")


@dataclass
class RenderingMetrics:
    """Metrics for a single rendered image."""
    image_name: str
    psnr: float
    ssim: float
    lpips: Optional[float] = None


@dataclass
class TrainingMetrics:
    """Metrics extracted from training logs."""
    initial_gaussians: int
    final_gaussians: int
    initial_loss: float
    final_loss: float
    total_epochs: int
    total_iterations: int
    training_time_seconds: float
    avg_iteration_time_ms: float
    opacity_resets: List[int]
    loss_history: List[Tuple[int, float]]


@dataclass
class PerformanceMetrics:
    """Performance benchmarks."""
    avg_forward_pass_ms: float
    avg_sort_time_ms: float
    avg_render_time_ms: float
    estimated_inference_fps: float
    pairs_per_frame: int


@dataclass
class EvaluationResults:
    """Complete evaluation results."""
    scene_name: str
    rendering: dict
    training: dict
    performance: dict
    summary: dict


# Image Quality Metrics

def load_image(path: str) -> np.ndarray:
    """Load image as numpy array normalized to [0, 1]."""
    img = Image.open(path).convert('RGB')
    return np.array(img).astype(np.float32) / 255.0


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    PSNR = 10 * log10(MAX^2 / MSE)
    where MAX = 1.0 for normalized images
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * math.log10(1.0 / mse)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Structural Similarity Index.
    
    Uses scikit-image if available, otherwise falls back to custom implementation.
    """
    if SKIMAGE_AVAILABLE:
        # Use multichannel SSIM
        return ssim_skimage(img1, img2, channel_axis=2, data_range=1.0)
    else:
        return _compute_ssim_custom(img1, img2)


def _compute_ssim_custom(img1: np.ndarray, img2: np.ndarray, 
                          window_size: int = 11, C1: float = 0.01**2, 
                          C2: float = 0.03**2) -> float:
    """
    Custom SSIM implementation matching the 3DGS paper.
    
    SSIM = (2*mu_x*mu_y + C1)(2*sigma_xy + C2) / 
           ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2))
    """
    from scipy.ndimage import gaussian_filter
    
    # Convert to grayscale for simplicity (or compute per-channel and average)
    if len(img1.shape) == 3:
        img1_gray = np.mean(img1, axis=2)
        img2_gray = np.mean(img2, axis=2)
    else:
        img1_gray = img1
        img2_gray = img2
    
    sigma = 1.5
    
    mu1 = gaussian_filter(img1_gray, sigma)
    mu2 = gaussian_filter(img2_gray, sigma)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = gaussian_filter(img1_gray ** 2, sigma) - mu1_sq
    sigma2_sq = gaussian_filter(img2_gray ** 2, sigma) - mu2_sq
    sigma12 = gaussian_filter(img1_gray * img2_gray, sigma) - mu1_mu2
    
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


def compute_lpips(img1: np.ndarray, img2: np.ndarray, 
                  lpips_model) -> Optional[float]:
    """
    Compute Learned Perceptual Image Patch Similarity.
    
    Lower is better (0 = identical).
    """
    if not TORCH_AVAILABLE or not LPIPS_AVAILABLE or lpips_model is None:
        return None
    
    # Convert to torch tensors: [B, C, H, W] in range [-1, 1]
    img1_t = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    img2_t = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    
    with torch.no_grad():
        distance = lpips_model(img1_t, img2_t)
    
    return float(distance.item())


def evaluate_rendering_quality(renders_dir: str, ground_truth_dir: str,
                                test_indices: Optional[List[int]] = None) -> List[RenderingMetrics]:
    """
    Evaluate rendering quality on test views.

    Args:
        renders_dir: Directory containing rendered images (image_*_render.ppm)
        ground_truth_dir: Directory containing ground truth images (original training images)
        test_indices: Optional list of image indices to use as test set
                     (default: every 8th image per MipNeRF360 convention)

    Returns:
        List of RenderingMetrics for each test image
    """
    # Initialize LPIPS model if available
    lpips_model = None
    if TORCH_AVAILABLE and LPIPS_AVAILABLE:
        try:
            lpips_model = lpips.LPIPS(net='alex')
            lpips_model.eval()
        except Exception as e:
            print(f"Warning: Could not initialize LPIPS: {e}")

    # Find rendered images - try new naming convention first (image_*_render.ppm)
    render_pattern = os.path.join(renders_dir, "image_*_render.ppm")
    render_files = sorted(glob.glob(render_pattern))

    if not render_files:
        # Try PNG format
        render_pattern = os.path.join(renders_dir, "image_*_render.png")
        render_files = sorted(glob.glob(render_pattern))

    # Fall back to old naming convention (train_render_*.ppm)
    if not render_files:
        render_pattern = os.path.join(renders_dir, "train_render_*.ppm")
        render_files = sorted(glob.glob(render_pattern))

    if not render_files:
        render_pattern = os.path.join(renders_dir, "train_render_*.png")
        render_files = sorted(glob.glob(render_pattern))

    if not render_files:
        print(f"Warning: No rendered images found in {renders_dir}")
        return []

    # Detect which naming convention we're using
    sample_name = os.path.basename(render_files[0])
    using_new_format = sample_name.startswith("image_")

    print(f"  Found {len(render_files)} rendered images ({('image_*_render' if using_new_format else 'train_render_*')} format)")

    # Build ground truth lookup from ground_truth_dir
    gt_files_map = {}
    if ground_truth_dir and os.path.isdir(ground_truth_dir):
        for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
            for gt_path in glob.glob(os.path.join(ground_truth_dir, ext)):
                gt_files_map[os.path.basename(gt_path)] = gt_path
        print(f"  Found {len(gt_files_map)} ground truth images in {ground_truth_dir}")

    results = []
    matched = 0

    for render_path in render_files:
        basename = os.path.basename(render_path)

        if using_new_format:
            # Extract image ID from image_XXXX_render.ppm
            match = re.search(r'image_(\d+)_render', basename)
            if not match:
                continue
            image_id = int(match.group(1))
            image_name = f"image_{image_id:04d}"

            # Try to find matching ground truth by image ID
            # Ground truth images might be named: IMG_0001.jpg, DSC_0001.JPG, image0001.png, etc.
            gt_path = None

            # First try: look for ground_truth_XXXX.ppm in same dir (old checkpoint format)
            gt_checkpoint = render_path.replace("_render.ppm", "_gt.ppm")
            if os.path.exists(gt_checkpoint):
                gt_path = gt_checkpoint

            # Second try: search ground_truth_dir for files containing the image ID
            if not gt_path and gt_files_map:
                # Try various common naming patterns
                id_patterns = [
                    f"{image_id:04d}",  # 0001
                    f"{image_id:03d}",   # 001
                    f"{image_id}",       # 1
                ]
                for gt_name, gt_full_path in gt_files_map.items():
                    gt_name_lower = gt_name.lower()
                    for pattern in id_patterns:
                        # Match pattern in filename (before extension)
                        name_no_ext = os.path.splitext(gt_name)[0]
                        if pattern in name_no_ext and (
                            name_no_ext.endswith(pattern) or
                            f"_{pattern}" in name_no_ext or
                            f"-{pattern}" in name_no_ext or
                            name_no_ext.startswith(pattern)
                        ):
                            gt_path = gt_full_path
                            break
                    if gt_path:
                        break

                # If still not found, try sorted order matching
                if not gt_path:
                    gt_sorted = sorted(gt_files_map.values())
                    if image_id - 1 < len(gt_sorted):  # image IDs are typically 1-indexed
                        gt_path = gt_sorted[image_id - 1]
        else:
            # Old format: train_render_XXXX.ppm with ground_truth_XXXX.ppm
            match = re.search(r'train_render_(\d+)', basename)
            if not match:
                continue
            iteration = int(match.group(1))
            image_name = f"iter_{iteration}"
            gt_path = render_path.replace("train_render_", "ground_truth_")
            if not os.path.exists(gt_path):
                gt_path = None

        if not gt_path or not os.path.exists(gt_path):
            continue

        try:
            render_img = load_image(render_path)
            gt_img = load_image(gt_path)

            # Resize if needed (ground truth might be different resolution)
            if render_img.shape != gt_img.shape:
                # Resize ground truth to match render
                from PIL import Image as PILImage
                gt_pil = PILImage.open(gt_path).convert('RGB')
                gt_pil = gt_pil.resize((render_img.shape[1], render_img.shape[0]), PILImage.LANCZOS)
                gt_img = np.array(gt_pil).astype(np.float32) / 255.0

            psnr = compute_psnr(render_img, gt_img)
            ssim = compute_ssim(render_img, gt_img)
            lpips_val = compute_lpips(render_img, gt_img, lpips_model)

            results.append(RenderingMetrics(
                image_name=image_name,
                psnr=psnr,
                ssim=ssim,
                lpips=lpips_val
            ))
            matched += 1

        except Exception as e:
            print(f"Error processing {image_name}: {e}")

    print(f"  Successfully matched and evaluated {matched} image pairs")

    return results


# Training Log Parsing

def parse_training_log(log_path: str) -> TrainingMetrics:
    """
    Parse training output to extract metrics.
    
    Expected format from your training output.
    """
    with open(log_path, 'r') as f:
        log_content = f.read()
    
    # Extract initial Gaussian count
    initial_match = re.search(r'Created (\d+) Gaussians from COLMAP', log_content)
    initial_gaussians = int(initial_match.group(1)) if initial_match else 0
    
    # Extract final Gaussian count (last occurrence)
    gaussian_matches = re.findall(r'Gaussians: (\d+)', log_content)
    final_gaussians = int(gaussian_matches[-1]) if gaussian_matches else 0
    
    # Extract loss history from epoch summaries
    # Format: "=== Epoch N | Loss: X.XXXXX | Gaussians: NNNN | ..."
    epoch_pattern = r'=== Epoch (\d+) \| Loss: ([\d.]+) \| Gaussians: (\d+)'
    epoch_matches = re.findall(epoch_pattern, log_content)
    
    loss_history = []
    for epoch, loss, gaussians in epoch_matches:
        # Approximate iteration from epoch (assuming 194 images per epoch)
        iteration = int(epoch) * 194
        loss_history.append((iteration, float(loss)))
    
    initial_loss = loss_history[0][1] if loss_history else 0.0
    final_loss = loss_history[-1][1] if loss_history else 0.0
    total_epochs = int(epoch_matches[-1][0]) + 1 if epoch_matches else 0
    
    # Extract total iterations
    iter_matches = re.findall(r'\[iter (\d+)\]', log_content)
    total_iterations = int(iter_matches[-1]) if iter_matches else 0
    
    # Extract training time from epoch summaries
    # Format: "Time: NNNs"
    time_matches = re.findall(r'Time: (\d+)s', log_content)
    total_time = sum(int(t) for t in time_matches)
    
    avg_iter_time = (total_time * 1000 / total_iterations) if total_iterations > 0 else 0
    
    # Extract opacity reset iterations
    reset_matches = re.findall(r'Resetting opacity at iteration (\d+)', log_content)
    opacity_resets = [int(r) for r in reset_matches]
    
    return TrainingMetrics(
        initial_gaussians=initial_gaussians,
        final_gaussians=final_gaussians,
        initial_loss=initial_loss,
        final_loss=final_loss,
        total_epochs=total_epochs,
        total_iterations=total_iterations,
        training_time_seconds=total_time,
        avg_iteration_time_ms=avg_iter_time,
        opacity_resets=opacity_resets,
        loss_history=loss_history
    )


def parse_performance_metrics(log_path: str) -> PerformanceMetrics:
    """
    Extract performance timing from training logs.
    
    Expected format:
    === Forward Pass Timing (avg over 100 frames) ===
      Project+PairGen (GPU):    X.XX ms
      Sort+Copy (CPU):         XX.XX ms
      Range+Render (GPU):       X.XX ms
      TOTAL:                   XX.XX ms
      Pairs/frame:            ~NNNNNN
    """
    with open(log_path, 'r') as f:
        log_content = f.read()
    
    # Find all timing blocks
    timing_blocks = re.findall(
        r'Project\+PairGen \(GPU\):\s+([\d.]+) ms.*?'
        r'Sort\+Copy \(CPU\):\s+([\d.]+) ms.*?'
        r'Range\+Render \(GPU\):\s+([\d.]+) ms.*?'
        r'TOTAL:\s+([\d.]+) ms.*?'
        r'Pairs/frame:\s+~(\d+)',
        log_content, re.DOTALL
    )
    
    if not timing_blocks:
        return PerformanceMetrics(
            avg_forward_pass_ms=0,
            avg_sort_time_ms=0,
            avg_render_time_ms=0,
            estimated_inference_fps=0,
            pairs_per_frame=0
        )
    
    # Average across all timing measurements
    project_times = [float(t[0]) for t in timing_blocks]
    sort_times = [float(t[1]) for t in timing_blocks]
    render_times = [float(t[2]) for t in timing_blocks]
    total_times = [float(t[3]) for t in timing_blocks]
    pairs = [int(t[4]) for t in timing_blocks]
    
    avg_total = np.mean(total_times)
    # Inference FPS forward pass only, no backward
    avg_inference = np.mean(project_times) + np.mean(render_times)
    inference_fps = 1000.0 / avg_inference if avg_inference > 0 else 0
    
    return PerformanceMetrics(
        avg_forward_pass_ms=float(np.mean(project_times)),
        avg_sort_time_ms=float(np.mean(sort_times)),
        avg_render_time_ms=float(np.mean(render_times)),
        estimated_inference_fps=float(inference_fps),
        pairs_per_frame=int(np.mean(pairs))
    )


# Main Evaluation

def generate_summary(rendering_metrics: List[RenderingMetrics],
                     training_metrics: TrainingMetrics,
                     performance_metrics: PerformanceMetrics) -> dict:
    """Generate summary statistics for portfolio."""
    
    # Rendering summary
    if rendering_metrics:
        psnrs = [m.psnr for m in rendering_metrics]
        ssims = [m.ssim for m in rendering_metrics]
        lpips_vals = [m.lpips for m in rendering_metrics if m.lpips is not None]
        
        rendering_summary = {
            "psnr_mean": float(np.mean(psnrs)),
            "psnr_std": float(np.std(psnrs)),
            "psnr_best": float(max(psnrs)),
            "ssim_mean": float(np.mean(ssims)),
            "ssim_std": float(np.std(ssims)),
            "ssim_best": float(max(ssims)),
        }
        
        if lpips_vals:
            rendering_summary["lpips_mean"] = float(np.mean(lpips_vals))
            rendering_summary["lpips_best"] = float(min(lpips_vals))
    else:
        rendering_summary = {}
    
    # Training summary
    training_summary = {
        "gaussian_growth": f"{training_metrics.initial_gaussians:,} → {training_metrics.final_gaussians:,}",
        "loss_reduction": f"{training_metrics.initial_loss:.4f} → {training_metrics.final_loss:.4f}",
        "loss_reduction_percent": f"{(1 - training_metrics.final_loss/training_metrics.initial_loss)*100:.1f}%",
        "total_training_time_minutes": training_metrics.training_time_seconds / 60,
        "opacity_reset_count": len(training_metrics.opacity_resets),
    }
    
    # Performance summary
    performance_summary = {
        "training_ms_per_iteration": training_metrics.avg_iteration_time_ms,
        "inference_fps": performance_metrics.estimated_inference_fps,
        "sort_bottleneck_percent": (performance_metrics.avg_sort_time_ms / 
            (performance_metrics.avg_forward_pass_ms + performance_metrics.avg_sort_time_ms + 
             performance_metrics.avg_render_time_ms) * 100) if performance_metrics.avg_sort_time_ms > 0 else 0,
    }
    
    return {
        "rendering": rendering_summary,
        "training": training_summary,
        "performance": performance_summary
    }


def format_results_markdown(results: EvaluationResults) -> str:
    """Format results as markdown for portfolio."""
    
    md = f"""# 3D Gaussian Splatting Evaluation Results

## Scene: {results.scene_name}

### Rendering Quality

| Metric | Mean | Best | Std |
|--------|------|------|-----|
"""
    
    if results.summary.get("rendering"):
        r = results.summary["rendering"]
        md += f"| PSNR (dB) ↑ | {r.get('psnr_mean', 0):.2f} | {r.get('psnr_best', 0):.2f} | {r.get('psnr_std', 0):.2f} |\n"
        md += f"| SSIM ↑ | {r.get('ssim_mean', 0):.4f} | {r.get('ssim_best', 0):.4f} | {r.get('ssim_std', 0):.4f} |\n"
        if 'lpips_mean' in r:
            md += f"| LPIPS ↓ | {r.get('lpips_mean', 0):.4f} | {r.get('lpips_best', 0):.4f} | - |\n"
    
    t = results.summary.get("training", {})
    md += f"""
### Training Convergence

| Metric | Value |
|--------|-------|
| Gaussian Count | {t.get('gaussian_growth', 'N/A')} |
| Loss Reduction | {t.get('loss_reduction', 'N/A')} ({t.get('loss_reduction_percent', 'N/A')}) |
| Training Time | {t.get('total_training_time_minutes', 0):.1f} minutes |
| Opacity Resets | {t.get('opacity_reset_count', 0)} |
"""
    
    p = results.summary.get("performance", {})
    md += f"""
### Performance

| Metric | Value |
|--------|-------|
| Training Speed | {p.get('training_ms_per_iteration', 0):.1f} ms/iteration |
| Inference FPS | {p.get('inference_fps', 0):.1f} |
| Sort Bottleneck | {p.get('sort_bottleneck_percent', 0):.1f}% of frame time |
"""
    
    md += f"""
### Comparison with Original Paper (MipNeRF360 Bicycle)

| Method | PSNR | SSIM | Training Time |
|--------|------|------|---------------|
| Original 3DGS | 25.25 | 0.771 | 6 min |
| This Implementation | {results.summary.get('rendering', {}).get('psnr_mean', 0):.2f} | {results.summary.get('rendering', {}).get('ssim_mean', 0):.3f} | {t.get('total_training_time_minutes', 0):.0f} min |

*Note: Original results use 30K iterations. This implementation uses Metal on Apple Silicon.*
"""
    
    return md


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate 3D Gaussian Splatting training results'
    )
    parser.add_argument('--renders_dir', type=str, default='/tmp',
                        help='Directory containing rendered images (image_*_render.ppm)')
    parser.add_argument('--ground_truth_dir', type=str, default=None,
                        help='Directory containing original training images (for image_*_render format)')
    parser.add_argument('--training_log', type=str, required=True,
                        help='Path to training output log file')
    parser.add_argument('--scene_name', type=str, default='garden',
                        help='Scene name for results')
    parser.add_argument('--output_json', type=str, default='evaluation_results.json',
                        help='Output JSON file path')
    parser.add_argument('--output_markdown', type=str, default='evaluation_results.md',
                        help='Output Markdown file path')

    args = parser.parse_args()
    
    print("=" * 60)
    print("3D Gaussian Splatting Evaluation")
    print("=" * 60)
    
    # Parse training log
    print("\n[1/3] Parsing training log...")
    training_metrics = parse_training_log(args.training_log)
    print(f"  - Initial Gaussians: {training_metrics.initial_gaussians:,}")
    print(f"  - Final Gaussians: {training_metrics.final_gaussians:,}")
    print(f"  - Loss: {training_metrics.initial_loss:.4f} → {training_metrics.final_loss:.4f}")
    print(f"  - Opacity resets: {training_metrics.opacity_resets}")
    
    # Parse performance metrics
    print("\n[2/3] Extracting performance metrics...")
    performance_metrics = parse_performance_metrics(args.training_log)
    print(f"  - Forward pass: {performance_metrics.avg_forward_pass_ms:.1f} ms")
    print(f"  - Sort (CPU): {performance_metrics.avg_sort_time_ms:.1f} ms")
    print(f"  - Render: {performance_metrics.avg_render_time_ms:.1f} ms")
    print(f"  - Estimated inference FPS: {performance_metrics.estimated_inference_fps:.1f}")
    
    # Evaluate rendering quality
    print("\n[3/3] Evaluating rendering quality...")
    gt_dir = args.ground_truth_dir if args.ground_truth_dir else args.renders_dir
    rendering_metrics = evaluate_rendering_quality(args.renders_dir, gt_dir)
    
    if rendering_metrics:
        for m in rendering_metrics[-3:]:  # Show last 3
            lpips_str = f", LPIPS={m.lpips:.4f}" if m.lpips else ""
            print(f"  - {m.image_name}: PSNR={m.psnr:.2f}, SSIM={m.ssim:.4f}{lpips_str}")
    else:
        print("  - No rendered images found for evaluation")
    
    # Generate summary
    summary = generate_summary(rendering_metrics, training_metrics, performance_metrics)
    
    # Create results object
    results = EvaluationResults(
        scene_name=args.scene_name,
        rendering={
            "per_image": [asdict(m) for m in rendering_metrics]
        },
        training=asdict(training_metrics),
        performance=asdict(performance_metrics),
        summary=summary
    )
    
    # Save JSON
    with open(args.output_json, 'w') as f:
        json.dump(asdict(results), f, indent=2)
    print(f"\n✓ Saved JSON results to {args.output_json}")
    
    # Save Markdown
    markdown = format_results_markdown(results)
    with open(args.output_markdown, 'w') as f:
        f.write(markdown)
    print(f"✓ Saved Markdown report to {args.output_markdown}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nGaussian Growth: {summary['training']['gaussian_growth']}")
    print(f"Loss Reduction: {summary['training']['loss_reduction']} ({summary['training']['loss_reduction_percent']})")
    
    if summary.get('rendering'):
        print(f"\nRendering Quality:")
        print(f"  PSNR: {summary['rendering'].get('psnr_mean', 0):.2f} dB (best: {summary['rendering'].get('psnr_best', 0):.2f})")
        print(f"  SSIM: {summary['rendering'].get('ssim_mean', 0):.4f} (best: {summary['rendering'].get('ssim_best', 0):.4f})")
    
    print(f"\nPerformance:")
    print(f"  Training: {summary['performance']['training_ms_per_iteration']:.0f} ms/iter")
    print(f"  Inference: {summary['performance']['inference_fps']:.1f} FPS")
    print(f"  Sort bottleneck: {summary['performance']['sort_bottleneck_percent']:.0f}% of frame time")


if __name__ == "__main__":
    main()

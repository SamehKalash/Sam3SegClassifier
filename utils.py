"""
Utility functions for SAM3 testing
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import torch

def overlay_masks(image, masks):
    """
    Overlay segmentation masks on an image with colored transparency
    
    Args:
        image: PIL Image
        masks: torch.Tensor of shape [N, H, W] where N is number of masks
    
    Returns:
        PIL Image with overlaid masks
    """
    image = image.convert("RGBA")
    masks_np = 255 * masks.cpu().numpy().astype(np.uint8)
    
    n_masks = masks_np.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks_np, colors):
        mask_img = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask_img.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    
    return image


def visualize_results(image, results, title="Segmentation Results"):
    """
    Create a visualization of segmentation results
    
    Args:
        image: PIL Image
        results: Dictionary with 'masks', 'boxes', 'scores'
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Image with masks
    result_image = overlay_masks(image, results['masks'])
    axes[1].imshow(result_image)
    axes[1].set_title(f"{title} ({len(results['masks'])} objects)")
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig


def save_results(image, results, output_path, show_boxes=True):
    """
    Save visualization of results to file
    
    Args:
        image: PIL Image
        results: Dictionary with 'masks', 'boxes', 'scores'
        output_path: Path to save image
        show_boxes: Whether to draw bounding boxes
    """
    fig = plt.figure(figsize=(12, 8))
    
    # Create overlay
    result_image = overlay_masks(image, results['masks'])
    plt.imshow(result_image)
    
    # Draw boxes if requested
    if show_boxes and 'boxes' in results:
        boxes = results['boxes'].cpu().numpy()
        scores = results['scores'].cpu().numpy()
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Draw box
            rect = plt.Rectangle(
                (x1, y1), width, height,
                fill=False, edgecolor='red', linewidth=2
            )
            plt.gca().add_patch(rect)
            
            # Add score label
            plt.text(
                x1, y1 - 5,
                f'{score:.2f}',
                color='red',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7)
            )
    
    plt.title(f"Segmentation Results - {len(results['masks'])} objects detected")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_model_info(model):
    """Print information about the model"""
    print("\n📊 Model Information:")
    print(f"  Model type: {type(model).__name__}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / (1024**3):.2f} GB (FP32)")
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  Dtype: {next(model.parameters()).dtype}")


def benchmark_inference(model, processor, image, num_runs=5):
    """
    Benchmark inference speed
    
    Args:
        model: The SAM3 model
        processor: The processor
        image: PIL Image
        num_runs: Number of runs for averaging
    
    Returns:
        Dictionary with timing statistics
    """
    import time
    
    device = next(model.parameters()).device
    inputs = processor(images=image, text="test", return_tensors="pt").to(device)
    
    # Warmup
    with torch.no_grad():
        _ = model(**inputs)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            _ = model(**inputs)
        times.append(time.time() - start)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'fps': 1.0 / np.mean(times)
    }


def create_comparison_plot(images_with_results, output_path):
    """
    Create a comparison plot of multiple segmentation results
    
    Args:
        images_with_results: List of tuples (image, results, title)
        output_path: Path to save the comparison
    """
    n = len(images_with_results)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    
    if n == 1:
        axes = [axes]
    
    for ax, (image, results, title) in zip(axes, images_with_results):
        result_image = overlay_masks(image, results['masks'])
        ax.imshow(result_image)
        ax.set_title(f"{title}\n({len(results['masks'])} objects)")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def memory_usage():
    """Get current memory usage"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    return {
        'rss_mb': mem_info.rss / (1024 ** 2),
        'vms_mb': mem_info.vms / (1024 ** 2)
    }

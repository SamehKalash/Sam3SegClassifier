"""
SAM3 Tracker Test - Promptable Visual Segmentation (PVS)
Tests point-based and box-based segmentation for interactive use
"""

import torch
from PIL import Image
from transformers import Sam3TrackerProcessor, Sam3TrackerModel
from accelerate import Accelerator
import requests
import matplotlib.pyplot as plt
import numpy as np
from utils import overlay_masks

def visualize_with_prompts(image, masks, points=None, boxes=None, title=""):
    """Visualize segmentation with prompt overlays"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original with prompts
    axes[0].imshow(image)
    
    if points is not None:
        for point_set, label_set in zip(points[0], points[1]):
            for point, label in zip(point_set, label_set):
                color = 'green' if label == 1 else 'red'
                marker = 'o' if label == 1 else 'x'
                axes[0].plot(point[0], point[1], marker=marker, 
                           color=color, markersize=10, markeredgewidth=2)
    
    if boxes is not None:
        for box in boxes[0]:
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                fill=False, edgecolor='blue', linewidth=2)
            axes[0].add_patch(rect)
    
    axes[0].set_title("Input with Prompts")
    axes[0].axis('off')
    
    # Result with masks
    result_image = overlay_masks(image, masks)
    axes[1].imshow(result_image)
    axes[1].set_title(f"{title} ({len(masks)} masks)")
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    print("=" * 70)
    print("SAM3 Tracker Test - Interactive Segmentation")
    print("=" * 70)
    
    # Device setup
    device = Accelerator().device
    print(f"\n✓ Using device: {device}")
    
    # Load model
    print("\n📥 Loading SAM3 Tracker model...")
    model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)
    processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    print("✓ Model loaded successfully!")
    
    # Load test image
    print("\n🖼️  Loading test image...")
    image_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"
    raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    print(f"✓ Image loaded: {raw_image.size}")
    
    # Test 1: Single point click
    print("\n" + "=" * 70)
    print("Test 1: Single Point Click")
    print("=" * 70)
    
    input_points = [[[[500, 375]]]]  # Click on truck
    input_labels = [[[1]]]  # Positive click
    
    print(f"🖱️  Point: {input_points[0][0][0]}")
    
    inputs = processor(
        images=raw_image, 
        input_points=input_points, 
        input_labels=input_labels, 
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"]
    )[0]
    
    print(f"✓ Generated {masks.shape[1]} masks (ranked by quality)")
    
    # Save best mask
    fig = visualize_with_prompts(
        raw_image, 
        masks[:, 0:1], 
        points=(input_points, input_labels),
        title="Single Point"
    )
    plt.savefig("tracker_single_point.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: tracker_single_point.png")
    
    # Test 2: Multiple points for refinement
    print("\n" + "=" * 70)
    print("Test 2: Multiple Points for Refinement")
    print("=" * 70)
    
    input_points = [[[[500, 375], [1125, 625]]]]  # Two points
    input_labels = [[[1, 1]]]  # Both positive
    
    print(f"🖱️  Points: {input_points[0][0]}")
    
    inputs = processor(
        images=raw_image,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"]
    )[0]
    
    print(f"✓ Generated {masks.shape[1]} masks")
    
    fig = visualize_with_prompts(
        raw_image,
        masks[:, 0:1],
        points=(input_points, input_labels),
        title="Multiple Points"
    )
    plt.savefig("tracker_multiple_points.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: tracker_multiple_points.png")
    
    # Test 3: Bounding box input
    print("\n" + "=" * 70)
    print("Test 3: Bounding Box Input")
    print("=" * 70)
    
    input_boxes = [[[75, 275, 1725, 850]]]  # Box around truck
    
    print(f"📦 Box (x1,y1,x2,y2): {input_boxes[0][0]}")
    
    inputs = processor(
        images=raw_image,
        input_boxes=input_boxes,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"]
    )[0]
    
    print(f"✓ Generated {masks.shape[1]} masks")
    
    fig = visualize_with_prompts(
        raw_image,
        masks[:, 0:1],
        boxes=input_boxes,
        title="Bounding Box"
    )
    plt.savefig("tracker_box.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: tracker_box.png")
    
    # Test 4: Multiple objects
    print("\n" + "=" * 70)
    print("Test 4: Segment Multiple Objects")
    print("=" * 70)
    
    # Points for different objects in the scene
    input_points = [[[[500, 375]], [[650, 750]]]]
    input_labels = [[[1], [1]]]
    
    print(f"🖱️  Object 1 point: {input_points[0][0][0]}")
    print(f"🖱️  Object 2 point: {input_points[0][1][0]}")
    
    inputs = processor(
        images=raw_image,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"]
    )[0]
    
    print(f"✓ Generated masks for {masks.shape[0]} objects")
    
    fig = visualize_with_prompts(
        raw_image,
        masks,
        points=(input_points, input_labels),
        title="Multiple Objects"
    )
    plt.savefig("tracker_multiple_objects.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: tracker_multiple_objects.png")
    
    # Test 5: Positive and negative points
    print("\n" + "=" * 70)
    print("Test 5: Positive and Negative Points")
    print("=" * 70)
    
    input_points = [[[[500, 375], [300, 400]]]]
    input_labels = [[[1, 0]]]  # First positive, second negative
    
    print(f"🖱️  Positive point: {input_points[0][0][0]}")
    print(f"🖱️  Negative point: {input_points[0][0][1]}")
    
    inputs = processor(
        images=raw_image,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"]
    )[0]
    
    print(f"✓ Generated {masks.shape[1]} masks (refined by negative point)")
    
    fig = visualize_with_prompts(
        raw_image,
        masks[:, 0:1],
        points=(input_points, input_labels),
        title="Positive + Negative"
    )
    plt.savefig("tracker_pos_neg.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: tracker_pos_neg.png")
    
    print("\n" + "=" * 70)
    print("✅ All tracker tests completed successfully!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - tracker_single_point.png")
    print("  - tracker_multiple_points.png")
    print("  - tracker_box.png")
    print("  - tracker_multiple_objects.png")
    print("  - tracker_pos_neg.png")
    
    print("\n💡 Use Cases:")
    print("  - Interactive segmentation with mouse clicks")
    print("  - Object selection with bounding boxes")
    print("  - Refinement with positive/negative points")
    print("  - Multi-object segmentation in single image")

if __name__ == "__main__":
    main()

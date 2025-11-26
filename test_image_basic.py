"""
Basic SAM3 Image Segmentation Test
Tests text-based prompting for image segmentation on Raspberry Pi 5
"""

import torch
from PIL import Image
from transformers import Sam3Processor, Sam3Model
import requests
import matplotlib.pyplot as plt
import numpy as np
from utils import overlay_masks, save_results

def main():
    print("=" * 60)
    print("SAM3 Image Segmentation Test - Raspberry Pi 5")
    print("=" * 60)
    
    # Device setup (CPU for Pi 5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n✓ Using device: {device}")
    
    # Load model
    print("\n📥 Loading SAM3 model from HuggingFace...")
    print("   (This may take a while on first run)")
    
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    
    print("✓ Model loaded successfully!")
    
    # Load test image
    print("\n🖼️  Loading test image...")
    #image_url = "7fzPt.jpg"
    #image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    image = Image.open("7fzPt.jpg").convert("RGB")
    print(f"✓ Image loaded: {image.size}")
    
    # Test 1: Text-based segmentation
    print("\n" + "=" * 60)
    print("Test 1: Text-based segmentation - 'cat'")
    print("=" * 60)
    
    text_prompt = "door"
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    
    print(f"🔍 Segmenting '{text_prompt}'...")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process results
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    
    print(f"✓ Found {len(results['masks'])} objects")
    print(f"  Scores: {[f'{s:.3f}' for s in results['scores'].tolist()]}")
    
    # Visualize
    result_image = overlay_masks(image, results["masks"])
    result_image.save("output_chair.png")
    print("✓ Saved result to: output_chair.png")
    
    # Test 2: Different prompt
    print("\n" + "=" * 60)
    print("Test 2: Text-based segmentation - 'ceiling'")
    print("=" * 60)
    
    text_prompt = "ceiling"
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    
    print(f"🔍 Segmenting '{text_prompt}'...")
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    
    print(f"✓ Found {len(results['masks'])} objects")
    
    # Visualize
    result_image = overlay_masks(image, results["masks"])
    result_image.save("output_ear.png")
    print("✓ Saved result to: output_ear.png")
    
     # Test 4: Different prompt
    print("\n" + "=" * 60)
    print("Test 4 : Text-based segmentation - 'Window'")
    print("=" * 60)
    
    text_prompt = "window"
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    
    print(f"🔍 Segmenting '{text_prompt}'...")
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    
    print(f"✓ Found {len(results['masks'])} objects")
    
    # Visualize
    result_image = overlay_masks(image, results["masks"])
    result_image.save("output_window.png")
    print("✓ Saved result to: output_window.png")
    
    # Test 3: Using bounding box
    print("\n" + "=" * 60)
    print("Test 3: Bounding box segmentation")
    print("=" * 60)
    
    # Define a box around the cat
    box_xyxy = [50, 50, 400, 400]
    input_boxes = [[box_xyxy]]
    input_boxes_labels = [[1]]  # 1 = positive box
    
    inputs = processor(
        images=image,
        input_boxes=input_boxes,
        input_boxes_labels=input_boxes_labels,
        return_tensors="pt"
    ).to(device)
    
    print(f"🔍 Segmenting with box: {box_xyxy}...")
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    
    print(f"✓ Found {len(results['masks'])} objects in box")
    
    # Visualize
    result_image = overlay_masks(image, results["masks"])
    result_image.save("output_box.png")
    print("✓ Saved result to: output_box.png")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - output_cat.png")
    print("  - output_ear.png")
    print("  - output_box.png")

if __name__ == "__main__":
    main()

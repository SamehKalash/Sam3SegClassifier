"""
SAM3 Segmentation + Text-Prompt Labeling Pipeline
Uses SAM3 text prompts as direct labels for segmented objects
Includes visualization and image saving
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import Sam3Processor, Sam3Model
import os
from pathlib import Path


class Sam3SegmentationPipeline:
    """Pipeline using SAM3 for segmentation with text prompts as labels"""
    
    def __init__(self, device="cuda"):
        """
        Initialize the SAM3 segmentation pipeline
        
        Args:
            device: Device to run inference on
        """
        self.device = device
        
        # Load SAM3
        print("[LOAD] Loading SAM3 model from HuggingFace...")
        self.sam_model = Sam3Model.from_pretrained("facebook/sam3").to(device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        print("[OK] SAM3 Model loaded successfully!")
        
        # Set model to evaluation mode
        self.sam_model.eval()
    
    def process_image_with_prompt(self, image, text_prompt):
        """
        Segment image with SAM3 using text prompt
        The prompt text becomes the class label for all detected objects
        
        Args:
            image: PIL Image or numpy array
            text_prompt: Text prompt describing what to segment (e.g., "wall", "door", "window")
                        This prompt becomes the class label for segmented objects
            
        Returns:
            List of segmented objects with text prompt as label
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        
        print(f"\n🔍 Segmenting with SAM3 prompt: '{text_prompt}'")
        
        # Process with SAM3 using text prompt
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.sam_model(**inputs)
        
        # Post-process to get instance segmentation
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        
        masks = results.get('masks', [])
        scores = results.get('scores', [])
        
        if len(masks) == 0:
            print(f"[WARN] No objects found for prompt: '{text_prompt}'")
            return []
        
        print(f"[OK] Found {len(masks)} objects matching '{text_prompt}'")
        
        # Convert results - use prompt as the class label
        segmented_objects = []
        image_np = np.array(image)
        
        for idx, (mask, score) in enumerate(zip(masks, scores)):
            # Convert mask to numpy if it's a tensor
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = mask
            
            # Get bounding box from mask
            rows = np.any(mask_np, axis=1)
            cols = np.any(mask_np, axis=0)
            
            if not rows.any() or not cols.any():
                continue
            
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # Extract region
            region = image_np[y_min:y_max+1, x_min:x_max+1]
            
            if region.size == 0:
                continue
            
            # ⭐ KEY: Use the text prompt as the class label
            segmented_objects.append({
                "class": text_prompt,              # ← Text prompt IS the class label
                "class_id": idx,
                "confidence": score.item() if isinstance(score, torch.Tensor) else float(score),
                "bbox": [x_min, y_min, x_max, y_max],
                "mask": mask_np,
                "region": Image.fromarray(region).convert("RGB"),
                "region_idx": idx
            })
        
        return segmented_objects
    
    def process_image_with_multiple_prompts(self, image, prompts):
        """
        Segment image with multiple text prompts
        
        Args:
            image: PIL Image or numpy array
            prompts: List of text prompts (e.g., ["wall", "door", "window"])
            
        Returns:
            Dictionary mapping prompts to their segmented objects
        """
        results = {}
        for prompt in prompts:
            results[prompt] = self.process_image_with_prompt(image, prompt)
    
        return results
    
    def visualize_segmentation(self, image, segmented_objects, prompt, output_path="segmentation_result.png"):
        """
        Visualize segmentation results with masks overlaid on the original image
        
        Args:
            image: Original PIL Image
            segmented_objects: List of segmented objects with masks
            prompt: Text prompt used for segmentation
            output_path: Path to save the visualization
            
        Returns:
            PIL Image of the visualization
        """
        # Convert image to numpy array
        image_np = np.array(image)
        
        # Create a copy for visualization
        vis_image = image_np.copy().astype(np.float32)
        
        # Create color map for different objects
        colors = [
            [255, 0, 0],      # Red
            [0, 255, 0],      # Green
            [0, 0, 255],      # Blue
            [255, 255, 0],    # Yellow
            [255, 0, 255],    # Magenta
            [0, 255, 255],    # Cyan
        ]
        
        # Overlay masks with transparency
        for idx, obj in enumerate(segmented_objects):
            mask = obj['mask'].astype(bool)
            color = colors[idx % len(colors)]
            
            # Apply color to masked regions
            vis_image[mask] = 0.6 * vis_image[mask] + 0.4 * np.array(color)
        
        # Convert back to uint8
        vis_image = vis_image.astype(np.uint8)
        vis_pil = Image.fromarray(vis_image)
        
        # Draw bounding boxes
        draw = ImageDraw.Draw(vis_pil)
        for idx, obj in enumerate(segmented_objects):
            bbox = obj['bbox']
            x_min, y_min, x_max, y_max = bbox
            color = tuple(colors[idx % len(colors)])
            
            # Draw rectangle
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
            
            # Draw label
            label = f"{obj['class']} ({obj['confidence']:.2f})"
            draw.text((x_min, y_min - 10), label, fill=color)
        
        # Save the result
        vis_pil.save(output_path)
        print(f"[OK] Segmentation visualization saved to: {output_path}")
        
        return vis_pil
    
    def visualize_multi_prompt_results(self, image, results_dict, output_dir="segmentation_results"):
        """
        Visualize results from multiple prompts
        
        Args:
            image: Original PIL Image
            results_dict: Dictionary mapping prompts to segmented objects
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary mapping prompts to their visualization images
        """
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        vis_images = {}
        for prompt, objects in results_dict.items():
            if objects:  # Only visualize if objects were found
                output_path = os.path.join(output_dir, f"segmentation_{prompt}.png")
                vis_images[prompt] = self.visualize_segmentation(image, objects, prompt, output_path)
            else:
                print(f"[WARN] No objects found for '{prompt}', skipping visualization")
        
        return vis_images

if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*70)
    print("SAM3 Segmentation Pipeline - Text Prompts as Labels")
    print("="*70)
    print(f"🖥️  Device: {device}")
    
    # Initialize SAM3 pipeline
    pipeline = Sam3SegmentationPipeline(device=device)
    
    # Load image
    image_path = "7fzPt.jpg"
    image = Image.open(image_path)
    print(f"\n📷 Loaded image: {image_path}")
    
    # EXAMPLE 1: Segment with single text prompt
    print("\n" + "="*70)
    print("EXAMPLE 1: Segment 'wall' - Text prompt becomes the class label")
    print("="*70)
    print("👉 The prompt you provide is used as the segmentation class label")
    
    # ⭐ THIS IS WHERE YOU PUT YOUR TEXT PROMPT ⭐
    # The text prompt becomes the class label for all detected objects
    results_wall = pipeline.process_image_with_prompt(image, text_prompt="wall")
    
    print("\n" + "-"*70)
    print("Results for 'wall':")
    print("-"*70)
    if results_wall:
        for obj in results_wall:
            print(f"  Object {obj['region_idx']}: Class='{obj['class']}' "
                  f"(confidence: {obj['confidence']:.3f})")
        
        # 🎨 VISUALIZE THE SEGMENTATION
        print("\n🎨 Generating visualization...")
        vis_wall = pipeline.visualize_segmentation(image, results_wall, "wall", "segmentation_wall.png")
        print("   Saved as: segmentation_wall.png")
    else:
        print("  No 'wall' objects found in the image.")
    
    # EXAMPLE 2: Segment another object
    print("\n" + "="*70)
    print("EXAMPLE 2: Segment 'door' - Different prompt, different label")
    print("="*70)
    
    # ⭐ CHANGE THE PROMPT HERE ⭐
    results_door = pipeline.process_image_with_prompt(image, text_prompt="door")
    
    print("\n" + "-"*70)
    print("Results for 'door':")
    print("-"*70)
    if results_door:
        for obj in results_door:
            print(f"  Object {obj['region_idx']}: Class='{obj['class']}' "
                  f"(confidence: {obj['confidence']:.3f})")
        
        # 🎨 VISUALIZE THE SEGMENTATION
        print("\n🎨 Generating visualization...")
        vis_door = pipeline.visualize_segmentation(image, results_door, "door", "segmentation_door.png")
        print("   Saved as: segmentation_door.png")
    else:
        print("  No 'door' objects found in the image.")
    
    # EXAMPLE 3: Multiple prompts
    print("\n" + "="*70)
    print("EXAMPLE 3: MULTIPLE PROMPTS - Segment different object types")
    print("="*70)
    
    prompts_list = ["window", "floor", "ceiling"]
    results_multiple = pipeline.process_image_with_multiple_prompts(image, prompts_list)
    
    print("\n" + "-"*70)
    print("Results Summary:")
    print("-"*70)
    for prompt, objects in results_multiple.items():
        print(f"  {prompt.upper()}: {len(objects)} objects found")
        if objects:
            for obj in objects:
                print(f"    → Confidence: {obj['confidence']:.3f}")
    
    # 🎨 VISUALIZE ALL RESULTS
    print("\n🎨 Generating visualizations for all prompts...")
    vis_multiple = pipeline.visualize_multi_prompt_results(image, results_multiple, "segmentation_results")
    print(f"[OK] All visualizations saved to 'segmentation_results' folder")
    
    print("\n" + "="*70)
    print("[COMPLETE] Pipeline execution completed!")
    print("="*70)
    print("\n📝 USAGE GUIDE:")
    print("   • Single prompt: pipeline.process_image_with_prompt(image, text_prompt='wall')")
    print("   • Multiple prompts: pipeline.process_image_with_multiple_prompts(image, ['wall', 'door'])")
    print("   • Visualize results: pipeline.visualize_segmentation(image, objects, 'prompt', 'output.png')")
    print("   • Batch visualize: pipeline.visualize_multi_prompt_results(image, results_dict)")
    print("\n🎨 VISUALIZATION OUTPUT:")
    print("   • Individual prompts saved as: segmentation_{prompt}.png")
    print("   • Multiple prompts saved in: segmentation_results/ folder")
    print("   • Each visualization shows:")
    print("     - Masks overlaid in different colors")
    print("     - Bounding boxes around detected objects")
    print("     - Class labels and confidence scores")
    print("\n   Available output keys for each object:")
    print("     - 'class': The text prompt (class label)")
    print("     - 'confidence': SAM3 confidence score (0-1)")
    print("     - 'bbox': Bounding box [x_min, y_min, x_max, y_max]")
    print("     - 'mask': Binary segmentation mask")
    print("     - 'region': PIL Image of the cropped region")
"""
3-Stage Point Cloud Processing Pipeline
Stage 1: 3D to 2D Projection (Forward Mapping)
Stage 2: 2D Model Inference (SAM3 Segmentation + SAM3 Classification via prompts)
Stage 3: 2D to 3D Back-projection (Reverse Mapping)

Supports both .las and .ply point cloud formats
"""

import os
import re
import sys
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import torch
from tqdm import tqdm


# Import point cloud utilities
try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

try:
    from plyfile import PlyData
    HAS_PLY = True
except ImportError:
    HAS_PLY = False

# Import SAM3 Pipeline
from Samclassifier import Sam3SegmentationPipeline
from util import generate_spherical_image, apply_segmentation_masks

# Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*70)
print("3-Stage Point Cloud Processing Pipeline")
print("="*70)
print(f"[DEVICE] {DEVICE}")

# Initialize SAM3 Pipeline (both segmentation and "classification" via prompts)
print("\n[INIT] Initializing SAM3 Pipeline...")
pipeline = Sam3SegmentationPipeline(device=DEVICE)
print("[OK] SAM3 Pipeline ready!")

# Configuration
offset_x, offset_y = 10, 10
min_bbox_size = 20
resolution_y = 500

# Output folders
output_folder_seg = Path("SegmentedImages3")
output_folder_csv = Path("ClassifiedPoints")
output_folder_seg.mkdir(exist_ok=True)
output_folder_csv.mkdir(exist_ok=True)


def load_point_cloud(file_path):
    """
    Load point cloud from .las or .ply file
    
    Returns:
        point_cloud: Nx3 array of (x, y, z)
        colors: Nx3 array of (r, g, b)
    """
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == ".las" and HAS_LASPY:
        print(f"  [LOAD] Loading LAS file: {file_path}")
        las = laspy.read(file_path)
        point_cloud = np.vstack((las.x, las.y, las.z)).T
        r = (las.red / 65535 * 255).astype(int)
        g = (las.green / 65535 * 255).astype(int)
        b = (las.blue / 65535 * 255).astype(int)
        colors = np.vstack((r, g, b)).T
        
    elif file_ext == ".ply" and HAS_PLY:
        print(f"  [LOAD] Loading PLY file: {file_path}")
        ply = PlyData.read(file_path)
        vertex = ply['vertex']
        point_cloud = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
        colors = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=1).astype(int)
        
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported: .las, .ply")
    
    return point_cloud, colors


def process_point_cloud(file_path, text_prompt="wall", rotation_matrix=None):
    """
    Process point cloud through 3 stages
    
    Args:
        file_path: Path to .las or .ply file
        text_prompt: Text prompt for segmentation (e.g., "wall", "door", "window")
        rotation_matrix: Optional 3x3 rotation matrix for point cloud
        
    Returns:
        classified_points: List of [x, y, z, r, g, b, class, instance_id]
    """
    
    print(f"\n{'='*70}")
    print(f"Processing: {Path(file_path).name}")
    print(f"{'='*70}")
    
    # ============================================================
    # STAGE 1: 3D to 2D Projection (Forward Mapping)
    # ============================================================
    print("\n[STAGE 1] 3D to 2D Projection (Forward Mapping)")
    print("-" * 70)
    
    # Load point cloud
    point_cloud, colors = load_point_cloud(file_path)
    print(f"  [OK] Loaded {len(point_cloud)} points")
    
    # Apply rotation if provided (useful for PLY files)
    if rotation_matrix is not None:
        point_cloud = point_cloud @ rotation_matrix.T
        print(f"  [OK] Applied rotation transformation")
    
    # Calculate center for spherical projection
    center_coordinates = point_cloud.mean(axis=0)
    print(f"  [OK] Center: ({center_coordinates[0]:.2f}, {center_coordinates[1]:.2f}, {center_coordinates[2]:.2f})")
    
    # Generate spherical projection (2D image + mapping)
    image, mapping = generate_spherical_image(center_coordinates, colors, point_cloud, resolution_y)
    print(f"  [OK] Generated 2D projection: {image.shape}")
    
    # Save projection for reference
    file_stem = Path(file_path).stem
    cv2.imwrite(f'projection_{file_stem}.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    presence_map = (mapping != -1).astype(np.uint8) * 255
    cv2.imwrite(f'mapping_mask_{file_stem}.jpg', presence_map)
    print(f"  [OK] Saved projection images")
    
    # ============================================================
    # STAGE 2: 2D Model Inference (SAM3 Segmentation)
    # ============================================================
    print("\n[STAGE 2] 2D Model Inference (SAM3 Segmentation)")
    print("-" * 70)
    
    # Convert to PIL Image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 and image.shape[2] == 3 else image
    image_pil = Image.fromarray(image_rgb).convert("RGB")
    
    # Use SAM3 with text prompt for segmentation
    print(f"  🔍 Using prompt: '{text_prompt}'")
    segmented_objects = pipeline.process_image_with_prompt(image_pil, text_prompt=text_prompt)
    
    print(f"  [OK] Found {len(segmented_objects)} segmented objects")
    
    if len(segmented_objects) == 0:
        print(f"  ⚠️  No objects found for prompt '{text_prompt}'")
        return []
    
    # Visualize segmentation results
    print(f"  🎨 Generating visualization...")
    pipeline.visualize_segmentation(image_pil, segmented_objects, text_prompt, 
                                    f"segmentation_{text_prompt}_{file_stem}.png")
    
    # ============================================================
    # STAGE 3: 2D to 3D Back-projection (Reverse Mapping)
    # ============================================================
    print("\n[STAGE 3] 2D to 3D Back-projection (Reverse Mapping)")
    print("-" * 70)
    
    classified_points = []
    img_height, img_width = resolution_y, 2 * resolution_y
    
    for obj_idx, obj in enumerate(segmented_objects):
        mask = obj['mask']
        class_label = obj['class']  # Text prompt IS the class label
        instance_id = obj['region_idx']
        
        # Convert mask to numpy if needed
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # Get all pixels belonging to this mask
        pixels = np.argwhere(mask)
        
        print(f"  Object {obj_idx} (Class: '{class_label}'): {len(pixels)} pixels")
        
        # Back-project each pixel to 3D
        for y, x in pixels:
            # Ensure coordinates are within bounds
            if 0 <= y < mapping.shape[0] and 0 <= x < mapping.shape[1]:
                point_idx = mapping[y, x]
                
                # Skip if no point maps to this pixel
                if point_idx == -1:
                    continue
                
                # Get 3D coordinates and color
                x_3d, y_3d, z_3d = point_cloud[point_idx]
                r_color, g_color, b_color = colors[point_idx]
                
                # Append classified point
                classified_points.append([
                    x_3d, y_3d, z_3d,           # 3D coordinates
                    r_color, g_color, b_color, # RGB color
                    class_label,                # Semantic class (from SAM3 prompt)
                    instance_id                 # Instance ID
                ])
    
    print(f"  [OK] Back-projected {len(classified_points)} points to 3D")
    
    # Save classified point cloud to CSV
    df = pd.DataFrame(classified_points, columns=[
        "x", "y", "z", "r", "g", "b", "class", "instance_id"
    ])
    
    output_csv = output_folder_csv / f"{file_stem}_{text_prompt}_classified.csv"
    df.to_csv(output_csv, index=False, float_format="%.6f")
    print(f"  [OK] Saved to: {output_csv}")
    
    print(f"\n✅ Processing completed: {len(classified_points)} classified points")
    
    return classified_points


if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("Point Cloud Processing Configuration")
    print("="*70)
    
    # ⭐ CONFIGURE YOUR INPUT HERE ⭐
    # Specify the file to process
    input_file = "living.ply"  # Change this to your .ply or .las file
    
    # ⭐ TEXT PROMPTS (MULTIPLE SUPPORTED) ⭐
    # Add as many prompts as you want - each will create a separate output
    text_prompts = ["wall", "chair", "table", "floor"]  # Change to your desired objects
    
    # Optional rotation matrix for PLY files
    # Rotate 90 degrees around Y-axis (useful for some PLY files)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    
    print(f"\n[FILE] Input file: {input_file}")
    print(f"[PROMPTS] Text prompts: {text_prompts}")
    print(f"[CONFIG] Using rotation transformation: Yes")
    print(f"[INFO] Processing {len(text_prompts)} prompts")
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"\n[ERROR] Error: File not found: {input_file}")
        print(f"   Please ensure the file exists in the current directory")
        exit(1)
    
    # Process the point cloud with MULTIPLE PROMPTS
    try:
        all_results = {}
        
        for prompt_idx, text_prompt in enumerate(text_prompts, 1):
            print(f"\n{'='*70}")
            print(f"Processing Prompt {prompt_idx}/{len(text_prompts)}: '{text_prompt}'")
            print(f"{'='*70}")
            
            classified_points = process_point_cloud(
                file_path=input_file,
                text_prompt=text_prompt,
                rotation_matrix=rotation_matrix
            )
            
            all_results[text_prompt] = classified_points
        
        # Print final summary
        print("\n" + "="*70)
        print("✅ All prompts processed successfully!")
        print("="*70)
        print(f"\n[SUMMARY] Final Results Summary:")
        print(f"   • Input file: {input_file}")
        print(f"   • Total prompts processed: {len(text_prompts)}")
        print(f"\n   Breakdown by prompt:")
        total_classified = 0
        for prompt, points in all_results.items():
            print(f"     - '{prompt}': {len(points):,} classified points")
            total_classified += len(points)
        print(f"\n   • Total classified points: {total_classified:,}")
        print(f"\n   Output files:")
        for prompt in text_prompts:
            csv_name = f"ClassifiedPoints/{Path(input_file).stem}_{prompt}_classified.csv"
            vis_name = f"segmentation_{prompt}_{Path(input_file).stem}.png"
            print(f"     - {csv_name}")
            print(f"     - {vis_name}")
        
    except Exception as e:
        print(f"\n❌ Error processing point cloud:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()


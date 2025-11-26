import os
import re
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pathlib import Path
from plyfile import PlyData
from torchvision import transforms

# Import SAM3 from transformers
from transformers import Sam3Processor, Sam3Model

# Assuming these are custom modules provided by the user or available in the environment
from classifier import EfficientNetB7Classifier
from util import generate_spherical_image, apply_segmentation_masks

# Define your device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load SAM3 model
print("📥 Loading SAM3 model from HuggingFace...")
model = Sam3Model.from_pretrained("facebook/sam3").to(DEVICE)
processor = Sam3Processor.from_pretrained("facebook/sam3")
print("✓ SAM3 Model loaded successfully!")

# Load the EfficientNetB7 classifier model
model_classifier = EfficientNetB7Classifier().to(DEVICE)
model_classifier.load_state_dict(torch.load("Models/EffiecientNetB7/best_model (1).pth", map_location=DEVICE))
model_classifier.eval()
class_names = model_classifier.class_names

# Define the image transformations for the classifier
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define output folders and create them if they don't exist
output_folder_seg = Path("SegmentedImages3")
output_folder_csv = Path("ClassifiedPoints")
output_folder_seg.mkdir(exist_ok=True)
output_folder_csv.mkdir(exist_ok=True)

# Define parameters for bounding box adjustments and image resolution
offset_x, offset_y = 10, 10
min_bbox_size = 20
resolution_y = 500

def process_ply(ply_path):
    """
    Processes a PLY file:
    1. Reads point cloud data and colors.
    2. Rotates the point cloud.
    3. Generates a spherical image projection from the point cloud.
    4. Uses SAM3 to generate segmentation masks on the spherical image.
    5. Classifies segmented objects using a pre-trained EfficientNetB7 model.
    6. Applies segmentation masks to the original image and saves it.
    7. Creates a CSV file with classified 3D points.

    Args:
        ply_path (str): Path to the input PLY file.
    """
    ply = PlyData.read(ply_path)
    vertex = ply['vertex']
    raw_points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)

    # Rotate 90 degrees to the right (clockwise) around Z-axis
    # This rotation matrix rotates around the Y-axis (vertical axis in a typical 3D coordinate system)
    # to align with a common spherical projection orientation.
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    point_cloud = raw_points @ rotation_matrix.T
    colors = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=1)

    # Calculate the center of the point cloud for spherical projection
    center_coordinates = point_cloud.mean(axis=0)
    
    # Generate spherical image and mapping from 3D points to 2D pixels
    image, mapping = generate_spherical_image(center_coordinates, colors, point_cloud, resolution_y)
    
    # Create and save a presence map (mask indicating where points are projected)
    presence_map = (mapping != -1).astype(np.uint8) * 255
    cv2.imwrite("mapping_mask.jpg", presence_map)
    cv2.imwrite('original_projection.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Convert BGR to RGB for SAM3 processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb).convert("RGB")
    
    # Generate masks using SAM3
    inputs = processor(images=image_pil, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process results to get instance segmentation
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    
    masks = results.get('masks', [])
    if len(masks) == 0:
        print("No masks generated")
        return

    img_height, img_width = resolution_y, 2 * resolution_y
    class_labels = {}  # Stores object ID to class label mapping
    id_to_pixels = {}  # Stores object ID to pixel coordinates mapping

    # Iterate through each generated mask
    for idx, mask in enumerate(masks):
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        
        # Create bounding box from mask
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        if not rows.any() or not cols.any():
            continue
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Apply offsets to bounding box for better cropping
        x_min = max(0, x_min - offset_x)
        y_min = max(0, y_min - offset_y)
        x_max = min(img_width, x_max + offset_x)
        y_max = min(img_height, y_max + offset_y)

        # Skip invalid or too small bounding boxes
        if x_max <= x_min or y_max <= y_min:
            continue
        if (x_max - x_min) < min_bbox_size or (y_max - y_min) < min_bbox_size:
            continue

        # Crop the object from the spherical image
        cropped_object = image_rgb[y_min:y_max, x_min:x_max]
        if cropped_object.size == 0:
            continue
        
        # Convert cropped object to PIL Image for classification
        cropped_pil_image = Image.fromarray(cropped_object).convert("RGB")
        image_tensor = transform(cropped_pil_image).unsqueeze(0).to(DEVICE)

        # Classify the cropped object using the pre-trained EfficientNetB7 model
        with torch.no_grad():
            output = model_classifier(image_tensor)
            _, predicted_class = torch.max(output, 1)
            class_label = class_names[predicted_class.item()]

        class_labels[idx] = class_label  # Store the classified label
        
        # Get pixel coordinates for the current mask
        pixels = np.argwhere(mask_np)
        # Store pixels associated with this object ID
        id_to_pixels[idx] = pixels if idx not in id_to_pixels else np.vstack((id_to_pixels[idx], pixels))

    # Convert masks to format compatible with apply_segmentation_masks
    masks_for_viz = []
    for idx, mask in enumerate(masks):
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        masks_for_viz.append({'segmentation': mask_np, 'id': idx})
    
    # Apply segmentation masks and class labels to the original spherical image for visualization
    segmented_image = apply_segmentation_masks(masks_for_viz, image, class_labels)
    cv2.imwrite(str(output_folder_seg / "segmented_result.jpg"), cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

    classified_points = []
    # Iterate through each object and its pixels to get 3D classified points
    for obj_id, pixels in id_to_pixels.items():
        label = class_labels.get(obj_id, "Unknown") # Get the class label for the object
        for y, x in pixels:
            # Ensure pixel coordinates are within the mapping dimensions
            if 0 <= y < mapping.shape[0] and 0 <= x < mapping.shape[1]:
                point_idx = mapping[y, x] # Get the original 3D point index from the mapping
                if point_idx == -1: # Skip if no 3D point maps to this pixel
                    continue
                
                # Retrieve 3D coordinates and colors
                x_, y_, z_ = point_cloud[point_idx]
                r_, g_, b_ = colors[point_idx]
                
                # Append classified point data
                classified_points.append([x_, y_, z_, r_, g_, b_, label, obj_id])

    # Create a Pandas DataFrame and save to CSV
    df = pd.DataFrame(classified_points, columns=["x", "y", "z", "r", "g", "b", "class", "instance_id"])
    df.to_csv(output_folder_csv / "classified_output.csv", index=False, float_format="%.6f")

if __name__ == "__main__":
    # Example usage: Change to your actual .ply path
    process_ply("stabilized.ply")
    print("\n🎉 PLY file processed successfully.")
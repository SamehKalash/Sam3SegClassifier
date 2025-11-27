"""
Minimal SAM3 Quick Test
✅ Works with gated models (requires login)
"""

import torch
from transformers import Sam3Processor, Sam3Model
from PIL import Image
import requests
import os

# 1️⃣ Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2️⃣ Load model & processor (with auth token)
print("Loading SAM3 model...")
try:
    model = Sam3Model.from_pretrained(
        "facebook/sam3",
        use_auth_token=True
    ).to(device)
    processor = Sam3Processor.from_pretrained(
        "facebook/sam3",
        use_auth_token=True
    )
    print("✅ Model loaded!")
except Exception as e:
    print("❌ Error loading model:", e)
    exit(1)

# 3️⃣ Load a test image
url = "http://images.cocodataset.org/val2017/000000077595.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
print(f"Loaded image: {image.size}")

# 4️⃣ Run segmentation for a text prompt
prompt = "cat"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]

print(f"Found {len(results['masks'])} {prompt}(s)")
print("Confidence scores:", [f"{s:.2f}" for s in results["scores"].tolist()])

# 5️⃣ Save masks overlay (optional)
try:
    # Simple overlay function
    from PIL import ImageDraw, ImageFont

    def overlay_masks(image, masks):
        img = image.copy().convert("RGBA")
        for mask in masks:
            mask_img = Image.fromarray((mask.cpu().numpy() * 255).astype("uint8"))
            color = (255, 0, 0, 100)  # red with transparency
            red_mask = Image.new("RGBA", img.size, color)
            img.paste(red_mask, mask=mask_img)
        return img

    result_image = overlay_masks(image, results["masks"])
    output_path = "quicktest_output.png"
    result_image.save(output_path)
    print(f"✅ Saved segmentation to {output_path}")
except Exception as e:
    print("⚠️ Could not save visualization:", e)

print("🎯 SAM3 quick test complete!")

"""
Simplest possible SAM3 example
Use this to verify your installation works
"""

print("🚀 SAM3 Quick Test - Starting...")

# 1. Import libraries
print("\n1️⃣ Importing libraries...")
try:
    import torch
    from transformers import Sam3Processor, Sam3Model
    from PIL import Image
    import requests
    print("   ✓ All libraries imported successfully")
except ImportError as e:
    print(f"   ❌ Error: {e}")
    print("   Run: pip install -r requirements.txt")
    exit(1)

# 2. Check device
print("\n2️⃣ Checking device...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Using: {device}")

# 3. Load model
print("\n3️⃣ Loading SAM3 model...")
print("   (This takes a few minutes on first run)")
try:
    model = Sam3Model.from_pretrained("facebook/sam3")
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    print("   ✓ Model loaded!")
except Exception as e:
    print(f"   ❌ Error loading model: {e}")
    print("   Make sure you've accepted the license on HuggingFace:")
    print("   https://huggingface.co/facebook/sam3")
    exit(1)

# 4. Load test image
print("\n4️⃣ Loading test image...")
try:
    url = "http://images.cocodataset.org/val2017/000000077595.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    print(f"   ✓ Image loaded: {image.size}")
except Exception as e:
    print(f"   ❌ Error loading image: {e}")
    exit(1)

# 5. Run segmentation
print("\n5️⃣ Running segmentation with text prompt: 'cat'...")
try:
    inputs = processor(images=image, text="cat", return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    
    print(f"   ✓ Found {len(results['masks'])} cats!")
    print(f"   Confidence scores: {[f'{s:.2f}' for s in results['scores'].tolist()]}")
except Exception as e:
    print(f"   ❌ Error during inference: {e}")
    exit(1)

# 6. Save result
print("\n6️⃣ Saving result...")
try:
    from utils import overlay_masks
    result_image = overlay_masks(image, results["masks"])
    result_image.save("quicktest_output.png")
    print("   ✓ Saved to: quicktest_output.png")
except Exception as e:
    print(f"   ⚠️ Could not save visualization: {e}")
    print("   But segmentation worked!")

# Done!
print("\n" + "="*50)
print("✅ SUCCESS! SAM3 is working on your system!")
print("="*50)
print("\nNext steps:")
print("  • Try other test scripts (test_image_basic.py, etc.)")
print("  • Use your own images")
print("  • Experiment with different prompts")
print("\nHave fun segmenting! 🎯")

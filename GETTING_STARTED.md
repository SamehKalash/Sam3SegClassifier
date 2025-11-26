# 🎯 SAM3 Testing Suite - Quick Reference

## What You Have

A complete testing environment for **SAM3 (0.9B parameters)** optimized for Raspberry Pi 5.

## Files Created

### 📚 Documentation
- `README.md` - Main overview and quick start
- `GUIDE.md` - Comprehensive usage guide
- `GETTING_STARTED.md` - This file

### 🔧 Setup Scripts
- `setup.sh` - Linux/Raspberry Pi setup (use this on your Pi 5)
- `setup.ps1` - Windows PowerShell setup (for testing on Windows)
- `requirements.txt` - Python dependencies

### 🧪 Test Scripts
- `quicktest.py` - **START HERE** - Simple verification test
- `test_image_basic.py` - Image segmentation with text/box prompts
- `test_tracker.py` - Interactive point/box segmentation
- `test_video_streaming.py` - **Recommended for Pi 5** - Streaming video
- `test_video_full.py` - Batch video processing
- `benchmark.py` - Performance testing

### 🛠️ Utilities
- `utils.py` - Helper functions for visualization

## Quick Start (3 Steps)

### On Your Raspberry Pi 5:

```bash
# Step 1: Install dependencies
chmod +x setup.sh
./setup.sh

# Step 2: Activate environment
source venv/bin/activate

# Step 3: Run quick test
python quicktest.py
```

That's it! If `quicktest.py` succeeds, you're ready to go! 🎉

## What Each Test Does

### 1. `quicktest.py` (RECOMMENDED FIRST)
**Time**: ~2-5 minutes (first run), ~30-60 seconds (after)  
**What it does**: Loads model, segments "cat" in test image  
**Good for**: Verifying installation works

```bash
python quicktest.py
```

### 2. `test_image_basic.py`
**Time**: ~1-2 minutes  
**What it does**: Tests text prompts, bounding boxes  
**Good for**: Understanding different prompting methods

```bash
python test_image_basic.py
```

### 3. `test_tracker.py`
**Time**: ~2-3 minutes  
**What it does**: Interactive segmentation (points, boxes)  
**Good for**: Click-to-segment applications

```bash
python test_tracker.py
```

### 4. `test_video_streaming.py` ⭐
**Time**: ~5-10 minutes  
**What it does**: Processes video frames one-by-one  
**Good for**: Real-time applications, webcam feeds  
**Best for Pi 5**: Yes - lower memory usage

```bash
python test_video_streaming.py
```

### 5. `test_video_full.py`
**Time**: ~10-15 minutes  
**What it does**: Loads entire video, better tracking  
**Good for**: Offline processing, best quality  
**Best for Pi 5**: Use only if you have 8GB RAM

```bash
python test_video_full.py
```

### 6. `benchmark.py`
**Time**: ~1 minute  
**What it does**: Measures FPS, memory usage  
**Good for**: Performance testing

```bash
python benchmark.py
```

## Expected Results on Raspberry Pi 5

### First Run
- Model download: ~5 minutes (requires ~3-4GB download)
- First inference: ~60-90 seconds (model loading + compilation)

### Subsequent Runs
- Model loading: ~10-20 seconds
- Image inference: ~2-5 seconds per image
- Video streaming: ~3-10 seconds per frame

### Memory Usage
- Model loaded: ~2-3 GB
- During inference: ~3-4 GB
- Total system: 4-6 GB (8GB Pi recommended)

## Common Use Cases

### Use Case 1: Segment Objects in Photos
```bash
python test_image_basic.py
# Modify the text prompts in the file for your objects
```

### Use Case 2: Interactive Selection
```bash
python test_tracker.py
# Click coordinates can be adjusted in the file
```

### Use Case 3: Track Objects in Video
```bash
python test_video_streaming.py
# Change text prompt to track different objects
```

### Use Case 4: Process Your Own Images
```python
from transformers import Sam3Model, Sam3Processor
from PIL import Image

model = Sam3Model.from_pretrained("facebook/sam3")
processor = Sam3Processor.from_pretrained("facebook/sam3")

# Your image
image = Image.open("your_photo.jpg")

# Your prompt
inputs = processor(images=image, text="your object", return_tensors="pt")
outputs = model(**inputs)
results = processor.post_process_instance_segmentation(
    outputs, threshold=0.5, mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]

print(f"Found {len(results['masks'])} objects!")
```

## Troubleshooting

### Issue: Model won't download
**Solution**: Accept license at https://huggingface.co/facebook/sam3

### Issue: Out of memory
**Solution**: 
- Close other applications
- Use streaming video mode
- Reduce image size
- Enable swap on Pi

### Issue: Very slow
**Solution**: 
- This is normal on Pi 5 (no GPU)
- Expected: 0.2-0.5 FPS
- Use smaller images for faster processing

### Issue: Import errors
**Solution**:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Tips for Best Performance on Pi 5

1. **Use streaming mode** for video (lower memory)
2. **Resize images** to 640x480 or 1024x768 before processing
3. **Process fewer frames** in videos (every 2nd or 3rd frame)
4. **Close other apps** to free memory
5. **Use bfloat16** dtype when possible
6. **Enable swap** if using 4GB Pi 5

## Next Steps

After running the tests:

1. **Modify prompts** - Change text in scripts to segment your objects
2. **Use your images** - Replace image URLs with your files
3. **Integrate webcam** - Use OpenCV to capture live video
4. **Build an app** - Create a GUI with tkinter or web interface
5. **Optimize** - Experiment with image sizes and settings

## Example Prompts to Try

### General Objects
- "person", "car", "dog", "cat", "chair", "table"
- "laptop", "phone", "bottle", "cup", "book"

### Specific Parts
- "face", "hand", "eye", "ear", "nose"
- "wheel", "window", "door", "handle"

### Multiple Words
- "red car", "black dog", "wooden chair"
- "person wearing hat", "cat on couch"

### Abstract Concepts
- "text", "logo", "sign", "button"
- "shadow", "reflection", "pattern"

## Resources

- **Model Page**: https://huggingface.co/facebook/sam3
- **GitHub**: https://github.com/facebookresearch/sam3
- **Demo**: https://huggingface.co/spaces/akhaliq/sam3
- **Transformers Docs**: https://huggingface.co/docs/transformers

## Support

If you have issues:
1. Check `GUIDE.md` for detailed documentation
2. Review error messages carefully
3. Ensure you've accepted the model license
4. Verify Python 3.8+ and dependencies installed

---

**Ready to start?** Run `python quicktest.py` now! 🚀

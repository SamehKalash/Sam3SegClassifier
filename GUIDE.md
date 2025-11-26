# SAM3 (Segment Anything Model 3) - Raspberry Pi 5 Testing Guide

## Overview

This repository contains testing scripts for the new **SAM3 model (0.9B parameters)** from Meta AI, optimized for testing on Raspberry Pi 5.

### What is SAM3?

SAM3 is Meta's latest foundation model for segmentation that introduces:
- **Promptable Concept Segmentation (PCS)**: Segment objects using text prompts
- **Promptable Visual Segmentation (PVS)**: Interactive segmentation with points/boxes
- **Video tracking**: Track objects across video frames
- **Open-vocabulary**: Handle 270K+ unique concepts

### Model Capabilities

1. **Image Segmentation**
   - Text prompts: "cat", "person", "car", etc.
   - Visual prompts: points, bounding boxes, masks
   - Mixed prompts: combine text with visual cues

2. **Video Segmentation**
   - Text-based tracking: automatically track all instances
   - Interactive tracking: click to track specific objects
   - Streaming mode: real-time frame-by-frame processing

3. **Multiple Modes**
   - SAM3: Text-based concept segmentation
   - SAM3 Tracker: Interactive visual segmentation
   - SAM3 Video: Video tracking with text
   - SAM3 Tracker Video: Interactive video tracking

## Installation

### On Raspberry Pi 5

```bash
# Clone or copy files to your Pi
cd SAM

# Run setup script
chmod +x setup.sh
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### On Windows (for development)

```powershell
# Run PowerShell setup
.\setup.ps1

# Or manually:
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Image Segmentation

```bash
python test_image_basic.py
```

This will:
- Download the SAM3 model (first run only)
- Test text-based segmentation ("cat", "ear")
- Test bounding box segmentation
- Save visualizations to PNG files

**Expected output:**
- `output_cat.png` - Segmented cats
- `output_ear.png` - Segmented ears
- `output_box.png` - Object in bounding box

### 2. Interactive Tracker

```bash
python test_tracker.py
```

Tests interactive segmentation:
- Single point click
- Multiple points for refinement
- Bounding box input
- Multiple objects
- Positive/negative points

### 3. Video Streaming (Recommended for Pi 5)

```bash
python test_video_streaming.py
```

Processes video frames one-by-one in real-time mode:
- Lower memory usage
- Suitable for live camera feeds
- Optimized for Pi 5 performance

### 4. Full Video Processing

```bash
python test_video_full.py
```

Loads entire video into memory:
- Better for offline processing
- Higher memory usage
- Better tracking quality

### 5. Performance Benchmark

```bash
python benchmark.py
```

Quick performance test showing:
- Model loading time
- Inference speed (FPS)
- Memory usage

## File Structure

```
SAM/
├── README.md                    # This file
├── GUIDE.md                     # Detailed guide
├── requirements.txt             # Python dependencies
├── setup.sh                     # Linux/Pi setup script
├── setup.ps1                    # Windows PowerShell setup
│
├── test_image_basic.py          # Basic image tests
├── test_tracker.py              # Interactive segmentation
├── test_video_streaming.py      # Streaming video mode
├── test_video_full.py           # Batch video mode
├── benchmark.py                 # Performance testing
│
└── utils.py                     # Helper functions
```

## Usage Examples

### Text-Based Segmentation

```python
from transformers import Sam3Model, Sam3Processor
from PIL import Image

model = Sam3Model.from_pretrained("facebook/sam3")
processor = Sam3Processor.from_pretrained("facebook/sam3")

image = Image.open("photo.jpg")
inputs = processor(images=image, text="person", return_tensors="pt")

outputs = model(**inputs)
results = processor.post_process_instance_segmentation(
    outputs, threshold=0.5, mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]

# results contains: masks, boxes, scores
```

### Interactive Point-Based

```python
from transformers import Sam3TrackerProcessor, Sam3TrackerModel

model = Sam3TrackerModel.from_pretrained("facebook/sam3")
processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")

# Click at coordinate (500, 375)
input_points = [[[[500, 375]]]]
input_labels = [[[1]]]  # 1 = positive, 0 = negative

inputs = processor(
    images=image,
    input_points=input_points,
    input_labels=input_labels,
    return_tensors="pt"
)

outputs = model(**inputs)
masks = processor.post_process_masks(
    outputs.pred_masks,
    inputs["original_sizes"]
)[0]
```

### Streaming Video

```python
from transformers import Sam3VideoModel, Sam3VideoProcessor

model = Sam3VideoModel.from_pretrained("facebook/sam3")
processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

# Initialize streaming session
session = processor.init_video_session(
    inference_device="cpu",
    dtype=torch.bfloat16
)

# Add text prompt
session = processor.add_text_prompt(
    inference_session=session,
    text="person"
)

# Process frames one by one
for frame_idx, frame in enumerate(video_frames):
    inputs = processor(images=frame, return_tensors="pt")
    
    outputs = model(
        inference_session=session,
        frame=inputs.pixel_values[0],
        reverse=False
    )
    
    results = processor.postprocess_outputs(
        session, outputs,
        original_sizes=inputs.original_sizes
    )
```

## Performance Tips for Raspberry Pi 5

### 1. Memory Optimization
- Use `dtype=torch.bfloat16` when possible
- Process smaller batches (1 frame at a time)
- Use streaming mode for video
- Clear cache between runs: `torch.cuda.empty_cache()` (if using GPU)

### 2. Speed Optimization
- Reduce input resolution (resize to 480p or 720p)
- Use CPU optimizations
- Limit frame processing (process every 2nd or 3rd frame)
- Use quantized models when available

### 3. Best Practices
```python
# Use efficient data types
model = model.to(device, dtype=torch.bfloat16)

# Process in batches of 1
for frame in video_frames:
    # Process single frame
    pass

# Use streaming mode for real-time
session = processor.init_video_session(
    inference_device="cpu",
    processing_device="cpu",
    video_storage_device="cpu"
)
```

## Expected Performance on Raspberry Pi 5

| Task | Expected FPS | Memory Usage |
|------|-------------|--------------|
| Image (text) | 0.2-0.5 | ~2-3 GB |
| Image (point) | 0.3-0.7 | ~2-3 GB |
| Video (streaming) | 0.1-0.3 | ~3-4 GB |
| Video (batch) | 0.1-0.2 | ~4-5 GB |

*Note: Actual performance varies based on input size and model configuration*

## Model Access

⚠️ **Important**: SAM3 requires accepting Meta's terms on HuggingFace.

1. Create a HuggingFace account
2. Visit https://huggingface.co/facebook/sam3
3. Accept the terms and conditions
4. Login on your Pi: `huggingface-cli login`

## Troubleshooting

### Out of Memory
- Reduce batch size to 1
- Use smaller input images
- Enable swap on Pi 5
- Use streaming mode instead of batch mode

### Slow Performance
- Normal for Pi 5 (no GPU)
- Reduce input resolution
- Process fewer frames
- Use lighter models if available

### Model Download Fails
- Check internet connection
- Verify HuggingFace access token
- Ensure sufficient disk space (~4GB)

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade transformers torch pillow
```

## Advanced Usage

### Custom Image Input
```python
from PIL import Image

# Load your own image
image = Image.open("/path/to/your/image.jpg")

# Segment with custom prompt
inputs = processor(images=image, text="your custom prompt", return_tensors="pt")
outputs = model(**inputs)
```

### Webcam Integration
```python
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Process frame
    inputs = processor(images=frame_pil, text="person", return_tensors="pt")
    outputs = model(**inputs)
    
    # Display results
    # ... (implement visualization)
```

### Save/Load Sessions
```python
# For video sessions that take time to process
import pickle

# Save session state
with open('session.pkl', 'wb') as f:
    pickle.dump(inference_session, f)

# Load later
with open('session.pkl', 'rb') as f:
    inference_session = pickle.load(f)
```

## Resources

- **HuggingFace Model**: https://huggingface.co/facebook/sam3
- **GitHub Repository**: https://github.com/facebookresearch/sam3
- **Interactive Demo**: https://huggingface.co/spaces/akhaliq/sam3
- **Paper/Documentation**: Check GitHub repo for latest research paper

## License

SAM3 model is provided by Meta AI under their license terms. See the HuggingFace model page for details.

## Contributing

Feel free to submit issues or improvements to these testing scripts!

## Version Info

- **SAM3 Version**: Latest from HuggingFace
- **Transformers**: >=4.50.0
- **PyTorch**: >=2.0.0
- **Tested on**: Raspberry Pi 5 (8GB RAM recommended)

---

Happy segmenting! 🎯

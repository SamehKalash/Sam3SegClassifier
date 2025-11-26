# SAM3 Testing on Raspberry Pi 5

This workspace contains scripts for testing the SAM3 (Segment Anything Model 3) model on Raspberry Pi 5.

## Model Information
- **Model**: facebook/sam3
- **Size**: 0.9B parameters
- **Capabilities**: 
  - Image segmentation with text/visual prompts
  - Video segmentation and tracking
  - Streaming inference support

## Requirements

Install the required packages:
```bash
pip install torch transformers pillow opencv-python matplotlib accelerate
```

For video support:
```bash
pip install av
```

## Usage

### 1. Image Segmentation
Test basic image segmentation with text prompts:
```bash
python test_image_basic.py
```

### 2. Streaming Video Inference
Test real-time video processing (optimized for Pi 5):
```bash
python test_video_streaming.py
```

### 3. Full Video Processing
Process entire videos (requires more memory):
```bash
python test_video_full.py
```

## Performance Tips for Raspberry Pi 5

1. **Use CPU with optimizations**: The model will automatically use CPU mode
2. **Use bfloat16 when possible**: Reduces memory usage
3. **Process smaller frames**: Resize input frames to 480p or 720p
4. **Streaming mode**: Use streaming inference for real-time applications
5. **Limit batch size**: Process one frame at a time

## Files

- `test_image_basic.py` - Basic image segmentation with text prompts
- `test_video_streaming.py` - Streaming video inference (recommended for Pi 5)
- `test_video_full.py` - Full video processing
- `utils.py` - Helper functions for visualization

## Notes

- The model requires accepting Meta's terms on HuggingFace before first download
- Expect slower inference on Pi 5 compared to GPU workstations
- Streaming mode is recommended for real-time applications on Pi hardware

"""
SAM3 Streaming Video Inference Test
Optimized for Raspberry Pi 5 - processes frames one by one
"""

import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video
from accelerate import Accelerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time

def visualize_frame_with_masks(frame, masks, object_ids, frame_idx, save_path):
    """Visualize a single frame with segmentation masks"""
    plt.figure(figsize=(12, 6))
    
    # Convert tensor to numpy if needed
    if isinstance(frame, torch.Tensor):
        frame_np = frame.permute(1, 2, 0).cpu().numpy()
    else:
        frame_np = np.array(frame)
    
    # Normalize if needed
    if frame_np.max() > 1.0:
        frame_np = frame_np / 255.0
    
    # Show original frame
    plt.subplot(1, 2, 1)
    plt.imshow(frame_np)
    plt.title(f"Frame {frame_idx}")
    plt.axis('off')
    
    # Show frame with masks
    plt.subplot(1, 2, 2)
    plt.imshow(frame_np)
    
    # Overlay masks
    if masks is not None and len(masks) > 0:
        masks_np = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks
        
        # Create color map
        colors = plt.cm.rainbow(np.linspace(0, 1, len(object_ids)))
        
        for i, (mask, obj_id) in enumerate(zip(masks_np, object_ids)):
            # Get binary mask
            binary_mask = mask > 0.5
            
            # Create colored overlay
            colored_mask = np.zeros((*binary_mask.shape, 4))
            colored_mask[binary_mask] = [*colors[i][:3], 0.5]
            
            plt.imshow(colored_mask)
    
    plt.title(f"Segmentation (Objects: {len(object_ids)})")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("=" * 70)
    print("SAM3 Streaming Video Inference Test - Raspberry Pi 5")
    print("=" * 70)
    
    # Device setup
    device = Accelerator().device
    print(f"\n✓ Using device: {device}")
    
    # Load model with bfloat16 for efficiency
    print("\n📥 Loading SAM3 Video model...")
    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    print("✓ Model loaded successfully!")
    
    # Load video
    print("\n🎥 Loading test video...")
    video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
    video_frames, _ = load_video(video_url)
    print(f"✓ Video loaded: {len(video_frames)} frames")
    
    # Limit frames for Pi 5 performance
    num_frames = min(30, len(video_frames))
    video_frames = video_frames[:num_frames]
    print(f"  Processing {num_frames} frames for testing")
    
    # Initialize streaming session
    print("\n🔧 Initializing streaming inference session...")
    streaming_inference_session = processor.init_video_session(
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )
    print("✓ Session initialized")
    
    # Add text prompt
    text_prompt = "person"
    print(f"\n🎯 Adding text prompt: '{text_prompt}'")
    streaming_inference_session = processor.add_text_prompt(
        inference_session=streaming_inference_session,
        text=text_prompt,
    )
    
    # Process frames one by one (streaming mode)
    print(f"\n🎬 Processing {num_frames} frames in streaming mode...")
    print("   (This is optimized for real-time processing on Pi 5)")
    
    streaming_outputs_per_frame = {}
    start_time = time.time()
    
    for frame_idx, frame in enumerate(video_frames):
        frame_start = time.time()
        
        # Process the frame using the processor
        inputs = processor(images=frame, device=device, return_tensors="pt")
        
        # Process frame using streaming inference
        model_outputs = model(
            inference_session=streaming_inference_session,
            frame=inputs.pixel_values[0],
            reverse=False,
        )
        
        # Post-process outputs
        processed_outputs = processor.postprocess_outputs(
            streaming_inference_session,
            model_outputs,
            original_sizes=inputs.original_sizes,
        )
        
        streaming_outputs_per_frame[frame_idx] = processed_outputs
        
        frame_time = time.time() - frame_start
        fps = 1.0 / frame_time if frame_time > 0 else 0
        
        if (frame_idx + 1) % 5 == 0 or frame_idx == 0:
            print(f"  Frame {frame_idx + 1}/{num_frames} - {frame_time:.2f}s ({fps:.1f} FPS)")
    
    total_time = time.time() - start_time
    avg_fps = num_frames / total_time
    
    print(f"\n✓ Streaming inference complete!")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Average time per frame: {total_time/num_frames:.2f}s")
    
    # Analyze results
    print("\n📊 Results Analysis:")
    frame_0_outputs = streaming_outputs_per_frame[0]
    print(f"  Detected {len(frame_0_outputs['object_ids'])} objects in first frame")
    print(f"  Object IDs: {frame_0_outputs['object_ids'].tolist()}")
    print(f"  Boxes shape (XYXY format): {frame_0_outputs['boxes'].shape}")
    print(f"  Masks shape: {frame_0_outputs['masks'].shape}")
    
    # Visualize key frames
    print("\n🎨 Generating visualizations...")
    
    key_frames = [0, num_frames // 2, num_frames - 1]
    for frame_idx in key_frames:
        output = streaming_outputs_per_frame[frame_idx]
        visualize_frame_with_masks(
            video_frames[frame_idx],
            output['masks'],
            output['object_ids'],
            frame_idx,
            f"streaming_frame_{frame_idx:03d}.png"
        )
        print(f"  ✓ Saved: streaming_frame_{frame_idx:03d}.png")
    
    # Track object consistency
    print("\n🔍 Object Tracking Analysis:")
    object_counts = [len(streaming_outputs_per_frame[i]['object_ids']) for i in range(num_frames)]
    print(f"  Min objects per frame: {min(object_counts)}")
    print(f"  Max objects per frame: {max(object_counts)}")
    print(f"  Avg objects per frame: {sum(object_counts)/len(object_counts):.1f}")
    
    # Count unique object IDs
    all_object_ids = set()
    for i in range(num_frames):
        all_object_ids.update(streaming_outputs_per_frame[i]['object_ids'].tolist())
    print(f"  Total unique objects tracked: {len(all_object_ids)}")
    
    print("\n" + "=" * 70)
    print("✅ Streaming video test completed successfully!")
    print("=" * 70)
    print("\nGenerated files:")
    for frame_idx in key_frames:
        print(f"  - streaming_frame_{frame_idx:03d}.png")
    
    print("\n💡 Tips for Raspberry Pi 5:")
    print("  - Streaming mode processes frames as they arrive")
    print("  - Lower FPS is expected on Pi 5 (GPU would be much faster)")
    print("  - Re~duce frame resolution for better performance")
    print("  - Use smaller batches or single frame processing")

if __name__ == "__main__":
    main()

"""
SAM3 Full Video Processing Test
Process entire video with all frames loaded (requires more memory)
"""

import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video
from accelerate import Accelerator
import time

def main():
    print("=" * 70)
    print("SAM3 Full Video Processing Test - Raspberry Pi 5")
    print("=" * 70)
    print("\n⚠️  WARNING: This mode loads all frames into memory")
    print("   For large videos, use test_video_streaming.py instead")
    
    # Device setup
    device = Accelerator().device
    print(f"\n✓ Using device: {device}")
    
    # Load model
    print("\n📥 Loading SAM3 Video model...")
    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    print("✓ Model loaded successfully!")
    
    # Load video
    print("\n🎥 Loading test video...")
    video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
    video_frames, _ = load_video(video_url)
    print(f"✓ Video loaded: {len(video_frames)} frames")
    
    # Limit frames for Pi 5
    max_frames = 50
    if len(video_frames) > max_frames:
        video_frames = video_frames[:max_frames]
        print(f"  Limited to {max_frames} frames for testing")
    
    # Initialize video inference session with all frames
    print("\n🔧 Initializing video inference session...")
    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )
    print("✓ Session initialized with all video frames")
    
    # Add text prompt
    text_prompt = "person"
    print(f"\n🎯 Adding text prompt: '{text_prompt}'")
    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=text_prompt,
    )
    
    # Process all frames
    print(f"\n🎬 Processing all {len(video_frames)} frames...")
    print("   (Using pre-loaded video mode)")
    
    outputs_per_frame = {}
    start_time = time.time()
    frame_count = 0
    
    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session, 
        max_frame_num_to_track=len(video_frames)
    ):
        processed_outputs = processor.postprocess_outputs(
            inference_session, 
            model_outputs
        )
        outputs_per_frame[model_outputs.frame_idx] = processed_outputs
        frame_count += 1
        
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"  Processed {frame_count} frames... ({fps:.2f} FPS)")
    
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time
    
    print(f"\n✓ Processing complete!")
    print(f"  Processed {frame_count} frames")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average FPS: {avg_fps:.2f}")
    
    # Analyze results
    print("\n📊 Results Analysis:")
    frame_0_outputs = outputs_per_frame[0]
    print(f"  Detected {len(frame_0_outputs['object_ids'])} objects in first frame")
    print(f"  Object IDs: {frame_0_outputs['object_ids'].tolist()}")
    print(f"  Scores: {[f'{s:.3f}' for s in frame_0_outputs['scores'].tolist()]}")
    print(f"  Boxes shape (XYXY format): {frame_0_outputs['boxes'].shape}")
    print(f"  Masks shape: {frame_0_outputs['masks'].shape}")
    
    # Object tracking over time
    print("\n🔍 Object Tracking Statistics:")
    object_counts = [len(outputs_per_frame[i]['object_ids']) for i in range(frame_count)]
    print(f"  Min objects: {min(object_counts)}")
    print(f"  Max objects: {max(object_counts)}")
    print(f"  Avg objects: {sum(object_counts)/len(object_counts):.1f}")
    
    # Track unique objects
    all_object_ids = set()
    for i in range(frame_count):
        all_object_ids.update(outputs_per_frame[i]['object_ids'].tolist())
    print(f"  Total unique objects: {len(all_object_ids)}")
    
    # Save summary
    print("\n💾 Saving results summary...")
    with open("video_processing_summary.txt", "w") as f:
        f.write("SAM3 Full Video Processing Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Device: {device}\n")
        f.write(f"Text Prompt: '{text_prompt}'\n")
        f.write(f"Total Frames: {frame_count}\n")
        f.write(f"Processing Time: {total_time:.2f}s\n")
        f.write(f"Average FPS: {avg_fps:.2f}\n\n")
        f.write("Frame-by-frame results:\n")
        f.write("-" * 50 + "\n")
        for i in range(frame_count):
            output = outputs_per_frame[i]
            f.write(f"Frame {i:3d}: {len(output['object_ids'])} objects - IDs: {output['object_ids'].tolist()}\n")
    
    print("✓ Saved: video_processing_summary.txt")
    
    print("\n" + "=" * 70)
    print("✅ Full video processing test completed!")
    print("=" * 70)
    print("\n💡 Notes:")
    print("  - This mode is best for offline processing")
    print("  - Requires loading all frames into memory")
    print("  - For real-time use, prefer streaming mode")

if __name__ == "__main__":
    main()

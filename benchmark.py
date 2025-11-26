"""
Quick benchmark script to test SAM3 performance on Raspberry Pi 5
"""

import torch
from PIL import Image
import requests
import time
import psutil
import os

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

def main():
    print("=" * 70)
    print("SAM3 Quick Benchmark - Raspberry Pi 5")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n💻 Device: {device}")
    print(f"🧠 Initial Memory: {get_memory_usage():.1f} MB")
    
    # Load model
    print("\n📥 Loading model...")
    start_time = time.time()
    
    from transformers import Sam3Model, Sam3Processor
    
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    
    load_time = time.time() - start_time
    mem_after_load = get_memory_usage()
    
    print(f"✓ Model loaded in {load_time:.2f}s")
    print(f"🧠 Memory after load: {mem_after_load:.1f} MB (+{mem_after_load - get_memory_usage():.1f} MB)")
    
    # Load test image
    print("\n🖼️  Loading test image...")
    image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    
    # Warmup run
    print("\n🔥 Warmup run...")
    inputs = processor(images=image, text="cat", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**inputs)
    
    # Benchmark runs
    print("\n⏱️  Benchmarking (5 runs)...")
    times = []
    
    for i in range(5):
        inputs = processor(images=image, text="cat", return_tensors="pt").to(device)
        
        start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        elapsed = time.time() - start
        
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s")
    
    # Results
    import numpy as np
    print("\n📊 Results:")
    print(f"  Mean: {np.mean(times):.3f}s")
    print(f"  Std:  {np.std(times):.3f}s")
    print(f"  Min:  {np.min(times):.3f}s")
    print(f"  Max:  {np.max(times):.3f}s")
    print(f"  FPS:  {1.0/np.mean(times):.2f}")
    
    print(f"\n🧠 Final Memory: {get_memory_usage():.1f} MB")
    
    # Process results
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    
    print(f"\n✓ Detected {len(results['masks'])} objects")
    
    print("\n" + "=" * 70)
    print("✅ Benchmark complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()

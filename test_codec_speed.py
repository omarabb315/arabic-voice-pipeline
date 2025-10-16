"""
Test VQ codec encoding speed on current machine.

This will help determine if T4 is fast enough for encoding
or if we should use H200's more powerful GPU.
"""

import torch
import numpy as np
import time
from pathlib import Path
from neucodec import NeuCodec
from tqdm import tqdm

def test_codec_speed(
    num_samples=100,
    sample_dir="segments_cache/ZadTVchannel/",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Test VQ encoding speed on sample segments.
    
    Args:
        num_samples: Number of segments to test
        sample_dir: Directory with sample segments
        device: Device to use (cuda or cpu)
    """
    
    print("=" * 70)
    print("VQ Codec Encoding Speed Test")
    print("=" * 70)
    print()
    
    # Check device
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # Load codec
    print("Loading NeuCodec model...")
    start = time.time()
    codec = NeuCodec.from_pretrained("neuphonic/neucodec")
    codec.eval().to(device)
    load_time = time.time() - start
    print(f"✅ Model loaded in {load_time:.2f} seconds")
    print()
    
    # Find sample files
    sample_path = Path(sample_dir)
    if not sample_path.exists():
        print(f"❌ Directory not found: {sample_dir}")
        return
    
    segment_files = list(sample_path.glob("*.npz"))[:num_samples]
    
    if len(segment_files) == 0:
        print(f"❌ No .npz files found in {sample_dir}")
        return
    
    print(f"Testing with {len(segment_files)} segments from {sample_dir}")
    print()
    
    # Test encoding
    encoding_times = []
    total_duration = 0
    
    print("Encoding segments...")
    for seg_file in tqdm(segment_files, desc="Processing"):
        try:
            # Load segment
            data = np.load(seg_file, allow_pickle=True)
            
            # Skip if already has codes
            if 'codes' in data and len(data['codes']) > 0:
                # Check if it's a tensor or array
                try:
                    codes_check = data['codes']
                    if hasattr(codes_check, 'is_cuda') or not isinstance(codes_check, (list, np.ndarray)):
                        continue  # Already encoded
                except:
                    continue
            
            audio_array = data['audio']
            duration = float(data['duration_sec'])
            total_duration += duration
            
            # Encode
            start = time.time()
            with torch.no_grad():
                # Don't move to device - encode_code handles it internally
                audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0).unsqueeze(0)
                vq_codes = codec.encode_code(audio_or_path=audio_tensor)
                if device == "cuda":
                    torch.cuda.synchronize()  # Wait for GPU to finish
            
            encoding_time = time.time() - start
            encoding_times.append(encoding_time)
            
        except Exception as e:
            print(f"\n⚠️ Error processing {seg_file.name}: {e}")
            continue
    
    # Calculate statistics
    if len(encoding_times) == 0:
        print("❌ No segments successfully encoded")
        return
    
    total_time = sum(encoding_times)
    avg_time = total_time / len(encoding_times)
    min_time = min(encoding_times)
    max_time = max(encoding_times)
    
    # Calculate throughput
    segments_per_second = len(encoding_times) / total_time
    audio_hours_per_hour = (total_duration / 3600) / (total_time / 3600)
    
    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print()
    print(f"Segments processed: {len(encoding_times)}")
    print(f"Total audio duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"Total encoding time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print()
    print(f"Average time per segment: {avg_time:.3f} seconds")
    print(f"Min time: {min_time:.3f} seconds")
    print(f"Max time: {max_time:.3f} seconds")
    print()
    print(f"Throughput: {segments_per_second:.2f} segments/second")
    print(f"Real-time factor: {audio_hours_per_hour:.2f}x")
    print(f"  (Process {audio_hours_per_hour:.2f} hours of audio per 1 hour of compute)")
    print()
    
    # Extrapolate to full dataset
    total_segments = 656300
    estimated_time_seconds = total_segments / segments_per_second
    estimated_hours = estimated_time_seconds / 3600
    
    print("=" * 70)
    print("Extrapolation to Full Dataset (656,300 segments)")
    print("=" * 70)
    print()
    print(f"Estimated time: {estimated_hours:.2f} hours ({estimated_hours/24:.2f} days)")
    print()
    
    if estimated_hours < 4:
        print("✅ FAST! T4 can encode the full dataset in under 4 hours.")
        print("   Recommendation: Encode on T4 directly, no need for H200 transfer.")
    elif estimated_hours < 12:
        print("⚡ MODERATE! T4 can encode in under 12 hours.")
        print("   Recommendation: T4 encoding is viable, but H200 might be faster.")
    else:
        print("🐌 SLOW! T4 would take over 12 hours.")
        print("   Recommendation: Use H200's more powerful GPU for encoding.")
    
    print()
    print("=" * 70)
    
    # GPU memory stats
    if device == "cuda":
        print()
        print("GPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    import sys
    
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    sample_dir = sys.argv[2] if len(sys.argv) > 2 else "segments_cache/ZadTVchannel/"
    
    print(f"Usage: python test_codec_speed.py [num_samples] [sample_dir]")
    print()
    
    test_codec_speed(num_samples, sample_dir)


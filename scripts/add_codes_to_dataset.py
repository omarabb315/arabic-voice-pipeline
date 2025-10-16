"""
Stage 2: Add VQ codes to segments (Run on H200 machine)

Takes .npz segments from Stage 1 (audio + metadata, no codes)
and adds VQ codec codes using NeuCodec.

Works directly with .npz files - fast and memory-efficient!
"""

import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from neucodec import NeuCodec
from fire import Fire


def add_codes_to_segments(
    segments_cache_dir: str = "segments_cache/",
    codec_device: str = "cuda" if torch.cuda.is_available() else "cpu",
    channels: list[str] = None,
):
    """
    Add VQ codes to .npz segment files.
    
    Args:
        segments_cache_dir: Directory with .npz segment files
        codec_device: Device for codec ("cuda" or "cpu")
        channels: List of channels to process (None = all)
    """
    
    print("=" * 70)
    print("Stage 2: Adding VQ Codes to Segments (H200)")
    print("=" * 70)
    print()
    
    # Load codec model
    print(f"Loading NeuCodec on {codec_device}...")
    codec = NeuCodec.from_pretrained("neuphonic/neucodec")
    codec.eval().to(codec_device)
    print(f"✅ Codec loaded on {codec_device}")
    print()
    
    # Find all segment files
    total_processed = 0
    total_skipped = 0
    
    cache_path = Path(segments_cache_dir)
    channel_dirs = [d for d in cache_path.iterdir() if d.is_dir()]
    
    # Filter by specified channels if provided
    if channels:
        channel_dirs = [d for d in channel_dirs if d.name in channels]
    
    for channel_dir in channel_dirs:
        print(f"\nProcessing channel: {channel_dir.name}")
        print("-" * 60)
        
        segment_files = list(channel_dir.glob("*.npz"))
        print(f"Found {len(segment_files)} segment files")
        
        for seg_file in tqdm(segment_files, desc=f"Encoding {channel_dir.name}"):
            try:
                # Load segment
                data = np.load(seg_file, allow_pickle=True)
                
                # Check if codes already exist
                if 'codes' in data and len(data['codes']) > 0:
                    total_skipped += 1
                    continue
                
                # Get audio
                audio_array = data['audio']
                
                # Encode to VQ codes
                with torch.no_grad():
                    audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0).unsqueeze(0)
                    vq_codes = codec.encode_code(audio_or_path=audio_tensor)
                    vq_codes = vq_codes.squeeze(0).squeeze(0)
                    if vq_codes.is_cuda:
                        vq_codes = vq_codes.cpu()
                    vq_codes_list = vq_codes.numpy()
                
                # Update segment file with codes
                segment_data = {
                    'audio': data['audio'],
                    'codes': vq_codes_list,
                    'text': data['text'],
                    'speaker_id': data['speaker_id'],
                    'channel': data['channel'],
                    'gender': data['gender'],
                    'dialect': data['dialect'],
                    'tone': data['tone'],
                    'duration_sec': data['duration_sec'],
                    'audio_id': data['audio_id'],
                }
                
                # Save updated segment
                np.savez_compressed(seg_file, **segment_data)
                total_processed += 1
                
                # Clear GPU cache periodically
                if codec_device == "cuda" and total_processed % 100 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\n⚠️ Error processing {seg_file.name}: {e}")
                continue
        
        print(f"✅ {channel_dir.name}: {total_processed} segments encoded, {total_skipped} already had codes")
    
    print()
    print("=" * 70)
    print("✅ Stage 2 Complete!")
    print("=" * 70)
    print(f"Total segments encoded: {total_processed}")
    print(f"Total segments skipped: {total_skipped}")
    print()
    print("Next: Create final HF dataset with codes")
    print(f"  python build_dataset_robust.py --create_hf_dataset=True")


if __name__ == "__main__":
    Fire(add_codes_to_segments)

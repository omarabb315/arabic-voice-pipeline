"""
Stage 2: Add VQ codes to segments - S3 Batch Processing Version

Designed for H200 with limited storage (33GB available).
Downloads batches from S3, processes, uploads back, and cleans up.

Supports resume capability - tracks processed files to restart after interruptions.
"""

import os
import shutil
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from neucodec import NeuCodec
from fire import Fire
from s3_utils import S3Client
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_codes_to_segments_s3(
    s3_prefix: str = "neutts_segments/stage1/",
    output_prefix: str = "neutts_segments/stage2_with_codes/",
    batch_size: int = 70000,  # ~24GB per batch (33GB available, 72% usage)
    temp_dir: str = "/tmp/neutts_batch/",
    codec_device: str = "cuda" if torch.cuda.is_available() else "cpu",
    resume_file: str = "processing_progress.json",
):
    """
    Add VQ codes to .npz segments using batch processing from S3.
    
    Args:
        s3_prefix: S3 prefix where stage1 segments are stored
        output_prefix: S3 prefix to save encoded segments
        batch_size: Number of segments per batch (70K = ~24GB)
        temp_dir: Local directory for temporary batch storage
        codec_device: Device for codec ("cuda" or "cpu")
        resume_file: JSON file to track progress for resuming
    """
    
    print("=" * 70)
    print("Stage 2: Adding VQ Codes to Segments (S3 Batch Processing)")
    print("=" * 70)
    print(f"S3 Input: {s3_prefix}")
    print(f"S3 Output: {output_prefix}")
    print(f"Batch size: {batch_size:,} segments (~{batch_size * 0.35 / 1024:.1f} GB per batch)")
    print(f"Temp dir: {temp_dir}")
    print(f"Device: {codec_device}")
    print()
    
    # Initialize S3 client
    s3_client = S3Client()
    
    # Load NeuCodec
    print(f"Loading NeuCodec on {codec_device}...")
    codec = NeuCodec.from_pretrained("neuphonic/neucodec")
    codec.eval().to(codec_device)
    print(f"✅ Codec loaded on {codec_device}")
    print()
    
    # List all segment files in S3
    print("Listing segment files in S3...")
    all_segment_files = s3_client.list_files(prefix=s3_prefix, suffix=".npz")
    print(f"Found {len(all_segment_files):,} segment files in S3")
    print()
    
    if len(all_segment_files) == 0:
        logger.error(f"No .npz files found in S3 prefix: {s3_prefix}")
        return
    
    # Load progress tracking
    processed_files = set()
    if os.path.exists(resume_file):
        try:
            with open(resume_file, 'r') as f:
                progress_data = json.load(f)
                processed_files = set(progress_data.get('processed', []))
            print(f"📋 Resuming: {len(processed_files):,} files already processed")
        except Exception as e:
            logger.warning(f"Could not load resume file: {e}")
    
    # Filter out already processed files
    files_to_process = [f for f in all_segment_files if f not in processed_files]
    print(f"Files to process: {len(files_to_process):,}")
    print(f"Already processed: {len(processed_files):,}")
    print()
    
    if len(files_to_process) == 0:
        print("✅ All files already processed!")
        return
    
    # Create temp directory
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process in batches
    total_processed = 0
    total_skipped = 0
    num_batches = (len(files_to_process) + batch_size - 1) // batch_size
    
    print(f"Processing {num_batches} batches...")
    print()
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(files_to_process))
        batch_files = files_to_process[batch_start:batch_end]
        
        print("=" * 70)
        print(f"Batch {batch_idx + 1}/{num_batches}")
        print(f"Files: {len(batch_files):,} ({batch_start:,} to {batch_end:,})")
        print("=" * 70)
        
        # Download batch
        print(f"\n📥 Downloading batch to {temp_dir}...")
        batch_local_paths = {}
        
        for s3_key in tqdm(batch_files, desc="Downloading"):
            # Preserve directory structure
            relative_path = s3_key.replace(s3_prefix, "")
            local_path = os.path.join(temp_dir, relative_path)
            
            # Download
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if s3_client.download_file(s3_key, local_path, show_progress=False):
                batch_local_paths[s3_key] = local_path
            else:
                logger.error(f"Failed to download: {s3_key}")
        
        print(f"✅ Downloaded {len(batch_local_paths):,} files")
        
        # Process batch
        print(f"\n🔧 Processing batch (adding VQ codes)...")
        batch_processed = 0
        batch_skipped = 0
        
        for s3_key, local_path in tqdm(list(batch_local_paths.items()), desc="Encoding"):
            try:
                # Load segment
                data = np.load(local_path, allow_pickle=True)
                
                # Check if codes already exist
                if 'codes' in data and len(data['codes']) > 0:
                    batch_skipped += 1
                    continue
                
                # Get audio
                audio_array = data['audio']
                
                # Encode to VQ codes
                with torch.no_grad():
                    audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0).unsqueeze(0)
                    audio_tensor = audio_tensor.to(codec_device)
                    vq_codes = codec.encode_code(audio_or_path=audio_tensor)
                    vq_codes = vq_codes.squeeze(0).squeeze(0)
                    if vq_codes.is_cuda:
                        vq_codes = vq_codes.cpu()
                    vq_codes_list = vq_codes.numpy()
                
                # Update segment with codes
                segment_data = {
                    'audio': data['audio'],
                    'codes': vq_codes_list,
                    'text': str(data['text']),
                    'speaker_id': str(data['speaker_id']),
                    'channel': str(data['channel']),
                    'gender': str(data['gender']),
                    'dialect': str(data['dialect']),
                    'tone': str(data['tone']),
                    'duration_sec': float(data['duration_sec']),
                    'audio_id': str(data['audio_id']),
                }
                
                # Save updated segment
                np.savez_compressed(local_path, **segment_data)
                batch_processed += 1
                
                # Clear GPU cache periodically
                if codec_device == "cuda" and batch_processed % 100 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error processing {local_path}: {e}")
                continue
        
        print(f"✅ Processed {batch_processed:,} segments, skipped {batch_skipped:,}")
        
        # Upload batch
        print(f"\n📤 Uploading encoded batch to S3...")
        batch_uploaded = 0
        
        for s3_key, local_path in tqdm(list(batch_local_paths.items()), desc="Uploading"):
            # Convert input path to output path
            output_s3_key = s3_key.replace(s3_prefix, output_prefix)
            
            if s3_client.upload_file(local_path, output_s3_key, show_progress=False):
                batch_uploaded += 1
                processed_files.add(s3_key)
            else:
                logger.error(f"Failed to upload: {output_s3_key}")
        
        print(f"✅ Uploaded {batch_uploaded:,} files")
        
        # Save progress
        with open(resume_file, 'w') as f:
            json.dump({'processed': list(processed_files)}, f)
        
        # Clean up batch
        print(f"\n🧹 Cleaning up batch...")
        try:
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Error cleaning temp dir: {e}")
        
        # Clear GPU memory
        if codec_device == "cuda":
            torch.cuda.empty_cache()
        
        total_processed += batch_processed
        total_skipped += batch_skipped
        
        print(f"\nBatch {batch_idx + 1} complete!")
        print(f"Running total: {total_processed:,} processed, {total_skipped:,} skipped")
        print()
    
    print()
    print("=" * 70)
    print("✅ Stage 2 Complete!")
    print("=" * 70)
    print(f"Total segments encoded: {total_processed:,}")
    print(f"Total segments skipped: {total_skipped:,}")
    print(f"Total files processed: {len(processed_files):,}")
    print()
    print("Next: Create final HF dataset")
    print(f"  python scripts/create_hf_dataset_from_s3.py \\")
    print(f"    --s3_prefix={output_prefix} \\")
    print(f"    --push_to_hub=True")
    print()
    
    # Clean up
    try:
        os.remove(resume_file)
        shutil.rmtree(temp_dir, ignore_errors=True)
    except:
        pass


if __name__ == "__main__":
    Fire(add_codes_to_segments_s3)


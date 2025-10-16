"""
Stage 2: Add VQ codes to segments - GCS Batch Processing Version

Designed for H200 with limited storage (33GB available, 98GB persistent).
Downloads batches from GCS, processes, uploads back, and cleans up.

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
from gcs_utils import GCSClient
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_codes_to_segments_gcs(
    gcs_bucket: str = "audio-data-gemini",
    input_prefix: str = "segments_cache/",
    output_prefix: str = "segments_with_codes/",
    batch_size: int = 70000,  # ~24GB per batch (33GB available, 72% usage)
    temp_dir: str = "/tmp/neutts_batch/",
    codec_device: str = "cuda" if torch.cuda.is_available() else "cpu",
    resume_file: str = "processing_progress.json",
):
    """
    Add VQ codes to .npz segments using batch processing from GCS.
    
    Args:
        gcs_bucket: GCS bucket name
        input_prefix: GCS prefix where stage1 segments are stored
        output_prefix: GCS prefix to save encoded segments
        batch_size: Number of segments per batch (70K = ~24GB)
        temp_dir: Local directory for temporary batch storage
        codec_device: Device for codec ("cuda" or "cpu")
        resume_file: JSON file to track progress for resuming
    """
    
    print("=" * 70)
    print("Stage 2: Adding VQ Codes to Segments (GCS Batch Processing)")
    print("=" * 70)
    print(f"GCS Bucket: {gcs_bucket}")
    print(f"Input prefix: {input_prefix}")
    print(f"Output prefix: {output_prefix}")
    print(f"Batch size: {batch_size:,} segments (~{batch_size * 0.35 / 1024:.1f} GB per batch)")
    print(f"Temp dir: {temp_dir}")
    print(f"Device: {codec_device}")
    print()
    
    # Initialize GCS client
    gcs_client = GCSClient(bucket_name=gcs_bucket)
    
    # Load NeuCodec
    print(f"Loading NeuCodec on {codec_device}...")
    codec = NeuCodec.from_pretrained("neuphonic/neucodec")
    codec.eval().to(codec_device)
    print(f"✅ Codec loaded on {codec_device}")
    print()
    
    # List all segment files in GCS
    print("Listing segment files in GCS...")
    all_segment_files = gcs_client.list_files(prefix=input_prefix, suffix=".npz")
    print(f"Found {len(all_segment_files):,} segment files in GCS")
    print()
    
    if len(all_segment_files) == 0:
        logger.error(f"No .npz files found in GCS prefix: {input_prefix}")
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
        
        for gcs_path in tqdm(batch_files, desc="Downloading"):
            # Preserve directory structure
            relative_path = gcs_path.replace(input_prefix, "")
            local_path = os.path.join(temp_dir, relative_path)
            
            # Download
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if gcs_client.download_file(gcs_path, local_path, show_progress=False):
                batch_local_paths[gcs_path] = local_path
            else:
                logger.error(f"Failed to download: {gcs_path}")
        
        print(f"✅ Downloaded {len(batch_local_paths):,} files")
        
        # Process batch
        print(f"\n🔧 Processing batch (adding VQ codes)...")
        batch_processed = 0
        batch_skipped = 0
        
        for gcs_path, local_path in tqdm(list(batch_local_paths.items()), desc="Encoding"):
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
                    # Don't move to device - encode_code handles it internally
                    audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0).unsqueeze(0)
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
        print(f"\n📤 Uploading encoded batch to GCS...")
        batch_uploaded = 0
        
        for gcs_path, local_path in tqdm(list(batch_local_paths.items()), desc="Uploading"):
            # Convert input path to output path
            output_gcs_path = gcs_path.replace(input_prefix, output_prefix)
            
            if gcs_client.upload_file(local_path, output_gcs_path, show_progress=False):
                batch_uploaded += 1
                processed_files.add(gcs_path)
            else:
                logger.error(f"Failed to upload: {output_gcs_path}")
        
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
    print(f"  python create_hf_dataset_from_npz.py \\")
    print(f"    --gcs_bucket={gcs_bucket} \\")
    print(f"    --gcs_prefix={output_prefix} \\")
    print(f"    --push_to_hub=True")
    print()
    
    # Clean up
    try:
        os.remove(resume_file)
        shutil.rmtree(temp_dir, ignore_errors=True)
    except:
        pass


if __name__ == "__main__":
    Fire(add_codes_to_segments_gcs)


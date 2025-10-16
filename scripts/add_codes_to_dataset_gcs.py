"""
Stage 2: Add VQ codes to segments - GCS RAM-Optimized Version

Designed for H200 with 2TB RAM - processes all segments at once.
Uses /tmp with OS RAM caching for optimal performance.

Supports resume capability with three-phase checkpointing:
- Phase 1: Download all segments from GCS
- Phase 2: Process all segments (add VQ codes)
- Phase 3: Upload all processed segments to GCS
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
from ram_disk_utils import get_optimal_temp_config, setup_temp_directory
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_codes_to_segments_gcs(
    gcs_bucket: str = "audio-data-gemini",
    input_prefix: str = "segments_cache/",
    output_prefix: str = "segments_with_codes/",
    temp_dir: str = None,  # Auto-detected if None
    codec_device: str = "cuda" if torch.cuda.is_available() else "cpu",
    resume_file: str = "processing_progress.json",
    checkpoint_interval: int = 5000,  # Save progress every N files
    auto_detect_ram: bool = True,  # Automatically detect and validate RAM
):
    """
    Add VQ codes to .npz segments - processes all segments at once using available RAM.
    
    Three-phase process with resume capability:
    1. Download all segments from GCS
    2. Process all segments (add VQ codes)
    3. Upload all processed segments to GCS
    
    Args:
        gcs_bucket: GCS bucket name
        input_prefix: GCS prefix where stage1 segments are stored
        output_prefix: GCS prefix to save encoded segments
        temp_dir: Local directory for temporary storage (auto-detected if None)
        codec_device: Device for codec ("cuda" or "cpu")
        resume_file: JSON file to track progress for resuming
        checkpoint_interval: Save progress every N files (default: 5000)
        auto_detect_ram: Automatically detect optimal temp location (default: True)
    """
    
    print("=" * 70)
    print("Stage 2: Adding VQ Codes to Segments (GCS RAM-Optimized)")
    print("=" * 70)
    print(f"GCS Bucket: {gcs_bucket}")
    print(f"Input prefix: {input_prefix}")
    print(f"Output prefix: {output_prefix}")
    print(f"Device: {codec_device}")
    print(f"Checkpoint interval: {checkpoint_interval:,} files")
    print()
    
    # Auto-detect optimal temp location if not specified
    if auto_detect_ram and temp_dir is None:
        ram_config = get_optimal_temp_config(
            required_space_gb=300.0,
            data_size_gb=221.0
        )
        temp_dir = ram_config['temp_dir']
        print(f"Auto-detected temp directory: {temp_dir}")
        print(f"Storage type: {ram_config['storage_type']}")
        print(f"Available RAM: {ram_config['available_ram_gb']:.1f} GB")
        print()
    elif temp_dir is None:
        temp_dir = "/tmp/neutts_processing"
        print(f"Using default temp directory: {temp_dir}")
        print()
    
    # Initialize GCS client
    print("Initializing GCS client...")
    gcs_client = GCSClient(bucket_name=gcs_bucket)
    print("✅ GCS client initialized")
    print()
    
    # List all segment files in GCS
    print("Listing segment files in GCS...")
    all_segment_files = gcs_client.list_files(prefix=input_prefix, suffix=".npz")
    total_files = len(all_segment_files)
    print(f"Found {total_files:,} segment files in GCS")
    
    if total_files == 0:
        logger.error(f"No .npz files found in GCS prefix: {input_prefix}")
        return
    
    # Estimate data size
    estimated_size_gb = total_files * 0.35 / 1024  # ~350KB per file
    print(f"Estimated data size: {estimated_size_gb:.1f} GB")
    print()
    
    # Load or initialize progress tracking
    progress = {
        'total_files': total_files,
        'phase': 'initializing',
        'downloaded': 0,
        'processed': 0,
        'uploaded': 0,
        'downloaded_files': set(),
        'processed_files': set(),
        'uploaded_files': set(),
        'start_time': datetime.now().isoformat(),
        'last_checkpoint': datetime.now().isoformat()
    }
    
    if os.path.exists(resume_file):
        try:
            with open(resume_file, 'r') as f:
                saved_progress = json.load(f)
                # Convert lists back to sets
                progress['downloaded_files'] = set(saved_progress.get('downloaded_files', []))
                progress['processed_files'] = set(saved_progress.get('processed_files', []))
                progress['uploaded_files'] = set(saved_progress.get('uploaded_files', []))
                progress['downloaded'] = len(progress['downloaded_files'])
                progress['processed'] = len(progress['processed_files'])
                progress['uploaded'] = len(progress['uploaded_files'])
                progress['phase'] = saved_progress.get('phase', 'initializing')
                progress['start_time'] = saved_progress.get('start_time', progress['start_time'])
            
            print("📋 Resuming from previous run:")
            print(f"   Phase: {progress['phase']}")
            print(f"   Downloaded: {progress['downloaded']:,} / {total_files:,}")
            print(f"   Processed: {progress['processed']:,} / {total_files:,}")
            print(f"   Uploaded: {progress['uploaded']:,} / {total_files:,}")
            print()
        except Exception as e:
            logger.warning(f"Could not load resume file, starting fresh: {e}")
            print()
    
    # Check if already complete
    if progress['uploaded'] >= total_files:
        print("✅ All files already processed and uploaded!")
        print()
        return
    
    # Setup temp directory
    temp_dir = setup_temp_directory(
        base_path=os.path.dirname(temp_dir) if os.path.dirname(temp_dir) else '/tmp',
        subdir=os.path.basename(temp_dir),
        clean_if_exists=False
    )
    print()
    
    def save_progress():
        """Save progress to JSON file"""
        with open(resume_file, 'w') as f:
            # Convert sets to lists for JSON serialization
            save_data = progress.copy()
            save_data['downloaded_files'] = list(progress['downloaded_files'])
            save_data['processed_files'] = list(progress['processed_files'])
            save_data['uploaded_files'] = list(progress['uploaded_files'])
            save_data['last_checkpoint'] = datetime.now().isoformat()
            json.dump(save_data, f, indent=2)
    
    # =========================================================================
    # PHASE 1: Download All Segments
    # =========================================================================
    
    if progress['downloaded'] < total_files:
        print("=" * 70)
        print("PHASE 1: Downloading All Segments from GCS")
        print("=" * 70)
        print(f"Target: {total_files:,} files (~{estimated_size_gb:.1f} GB)")
        print(f"Already downloaded: {progress['downloaded']:,} files")
        print()
        
        progress['phase'] = 'downloading'
        save_progress()
        
        files_to_download = [f for f in all_segment_files if f not in progress['downloaded_files']]
        print(f"Downloading {len(files_to_download):,} remaining files...")
        print()
        
        downloaded_count = 0
        for gcs_path in tqdm(files_to_download, desc="Downloading"):
            try:
                # Preserve directory structure
                relative_path = gcs_path.replace(input_prefix, "")
                local_path = os.path.join(temp_dir, relative_path)
                
                # Download
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                if gcs_client.download_file(gcs_path, local_path, show_progress=False):
                    progress['downloaded_files'].add(gcs_path)
                    downloaded_count += 1
                    
                    # Checkpoint periodically
                    if downloaded_count % checkpoint_interval == 0:
                        progress['downloaded'] = len(progress['downloaded_files'])
                        save_progress()
                else:
                    logger.error(f"Failed to download: {gcs_path}")
            
            except Exception as e:
                logger.error(f"Error downloading {gcs_path}: {e}")
                continue
        
        progress['downloaded'] = len(progress['downloaded_files'])
        save_progress()
        
        print()
        print(f"✅ Download complete: {progress['downloaded']:,} files")
        print()
    else:
        print("✅ Phase 1: All files already downloaded")
        print()
    
    # =========================================================================
    # PHASE 2: Process All Segments (Add VQ Codes)
    # =========================================================================
    
    if progress['processed'] < total_files:
        print("=" * 70)
        print("PHASE 2: Processing All Segments (Adding VQ Codes)")
        print("=" * 70)
        print(f"Target: {total_files:,} files")
        print(f"Already processed: {progress['processed']:,} files")
        print()
        
        progress['phase'] = 'processing'
        save_progress()
        
        # Load NeuCodec
        print(f"Loading NeuCodec on {codec_device}...")
        codec = NeuCodec.from_pretrained("neuphonic/neucodec")
        codec.eval().to(codec_device)
        print(f"✅ Codec loaded on {codec_device}")
        print()
        
        # Get all local files
        files_to_process = []
        for gcs_path in all_segment_files:
            if gcs_path not in progress['processed_files']:
                relative_path = gcs_path.replace(input_prefix, "")
                local_path = os.path.join(temp_dir, relative_path)
                if os.path.exists(local_path):
                    files_to_process.append((gcs_path, local_path))
        
        print(f"Processing {len(files_to_process):,} remaining files...")
        print()
        
        processed_count = 0
        skipped_count = 0
        
        for gcs_path, local_path in tqdm(files_to_process, desc="Encoding"):
            try:
                # Load segment
                data = np.load(local_path, allow_pickle=True)
                
                # Check if codes already exist
                if 'codes' in data and len(data['codes']) > 0:
                    skipped_count += 1
                    progress['processed_files'].add(gcs_path)
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
                progress['processed_files'].add(gcs_path)
                processed_count += 1
                
                # Clear GPU cache periodically
                if codec_device == "cuda" and processed_count % 100 == 0:
                    torch.cuda.empty_cache()
                
                # Checkpoint periodically
                if processed_count % checkpoint_interval == 0:
                    progress['processed'] = len(progress['processed_files'])
                    save_progress()
                    logger.info(f"Checkpoint: {processed_count:,} files processed")
            
            except Exception as e:
                logger.error(f"Error processing {local_path}: {e}")
                continue
        
        progress['processed'] = len(progress['processed_files'])
        save_progress()
        
        print()
        print(f"✅ Processing complete: {processed_count:,} encoded, {skipped_count:,} skipped")
        print()
        
        # Clear GPU memory
        if codec_device == "cuda":
            torch.cuda.empty_cache()
        del codec
    else:
        print("✅ Phase 2: All files already processed")
        print()
    
    # =========================================================================
    # PHASE 3: Upload All Processed Segments
    # =========================================================================
    
    if progress['uploaded'] < total_files:
        print("=" * 70)
        print("PHASE 3: Uploading All Processed Segments to GCS")
        print("=" * 70)
        print(f"Target: {total_files:,} files")
        print(f"Already uploaded: {progress['uploaded']:,} files")
        print()
        
        progress['phase'] = 'uploading'
        save_progress()
        
        # Get files to upload (processed but not yet uploaded)
        files_to_upload = []
        for gcs_path in progress['processed_files']:
            if gcs_path not in progress['uploaded_files']:
                relative_path = gcs_path.replace(input_prefix, "")
                local_path = os.path.join(temp_dir, relative_path)
                if os.path.exists(local_path):
                    files_to_upload.append((gcs_path, local_path))
        
        print(f"Uploading {len(files_to_upload):,} remaining files...")
        print()
        
        uploaded_count = 0
        for gcs_path, local_path in tqdm(files_to_upload, desc="Uploading"):
            try:
                # Convert input path to output path
                output_gcs_path = gcs_path.replace(input_prefix, output_prefix)
                
                if gcs_client.upload_file(local_path, output_gcs_path, show_progress=False):
                    progress['uploaded_files'].add(gcs_path)
                    uploaded_count += 1
                    
                    # Checkpoint periodically
                    if uploaded_count % checkpoint_interval == 0:
                        progress['uploaded'] = len(progress['uploaded_files'])
                        save_progress()
                else:
                    logger.error(f"Failed to upload: {output_gcs_path}")
            
            except Exception as e:
                logger.error(f"Error uploading {gcs_path}: {e}")
                continue
        
        progress['uploaded'] = len(progress['uploaded_files'])
        save_progress()
        
        print()
        print(f"✅ Upload complete: {uploaded_count:,} files")
        print()
    else:
        print("✅ Phase 3: All files already uploaded")
        print()
    
    # =========================================================================
    # Complete & Cleanup
    # =========================================================================
    
    progress['phase'] = 'complete'
    save_progress()
    
    print()
    print("=" * 70)
    print("✅ Stage 2 Complete!")
    print("=" * 70)
    print(f"Total files processed: {progress['processed']:,}")
    print(f"Total files uploaded: {progress['uploaded']:,}")
    print(f"Started: {progress['start_time']}")
    print(f"Completed: {datetime.now().isoformat()}")
    print()
    print("Next: Create final HF dataset")
    print(f"  python create_hf_dataset_from_npz.py \\")
    print(f"    --gcs_bucket={gcs_bucket} \\")
    print(f"    --gcs_prefix={output_prefix} \\")
    print(f"    --push_to_hub=True")
    print()
    
    # Cleanup
    print("🧹 Cleaning up temporary files...")
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"✅ Removed temp directory: {temp_dir}")
    except Exception as e:
        logger.warning(f"Could not remove temp dir: {e}")
    
    try:
        os.remove(resume_file)
        print(f"✅ Removed progress file: {resume_file}")
    except Exception as e:
        logger.warning(f"Could not remove progress file: {e}")
    
    print()
    print("All done! 🎉")


if __name__ == "__main__":
    Fire(add_codes_to_segments_gcs)


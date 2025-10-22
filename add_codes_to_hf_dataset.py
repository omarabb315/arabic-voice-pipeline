"""
Add VQ Codes to HuggingFace Dataset

This script downloads a HuggingFace dataset, adds VQ codec codes using NeuCodec,
and pushes the updated dataset back to HuggingFace Hub.

Designed for H200 machines with large RAM/VRAM for efficient batch processing.

Usage:
    # Basic usage (download, add codes, push back)
    python add_codes_to_hf_dataset.py \
        --hf_repo=omarabb315/mix-1000 \
        --codec_device=cuda \
        --push_to_hub=True

    # Test with small sample first
    python add_codes_to_hf_dataset.py \
        --hf_repo=omarabb315/mix-1000 \
        --codec_device=cuda \
        --max_samples=100 \
        --push_to_hub=False

    # Use larger batches for faster processing (if you have enough VRAM)
    python add_codes_to_hf_dataset.py \
        --hf_repo=omarabb315/mix-1000 \
        --codec_device=cuda \
        --batch_size=500 \
        --push_to_hub=True

    # Skip audio longer than 200 seconds (default)
    python add_codes_to_hf_dataset.py \
        --hf_repo=omarabb315/mix-1000 \
        --codec_device=cuda \
        --max_duration_sec=200 \
        --push_to_hub=True

    # Resume interrupted processing (automatically skips samples with codes)
    # If processing is interrupted, just run the same command again
    python add_codes_to_hf_dataset.py \
        --hf_repo=omarabb315/mix-1000 \
        --codec_device=cuda \
        --push_to_hub=True

    # Disable checkpoints (saves disk space)
    python add_codes_to_hf_dataset.py \
        --hf_repo=omarabb315/mix-1000 \
        --codec_device=cuda \
        --checkpoint_every=0 \
        --push_to_hub=True

    # Force reprocess samples that already have codes
    python add_codes_to_hf_dataset.py \
        --hf_repo=omarabb315/mix-1000 \
        --codec_device=cuda \
        --force_reprocess=True

Workflow:
    1. Download dataset from HuggingFace Hub (or load from checkpoint)
    2. Load NeuCodec model on GPU
    3. Process dataset in batches to add VQ codes
       - Automatically skips samples that already have codes (resume capability)
       - Skips audio longer than max_duration_sec to prevent memory issues
    4. Save checkpoint to disk (optional, for additional safety)
    5. Push updated dataset back to HuggingFace Hub

Resume Capability:
    - If interrupted, just run the same command again
    - Samples with existing codes are automatically skipped
    - Checkpoint saved after processing for additional safety
    - No data loss even if push to Hub fails

Requirements:
    - HuggingFace account with login (huggingface-cli login)
    - Sufficient RAM for dataset (auto-cached by HF)
    - GPU with enough VRAM for NeuCodec (40GB+ recommended)
    - Disk space for checkpoint (optional, ~same size as dataset)
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm
from datasets import load_dataset, Dataset, Audio, Features, Value, Sequence, concatenate_datasets
from neucodec import NeuCodec
from fire import Fire


def check_hf_auth():
    """Check if user is authenticated with HuggingFace."""
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        print("❌ Not authenticated with HuggingFace!")
        print("   Please run: huggingface-cli login")
        return False


def load_neucodec(device: str = "cuda"):
    """Load NeuCodec model on specified device."""
    print(f"Loading NeuCodec on {device}...")
    codec = NeuCodec.from_pretrained("neuphonic/neucodec")
    codec.eval().to(device)
    print(f"✅ NeuCodec loaded on {device}")
    return codec


def add_codes_batch(
    batch: Dict[str, Any],
    codec: NeuCodec,
    device: str,
    force_reprocess: bool = False,
    max_duration_sec: float = 200.0,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Add VQ codes to a batch of samples.
    
    Args:
        batch: Dictionary with 'audio' key containing audio arrays
        codec: NeuCodec model
        device: Device to run encoding on
        force_reprocess: If True, reprocess even if codes exist
        max_duration_sec: Maximum audio duration to process (longer samples are skipped)
        debug: Enable debug output
    
    Returns:
        Dictionary with 'codes' key containing VQ codes
    """
    # Extract audio arrays and existing codes (for resume functionality)
    audio_arrays = batch['audio']
    existing_codes = batch.get('codes', [None] * len(audio_arrays))
    batch_codes = []
    
    # Process each audio in the batch
    for i, audio_data in enumerate(audio_arrays):
        # Check if this sample already has codes (resume functionality)
        if not force_reprocess and existing_codes and i < len(existing_codes):
            existing_code = existing_codes[i]
            # If codes exist and are non-empty, keep them
            if existing_code is not None and len(existing_code) > 0:
                batch_codes.append(existing_code)
                continue
        
        # Simply access the 'array' key - HuggingFace Audio feature always uses this format
        try:
            audio_array = audio_data['array']
            sampling_rate = audio_data['sampling_rate']
        except (KeyError, TypeError) as e:
            # Fallback: if not dict-like, try as numpy array directly
            if isinstance(audio_data, np.ndarray):
                audio_array = audio_data
                sampling_rate = 16000
            else:
                if debug:
                    print(f"ERROR: Cannot extract audio array from sample {i}. Type: {type(audio_data)}, Error: {e}")
                batch_codes.append([])
                continue
        except RuntimeError as e:
            # Handle corrupted/empty audio files that can't be decoded
            if "No audio frames were decoded" in str(e):
                if debug:
                    print(f"⚠️ Skipping sample {i}: Corrupted or empty audio file")
                batch_codes.append([])
                continue
            else:
                # Re-raise if it's a different RuntimeError
                raise
        except Exception as e:
            # Catch any other audio decoding errors
            if debug:
                print(f"⚠️ Skipping sample {i}: Audio decoding error: {e}")
            batch_codes.append([])
            continue
        
        # Calculate duration and skip if too long (prevents memory issues)
        duration_sec = len(audio_array) / sampling_rate
        if duration_sec > max_duration_sec:
            if debug:
                print(f"⚠️ Skipping sample {i}: duration {duration_sec:.1f}s exceeds {max_duration_sec}s limit")
            batch_codes.append([])
            continue
        
        # Ensure proper dtype for torch (should already be float32, but double-check)
        if audio_array.dtype not in [np.float32, np.float64]:
            audio_array = audio_array.astype(np.float32)
        
        # Convert to tensor and encode to VQ codes
        with torch.no_grad():
            audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0).unsqueeze(0)
            
            # Encode to VQ codes
            vq_codes = codec.encode_code(audio_or_path=audio_tensor)
            vq_codes = vq_codes.squeeze(0).squeeze(0)
            
            if vq_codes.is_cuda:
                vq_codes = vq_codes.cpu()
            
            vq_codes_list = vq_codes.numpy().tolist()
            batch_codes.append(vq_codes_list)
            
            # Clean up tensors to prevent memory leak
            del audio_tensor, vq_codes
    
    # Add codes to batch
    batch['codes'] = batch_codes
    
    # Clear GPU cache after each batch to prevent memory buildup
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return batch


def process_dataset(
    hf_repo: str,
    codec_device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 100,
    max_samples: Optional[int] = None,
    max_duration_sec: float = 200.0,
    checkpoint_every: int = 10000,
    checkpoint_path: str = "checkpoint_dataset",
    force_reprocess: bool = False,
    push_to_hub: bool = True,
    split: str = "train",
):
    """
    Main processing function with resume capability.
    
    Args:
        hf_repo: HuggingFace repository name (e.g., 'username/dataset-name')
        codec_device: Device for NeuCodec ('cuda' or 'cpu')
        batch_size: Number of samples to process at once
        max_samples: Maximum number of samples to process (None = all)
        max_duration_sec: Maximum audio duration to process (longer samples are skipped)
        checkpoint_every: Save checkpoint every N samples (0 to disable)
        checkpoint_path: Path to save checkpoints for resume capability
        force_reprocess: Reprocess samples that already have codes
        push_to_hub: Push updated dataset back to HuggingFace Hub
        split: Dataset split to process ('train', 'test', etc.)
    
    Note:
        - Multiprocessing is disabled (num_proc=None) to avoid CUDA fork issues
        - Processing runs sequentially in batches for memory efficiency
        - Resume: If interrupted, rerun with same command - samples with codes are skipped
        - Checkpoints: Saved every checkpoint_every samples for additional safety
    """
    print("=" * 70)
    print("Add VQ Codes to HuggingFace Dataset")
    print("=" * 70)
    print()
    
    # Check authentication
    if push_to_hub and not check_hf_auth():
        print("⚠️ Warning: Not authenticated. Set push_to_hub=False or login first.")
        return
    
    print()
    print(f"Configuration:")
    print(f"  Repository: {hf_repo}")
    print(f"  Split: {split}")
    print(f"  Codec Device: {codec_device}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Max Samples: {max_samples if max_samples else 'All'}")
    print(f"  Max Duration: {max_duration_sec}s (longer samples will be skipped)")
    print(f"  Checkpoint Every: {checkpoint_every if checkpoint_every > 0 else 'Disabled'}")
    print(f"  Checkpoint Path: {checkpoint_path}")
    print(f"  Force Reprocess: {force_reprocess}")
    print(f"  Push to Hub: {push_to_hub}")
    print(f"  Multiprocessing: Disabled (CUDA compatibility)")
    print()
    
    # Check if checkpoint exists (for resume)
    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.exists() and not force_reprocess:
        print(f"📂 Found checkpoint at {checkpoint_path}")
        print(f"   Loading from checkpoint to resume processing...")
        try:
            dataset = Dataset.load_from_disk(str(checkpoint_file))
            print(f"✅ Checkpoint loaded: {len(dataset):,} samples")
            print(f"   Samples with existing codes will be skipped")
            print()
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            print(f"   Loading from HuggingFace Hub instead...")
            checkpoint_file = None
    else:
        checkpoint_file = None
    
    # Load dataset from Hub if no checkpoint
    if checkpoint_file is None:
        print(f"Loading dataset from {hf_repo}...")
        try:
            dataset = load_dataset(hf_repo, split=split)
            print(f"✅ Dataset loaded: {len(dataset):,} samples")
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            print("   Make sure the repository exists and you have access.")
            return
    
    # Limit samples if specified (for testing)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"📊 Limited to {len(dataset):,} samples for processing")
    
    print()
    print(f"Dataset info:")
    print(f"  Features: {list(dataset.features.keys())}")
    if 'audio' in dataset.features:
        print(f"  Audio sampling rate: {dataset.features['audio'].sampling_rate}")
    if 'codes' in dataset.features:
        print(f"  ⚠️ Dataset already has 'codes' column")
        if not force_reprocess:
            print(f"     Will skip samples that already have codes")
        else:
            print(f"     Will reprocess all samples (force_reprocess=True)")
    print()
    
    # Ensure audio is decoded (important for batched processing)
    if 'audio' in dataset.features:
        print("Ensuring audio decoding is enabled...")
        # Set format to enable decoding
        try:
            # First try casting
            dataset = dataset.cast_column('audio', Audio(sampling_rate=16000, decode=True))
            print("✅ Audio decoding enabled via cast")
        except Exception as e:
            print(f"⚠️ Cast failed: {e}, trying alternative method...")
            # Alternative: use with_format
            dataset.set_format(type=None)  # Reset format to decode audio
            print("✅ Audio format reset for decoding")
    print()
    
    # Load codec model
    codec = load_neucodec(codec_device)
    print()
    
    # Process dataset
    print(f"Processing dataset to add VQ codes...")
    print(f"This may take several hours for large datasets...")
    print()
    
    # Process with incremental checkpointing
    # NOTE: Must use num_proc=None (no multiprocessing) to avoid CUDA fork issues
    print("⚠️ Debug mode enabled - will show when samples are skipped")
    if checkpoint_every > 0:
        print(f"💾 Incremental checkpointing enabled - saving every {checkpoint_every:,} samples")
    print()
    
    # Process in chunks with checkpointing
    if checkpoint_every > 0:
        total_samples = len(dataset)
        processed_samples = 0
        
        # Process in chunks
        while processed_samples < total_samples:
            chunk_end = min(processed_samples + checkpoint_every, total_samples)
            chunk_size = chunk_end - processed_samples
            
            print(f"Processing samples {processed_samples:,} to {chunk_end:,} ({chunk_size:,} samples)...")
            
            # Select chunk to process
            chunk_dataset = dataset.select(range(processed_samples, chunk_end))
            
            # Process this chunk
            processed_chunk = chunk_dataset.map(
                lambda batch: add_codes_batch(
                    batch=batch,
                    codec=codec,
                    device=codec_device,
                    force_reprocess=force_reprocess,
                    max_duration_sec=max_duration_sec,
                    debug=True,
                ),
                batched=True,
                batch_size=batch_size,
                num_proc=None,
                desc=f"Chunk {processed_samples//checkpoint_every + 1}",
                remove_columns=[],
            )
            
            # Update the dataset with processed chunk
            if processed_samples == 0:
                # First chunk - initialize processed_dataset
                processed_dataset = processed_chunk
            else:
                # Subsequent chunks - concatenate
                processed_dataset = concatenate_datasets([processed_dataset, processed_chunk])
            
            processed_samples = chunk_end
            
            # Save checkpoint after each chunk
            print(f"💾 Saving checkpoint at {processed_samples:,}/{total_samples:,} samples...")
            try:
                processed_dataset.save_to_disk(checkpoint_path)
                print(f"✅ Checkpoint saved to {checkpoint_path}")
            except Exception as e:
                print(f"⚠️ Failed to save checkpoint: {e}")
            print()
        
        print("✅ All chunks processed!")
    else:
        # No checkpointing - process all at once
        print("⚠️ Checkpointing disabled - processing entire dataset at once")
        processed_dataset = dataset.map(
            lambda batch: add_codes_batch(
                batch=batch,
                codec=codec,
                device=codec_device,
                force_reprocess=force_reprocess,
                max_duration_sec=max_duration_sec,
                debug=True,
            ),
            batched=True,
            batch_size=batch_size,
            num_proc=None,
            desc="Adding VQ codes",
            remove_columns=[],
        )
    
    print()
    print("✅ Processing complete!")
    print()
    
    # Verify codes were added
    if 'codes' in processed_dataset.features:
        sample = processed_dataset[0]
        if sample['codes'] and len(sample['codes']) > 0:
            print(f"✅ Codes successfully added!")
            print(f"   Sample code shape: {len(sample['codes'])} codes")
        else:
            print(f"⚠️ Warning: Codes column exists but may be empty")
    else:
        print(f"⚠️ Warning: Codes column not found in processed dataset")
    
    print()
    
    # Save final checkpoint before pushing (in case push fails)
    # Note: If incremental checkpointing was enabled, this is already saved
    if checkpoint_every > 0:
        print(f"💾 Saving final checkpoint to {checkpoint_path}...")
        try:
            processed_dataset.save_to_disk(checkpoint_path)
            print(f"✅ Final checkpoint saved successfully")
            print()
        except Exception as e:
            print(f"⚠️ Failed to save final checkpoint: {e}")
            print()
    elif checkpoint_every == 0:
        # User disabled checkpointing but we still recommend saving before push
        print("ℹ️ No checkpoint saved (checkpoint_every=0)")
        print()
    
    # Push to hub if requested
    if push_to_hub:
        print(f"📤 Pushing updated dataset to {hf_repo}...")
        try:
            processed_dataset.push_to_hub(
                hf_repo,
                private=False,
                commit_message="Add VQ codec codes to dataset"
            )
            print(f"✅ Dataset uploaded successfully!")
            print(f"   View at: https://huggingface.co/datasets/{hf_repo}")
            print()
            
            # Suggest cleanup
            if checkpoint_every > 0 and Path(checkpoint_path).exists():
                print(f"💡 Tip: You can now safely delete the checkpoint to save disk space:")
                print(f"   rm -rf {checkpoint_path}")
        except Exception as e:
            print(f"❌ Failed to push to Hub: {e}")
            print("   The processed dataset is saved in checkpoint.")
            print(f"   You can retry pushing later or manually push from {checkpoint_path}")
    else:
        print("ℹ️ Skipping push to Hub (push_to_hub=False)")
        print("   To save locally: dataset.save_to_disk('path/to/save')")
    
    print()
    print("=" * 70)
    print("✅ All Done!")
    print("=" * 70)
    
    # Cleanup
    del codec
    if codec_device == "cuda":
        torch.cuda.empty_cache()
    
    return processed_dataset


if __name__ == "__main__":
    Fire(process_dataset)


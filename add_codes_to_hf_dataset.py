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
        --batch_size=100 \
        --push_to_hub=True

    # Test with small sample first
    python add_codes_to_hf_dataset.py \
        --hf_repo=omarabb315/mix-1000 \
        --codec_device=cuda \
        --batch_size=50 \
        --max_samples=100 \
        --push_to_hub=False

    # Force reprocess samples that already have codes
    python add_codes_to_hf_dataset.py \
        --hf_repo=omarabb315/mix-1000 \
        --codec_device=cuda \
        --force_reprocess=True

Workflow:
    1. Download dataset from HuggingFace Hub
    2. Load NeuCodec model on GPU
    3. Process dataset in batches to add VQ codes
    4. Push updated dataset back to HuggingFace Hub

Requirements:
    - HuggingFace account with login (huggingface-cli login)
    - Sufficient RAM for dataset (auto-cached by HF)
    - GPU with enough VRAM for NeuCodec (40GB+ recommended)
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm
from datasets import load_dataset, Dataset, Audio, Features, Value, Sequence
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
) -> Dict[str, Any]:
    """
    Add VQ codes to a batch of samples.
    
    Args:
        batch: Dictionary with 'audio' key containing audio arrays
        codec: NeuCodec model
        device: Device to run encoding on
        force_reprocess: If True, reprocess even if codes exist
    
    Returns:
        Dictionary with 'codes' key containing VQ codes
    """
    # Check if codes already exist
    if 'codes' in batch and not force_reprocess:
        # Check if any sample in batch already has codes
        existing_codes = batch['codes']
        if existing_codes and len(existing_codes) > 0:
            # If first sample has codes, assume all do (skip batch)
            if existing_codes[0] is not None and len(existing_codes[0]) > 0:
                return batch
    
    # Extract audio arrays
    audio_arrays = batch['audio']
    batch_codes = []
    
    # Process each audio in the batch
    for audio_data in audio_arrays:
        # Handle different audio formats from HuggingFace
        if isinstance(audio_data, dict):
            audio_array = audio_data['array']
        else:
            audio_array = audio_data
        
        # Convert to tensor
        with torch.no_grad():
            audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0).unsqueeze(0)
            audio_tensor = audio_tensor.to(device)
            
            # Encode to VQ codes
            vq_codes = codec.encode_code(audio_or_path=audio_tensor)
            vq_codes = vq_codes.squeeze(0).squeeze(0)
            
            if vq_codes.is_cuda:
                vq_codes = vq_codes.cpu()
            
            vq_codes_list = vq_codes.numpy().tolist()
            batch_codes.append(vq_codes_list)
    
    # Add codes to batch
    batch['codes'] = batch_codes
    
    return batch


def process_dataset(
    hf_repo: str,
    codec_device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 100,
    num_proc: int = 4,
    max_samples: Optional[int] = None,
    force_reprocess: bool = False,
    push_to_hub: bool = True,
    split: str = "train",
):
    """
    Main processing function.
    
    Args:
        hf_repo: HuggingFace repository name (e.g., 'username/dataset-name')
        codec_device: Device for NeuCodec ('cuda' or 'cpu')
        batch_size: Number of samples to process at once
        num_proc: Number of processes for parallel processing
        max_samples: Maximum number of samples to process (None = all)
        force_reprocess: Reprocess samples that already have codes
        push_to_hub: Push updated dataset back to HuggingFace Hub
        split: Dataset split to process ('train', 'test', etc.)
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
    print(f"  Num Processes: {num_proc}")
    print(f"  Max Samples: {max_samples if max_samples else 'All'}")
    print(f"  Force Reprocess: {force_reprocess}")
    print(f"  Push to Hub: {push_to_hub}")
    print()
    
    # Load dataset
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
    
    # Load codec model
    codec = load_neucodec(codec_device)
    print()
    
    # Process dataset
    print(f"Processing dataset to add VQ codes...")
    print(f"This may take several hours for large datasets...")
    print()
    
    # Use dataset.map with batching for efficient processing
    processed_dataset = dataset.map(
        lambda batch: add_codes_batch(
            batch=batch,
            codec=codec,
            device=codec_device,
            force_reprocess=force_reprocess,
        ),
        batched=True,
        batch_size=batch_size,
        num_proc=1,  # Set to 1 to avoid GPU conflicts (NeuCodec is GPU-bound)
        desc="Adding VQ codes",
        # Remove codes column if it exists, we'll add it fresh
        remove_columns=['codes'] if 'codes' in dataset.features and force_reprocess else [],
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
        except Exception as e:
            print(f"❌ Failed to push to Hub: {e}")
            print("   The processed dataset is still available locally.")
            print("   You can manually save it using dataset.save_to_disk()")
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


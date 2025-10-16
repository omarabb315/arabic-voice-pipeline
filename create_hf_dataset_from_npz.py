"""
Create HuggingFace Dataset from .npz segment files.

Run this after Stage 1 (or Stage 2) to create the final training dataset.
Fast and memory-efficient - loads .npz files in batches.

Supports:
- Local directory processing
- S3 streaming (NEW)
- Push to HuggingFace Hub (NEW)
"""

import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, Audio, Features, Value, Sequence
from fire import Fire

# Add scripts directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))


def create_dataset_from_npz(
    segments_cache_dir: str = "segments_cache/",
    output_path: str = "dataset_cache/segments_dataset/",
    channels: list[str] = None,
    batch_size: int = 1000,
    s3_prefix: str = None,
    gcs_bucket: str = None,
    gcs_prefix: str = None,
    push_to_hub: bool = False,
    hf_repo: str = None,
):
    """
    Create HuggingFace dataset from .npz segment files.
    
    Args:
        segments_cache_dir: Local directory with .npz files (ignored if s3_prefix/gcs_prefix is set)
        output_path: Where to save final HF dataset
        channels: List of channels to include (None = all)
        batch_size: Number of segments to load at once (for memory efficiency)
        s3_prefix: S3 prefix to stream from (alternative to local)
        gcs_bucket: GCS bucket name (alternative to local)
        gcs_prefix: GCS prefix to stream from (alternative to local)
        push_to_hub: Push dataset to HuggingFace Hub
        hf_repo: HuggingFace repository name (e.g., 'username/dataset-name')
    """
    
    print("=" * 70)
    print("Creating HuggingFace Dataset from .npz Files")
    print("=" * 70)
    print()
    
    # Determine source: S3, GCS, or local
    if s3_prefix:
        print(f"Source: S3 ({s3_prefix})")
        from s3_utils import S3Client
        s3_client = S3Client()
        
        # List all files in S3
        print("Listing files in S3...")
        all_files = s3_client.list_files(prefix=s3_prefix, suffix=".npz")
        print(f"Found {len(all_files):,} files in S3")
        
        # Filter by channels if specified
        if channels:
            filtered_files = []
            for f in all_files:
                for channel in channels:
                    if f"/{channel}/" in f:
                        filtered_files.append(f)
                        break
            all_files = filtered_files
            print(f"After channel filter: {len(all_files):,} files")
        
        segment_files = all_files
        source_is_s3 = True
        source_is_gcs = False
    elif gcs_prefix:
        print(f"Source: GCS (gs://{gcs_bucket}/{gcs_prefix})")
        from gcs_utils import GCSClient
        gcs_client = GCSClient(bucket_name=gcs_bucket)
        
        # List all files in GCS
        print("Listing files in GCS...")
        all_files = gcs_client.list_files(prefix=gcs_prefix, suffix=".npz")
        print(f"Found {len(all_files):,} files in GCS")
        
        # Filter by channels if specified
        if channels:
            filtered_files = []
            for f in all_files:
                for channel in channels:
                    if f"/{channel}/" in f:
                        filtered_files.append(f)
                        break
            all_files = filtered_files
            print(f"After channel filter: {len(all_files):,} files")
        
        segment_files = all_files
        source_is_s3 = False
        source_is_gcs = True
    else:
        print(f"Source: Local ({segments_cache_dir})")
        cache_path = Path(segments_cache_dir)
        
        # Find all channels
        channel_dirs = [d for d in cache_path.iterdir() if d.is_dir()]
        if channels:
            channel_dirs = [d for d in channel_dirs if d.name in channels]
        
        print(f"Found {len(channel_dirs)} channels to process")
        for ch in channel_dirs:
            seg_count = len(list(ch.glob("*.npz")))
            print(f"  - {ch.name}: {seg_count:,} segments")
        
        # Collect all file paths
        segment_files = []
        for channel_dir in channel_dirs:
            segment_files.extend(list(channel_dir.glob("*.npz")))
        
        source_is_s3 = False
        source_is_gcs = False
    
    print()
    print(f"Total segments to process: {len(segment_files):,}")
    print()
    
    # Load segments
    all_segments = []
    total_loaded = 0
    
    print("Loading segments...")
    for seg_file in tqdm(segment_files, desc="Loading"):
        try:
            if source_is_s3:
                # Download to temp location
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.npz', delete=True) as tmp:
                    if not s3_client.download_file(seg_file, tmp.name, show_progress=False):
                        continue
                    data = np.load(tmp.name, allow_pickle=True)
            elif source_is_gcs:
                # Download to temp location
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.npz', delete=True) as tmp:
                    if not gcs_client.download_file(seg_file, tmp.name, show_progress=False):
                        continue
                    data = np.load(tmp.name, allow_pickle=True)
            else:
                data = np.load(seg_file, allow_pickle=True)
            
            # Convert to HF format
            segment = {
                'audio': {
                    'array': data['audio'],
                    'sampling_rate': 16000
                },
                'codes': data['codes'].tolist() if isinstance(data['codes'], np.ndarray) else list(data['codes']),
                'text': str(data['text']),
                'speaker_id': str(data['speaker_id']),
                'channel': str(data['channel']),
                'gender': str(data['gender']),
                'dialect': str(data['dialect']),
                'tone': str(data['tone']),
                'duration_sec': float(data['duration_sec']),
                'audio_id': str(data['audio_id']),
            }
            all_segments.append(segment)
            total_loaded += 1
            
        except Exception as e:
            print(f"⚠️ Error loading {seg_file}: {e}")
            continue
    
    if len(all_segments) == 0:
        print("❌ No segments loaded!")
        return
    
    print(f"\nCreating HuggingFace Dataset with {len(all_segments):,} segments...")
    
    features = Features({
        'audio': Audio(sampling_rate=16000),
        'codes': Sequence(Value('int32')),
        'text': Value('string'),
        'speaker_id': Value('string'),
        'channel': Value('string'),
        'gender': Value('string'),
        'dialect': Value('string'),
        'tone': Value('string'),
        'duration_sec': Value('float32'),
        'audio_id': Value('string'),
    })
    
    dataset = Dataset.from_list(all_segments, features=features)
    
    print(f"Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    dataset.save_to_disk(output_path)
    
    print()
    print("✅ Dataset created successfully!")
    print(f"📊 Dataset info:")
    print(f"   - Total segments: {len(dataset):,}")
    print(f"   - Channels: {set(dataset['channel'])}")
    print(f"   - Unique speakers: {len(set(dataset['speaker_id'])):,}")
    print(f"   - Total duration: {sum(dataset['duration_sec']) / 3600:.2f} hours")
    print(f"   - Saved to: {output_path}")
    print()
    
    # Push to HuggingFace Hub if requested
    if push_to_hub:
        if not hf_repo:
            print("⚠️ Warning: push_to_hub=True but no hf_repo specified!")
            print("   Skipping HuggingFace Hub upload")
        else:
            print(f"📤 Pushing to HuggingFace Hub: {hf_repo}")
            try:
                dataset.push_to_hub(hf_repo, private=False)
                print(f"✅ Dataset uploaded to https://huggingface.co/datasets/{hf_repo}")
            except Exception as e:
                print(f"❌ Failed to push to Hub: {e}")
                print("   Make sure you're logged in: huggingface-cli login")
    
    print()
    print("💡 Tip: You can delete segments_cache/ to save disk space")
    print("   (or keep it for easier debugging/reprocessing)")


if __name__ == "__main__":
    Fire(create_dataset_from_npz)


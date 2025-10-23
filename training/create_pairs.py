import os
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from datasets import load_from_disk, Dataset, Audio, Features, Value, Sequence
from tqdm import tqdm
from fire import Fire


def detect_input_format(segments_path: str) -> str:
    """
    Auto-detect input format (npz files or HuggingFace dataset).
    
    Args:
        segments_path: Path to check
        
    Returns:
        "npz" or "hf_dataset"
    """
    if not os.path.exists(segments_path):
        raise ValueError(f"Path does not exist: {segments_path}")
    
    # Check for HuggingFace dataset markers
    if os.path.isdir(segments_path):
        if os.path.exists(os.path.join(segments_path, "dataset_info.json")):
            return "hf_dataset"
        
        # Check if directory contains .npz files
        npz_files = list(Path(segments_path).rglob("*.npz"))
        if len(npz_files) > 0:
            return "npz"
    
    raise ValueError(f"Could not detect format for: {segments_path}")


def load_segments_from_npz(segments_dir: str, min_duration: float = 0.5, max_duration: float = 15.0) -> list:
    """
    Load segments from .npz files directly.
    
    Args:
        segments_dir: Directory containing .npz segment files
        min_duration: Minimum duration filter
        max_duration: Maximum duration filter
        
    Returns:
        List of segment dictionaries
    """
    print(f"Loading segments from .npz files in {segments_dir}...")
    
    # Find all .npz files recursively
    npz_files = list(Path(segments_dir).rglob("*.npz"))
    print(f"Found {len(npz_files):,} .npz files")
    
    if len(npz_files) == 0:
        raise ValueError(f"No .npz files found in {segments_dir}")
    
    # Load segments
    segments = []
    skipped_no_codes = 0
    skipped_duration = 0
    
    for npz_path in tqdm(npz_files, desc="Loading segments"):
        try:
            data = np.load(npz_path, allow_pickle=True)
            
            # Check if codes exist
            if 'codes' not in data or data['codes'] is None or len(data['codes']) == 0:
                skipped_no_codes += 1
                continue
            
            # Check duration
            duration = float(data['duration_sec'])
            if duration < min_duration or duration > max_duration:
                skipped_duration += 1
                continue
            
            # Create segment dict
            segment = {
                'audio': {'array': data['audio'], 'sampling_rate': 16000},
                'text': str(data['text']),
                'codes': data['codes'].tolist() if isinstance(data['codes'], np.ndarray) else data['codes'],
                'speaker_id': str(data['speaker_id']),
                'channel': str(data['channel']),
                'gender': str(data['gender']),
                'dialect': str(data['dialect']),
                'tone': str(data['tone']),
                'duration_sec': duration,
            }
            
            segments.append(segment)
            
        except Exception as e:
            print(f"Warning: Could not load {npz_path}: {e}")
            continue
    
    print(f"\nLoaded {len(segments):,} valid segments")
    print(f"Skipped {skipped_no_codes:,} segments without codes")
    print(f"Skipped {skipped_duration:,} segments due to duration filter")
    
    return segments


def load_segments_from_hf_dataset(dataset_path: str, min_duration: float = 0.5, max_duration: float = 15.0) -> list:
    """
    Load segments from HuggingFace dataset.
    
    Args:
        dataset_path: Path to HF dataset
        min_duration: Minimum duration filter
        max_duration: Maximum duration filter
        
    Returns:
        List of segment dictionaries
    """
    print(f"Loading HuggingFace dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    print(f"Loaded {len(dataset):,} segments")
    
    # Apply filters
    print(f"Filtering by duration ({min_duration}s - {max_duration}s)...")
    dataset = dataset.filter(
        lambda x: min_duration <= x['duration_sec'] <= max_duration
    )
    print(f"Segments after duration filter: {len(dataset):,}")
    
    print("Filtering out segments with empty codes...")
    dataset = dataset.filter(
        lambda x: x['codes'] is not None and len(x['codes']) > 0
    )
    print(f"Segments after code filter: {len(dataset):,}")
    
    # Convert to list of dicts
    segments = []
    for sample in tqdm(dataset, desc="Converting to list"):
        segments.append(sample)
    
    return segments


def create_paired_dataset(
    segments_path: str = "dataset_cache/segments_dataset/",
    output_path: str = "dataset_cache/paired_dataset/",
    input_format: str = "auto",
    min_duration_sec: float = 0.5,
    max_duration_sec: float = 15.0,
    min_segments_per_speaker: int = 2,
    max_pairs_per_speaker: int = None,
    force_rebuild: bool = False,
    random_seed: int = 42,
):
    """
    Create paired dataset for voice cloning training.
    
    Supports loading from:
    - .npz files (raw segments) - more efficient, saves disk space
    - HuggingFace dataset - backward compatible
    
    Args:
        segments_path: Path to segments (directory with .npz files or HF dataset)
        output_path: Path to save paired dataset
        input_format: "auto" (auto-detect), "npz" (load .npz files), or "hf_dataset" (load HF)
        min_duration_sec: Minimum segment duration
        max_duration_sec: Maximum segment duration
        min_segments_per_speaker: Minimum segments required per speaker
        max_pairs_per_speaker: Maximum pairs per speaker (None = all possible pairs)
        force_rebuild: If True, rebuild even if dataset exists
        random_seed: Random seed for pair sampling
    """
    
    # Check if dataset already exists
    if os.path.exists(output_path) and not force_rebuild:
        print(f"Paired dataset already exists at {output_path}. Use force_rebuild=True to rebuild.")
        return
    
    # Detect format
    if input_format == "auto":
        input_format = detect_input_format(segments_path)
        print(f"Auto-detected input format: {input_format}")
    
    # Load segments based on format
    if input_format == "npz":
        segments = load_segments_from_npz(segments_path, min_duration_sec, max_duration_sec)
    elif input_format == "hf_dataset":
        segments = load_segments_from_hf_dataset(segments_path, min_duration_sec, max_duration_sec)
    else:
        raise ValueError(f"Unknown input format: {input_format}. Use 'auto', 'npz', or 'hf_dataset'")
    
    if len(segments) == 0:
        print("❌ No valid segments found. Check your data and filters.")
        return
    
    # Check if audio_id field exists in dataset
    has_audio_id = 'audio_id' in segments[0]
    
    # Group by unique speaker identity
    # Note: speaker_id is incremental per audio file, so we need audio_id to ensure uniqueness
    if has_audio_id:
        print("\nGrouping segments by (audio_id, speaker_id) to ensure speaker uniqueness...")
        speaker_groups = defaultdict(list)
        
        for idx, sample in enumerate(tqdm(segments, desc="Grouping")):
            # Extract base audio_id (remove _seg_X suffix if present)
            audio_id_full = sample['audio_id']
            audio_id_base = audio_id_full.rsplit('_seg_', 1)[0] if '_seg_' in audio_id_full else audio_id_full
            key = (audio_id_base, sample['speaker_id'])
            speaker_groups[key].append(idx)
        
        print(f"Found {len(speaker_groups)} unique (audio_id, speaker_id) combinations")
    else:
        print("\n⚠️  WARNING: No 'audio_id' field found. Falling back to (channel, speaker_id) grouping.")
        print("   This may cause issues if speaker_id is not globally unique!")
        speaker_groups = defaultdict(list)
        
        for idx, sample in enumerate(tqdm(segments, desc="Grouping")):
            key = (sample['channel'], sample['speaker_id'])
            speaker_groups[key].append(idx)
        
        print(f"Found {len(speaker_groups)} unique (channel, speaker_id) combinations")
    
    # Filter speakers with insufficient segments
    print(f"\nFiltering speakers with < {min_segments_per_speaker} segments...")
    valid_speakers = {
        speaker: indices 
        for speaker, indices in speaker_groups.items() 
        if len(indices) >= min_segments_per_speaker
    }
    print(f"Valid speakers: {len(valid_speakers)}")
    
    # Create pairs
    print("\nCreating reference/target pairs...")
    random.seed(random_seed)
    all_pairs = []
    
    speaker_pair_counts = []
    
    for speaker, indices in tqdm(valid_speakers.items(), desc="Creating pairs"):
        n_segments = len(indices)
        
        # Generate all possible pairs (ref != target)
        possible_pairs = [
            (ref_idx, target_idx)
            for ref_idx in indices
            for target_idx in indices
            if ref_idx != target_idx
        ]
        
        # Limit pairs per speaker if specified
        if max_pairs_per_speaker is not None and len(possible_pairs) > max_pairs_per_speaker:
            possible_pairs = random.sample(possible_pairs, max_pairs_per_speaker)
        
        # Create pair entries
        for ref_idx, target_idx in possible_pairs:
            ref_sample = segments[ref_idx]
            target_sample = segments[target_idx]
            
            pair = {
                # Reference
                'ref_audio': ref_sample['audio'],
                'ref_text': ref_sample['text'],
                'ref_codes': ref_sample['codes'],
                'ref_speaker_id': ref_sample['speaker_id'],
                'ref_gender': ref_sample['gender'],
                'ref_dialect': ref_sample['dialect'],
                'ref_tone': ref_sample['tone'],
                'ref_duration_sec': ref_sample['duration_sec'],
                
                # Target
                'target_audio': target_sample['audio'],
                'target_text': target_sample['text'],
                'target_codes': target_sample['codes'],
                'target_speaker_id': target_sample['speaker_id'],
                'target_gender': target_sample['gender'],
                'target_dialect': target_sample['dialect'],
                'target_tone': target_sample['tone'],
                'target_duration_sec': target_sample['duration_sec'],
                
                # Metadata
                'channel': ref_sample['channel'],
            }
            
            all_pairs.append(pair)
        
        speaker_pair_counts.append(len(possible_pairs))
    
    print(f"\n{'='*60}")
    print(f"Total pairs created: {len(all_pairs)}")
    print(f"{'='*60}")
    
    if len(all_pairs) == 0:
        print("❌ No pairs created. Check your filtering parameters.")
        return
    
    # Statistics
    print(f"\n📊 Pair Statistics:")
    print(f"   - Total pairs: {len(all_pairs)}")
    print(f"   - Speakers with pairs: {len(valid_speakers)}")
    print(f"   - Avg pairs per speaker: {sum(speaker_pair_counts) / len(speaker_pair_counts):.1f}")
    print(f"   - Min pairs per speaker: {min(speaker_pair_counts)}")
    print(f"   - Max pairs per speaker: {max(speaker_pair_counts)}")
    
    # Create HuggingFace Dataset
    print("\nCreating paired HuggingFace Dataset...")
    
    features = Features({
        # Reference
        'ref_audio': Audio(sampling_rate=16000),
        'ref_text': Value('string'),
        'ref_codes': Sequence(Value('int32')),
        'ref_speaker_id': Value('string'),
        'ref_gender': Value('string'),
        'ref_dialect': Value('string'),
        'ref_tone': Value('string'),
        'ref_duration_sec': Value('float32'),
        
        # Target
        'target_audio': Audio(sampling_rate=16000),
        'target_text': Value('string'),
        'target_codes': Sequence(Value('int32')),
        'target_speaker_id': Value('string'),
        'target_gender': Value('string'),
        'target_dialect': Value('string'),
        'target_tone': Value('string'),
        'target_duration_sec': Value('float32'),
        
        # Metadata
        'channel': Value('string'),
    })
    
    paired_dataset = Dataset.from_list(all_pairs, features=features)
    
    # Save to disk
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    paired_dataset.save_to_disk(output_path)
    
    print(f"✅ Paired dataset saved to: {output_path}")
    print(f"📊 Dataset size: {len(paired_dataset)} pairs")


if __name__ == "__main__":
    Fire(create_paired_dataset)


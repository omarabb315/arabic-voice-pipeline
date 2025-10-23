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


def build_speaker_index_from_npz(segments_dir: str, min_duration: float = 0.5, max_duration: float = 15.0) -> tuple:
    """
    Build a lightweight index of segments grouped by speaker.
    Only stores file paths and minimal metadata, not full data.
    
    Returns:
        (speaker_groups, segment_paths) where:
        - speaker_groups: dict mapping speaker -> list of segment indices
        - segment_paths: list of paths to valid .npz files
    """
    print(f"Building speaker index from .npz files in {segments_dir}...")
    
    # Find all .npz files
    npz_files = list(Path(segments_dir).rglob("*.npz"))
    print(f"Found {len(npz_files):,} .npz files")
    
    if len(npz_files) == 0:
        raise ValueError(f"No .npz files found in {segments_dir}")
    
    speaker_groups = defaultdict(list)
    segment_paths = []
    skipped_no_codes = 0
    skipped_duration = 0
    
    # First pass: build index without loading audio data
    for npz_path in tqdm(npz_files, desc="Building index"):
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
            
            # Add to index
            segment_idx = len(segment_paths)
            segment_paths.append(str(npz_path))
            
            # Group by speaker
            speaker_id = str(data['speaker_id'])
            channel = str(data['channel'])
            
            # Use channel + speaker_id as key for uniqueness
            speaker_key = (channel, speaker_id)
            speaker_groups[speaker_key].append(segment_idx)
            
        except Exception as e:
            print(f"Warning: Could not read {npz_path}: {e}")
            continue
    
    print(f"\nIndexed {len(segment_paths):,} valid segments")
    print(f"Found {len(speaker_groups)} unique speakers")
    print(f"Skipped {skipped_no_codes:,} segments without codes")
    print(f"Skipped {skipped_duration:,} segments due to duration filter")
    
    return speaker_groups, segment_paths


def load_segment_from_npz(npz_path: str) -> dict:
    """Load a single segment from an .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    
    return {
        'audio': {'array': data['audio'], 'sampling_rate': 16000},
        'text': str(data['text']),
        'codes': data['codes'].tolist() if isinstance(data['codes'], np.ndarray) else data['codes'],
        'speaker_id': str(data['speaker_id']),
        'channel': str(data['channel']),
        'gender': str(data['gender']),
        'dialect': str(data['dialect']),
        'tone': str(data['tone']),
        'duration_sec': float(data['duration_sec']),
    }


def build_speaker_index_from_hf_dataset(dataset_path: str, min_duration: float = 0.5, max_duration: float = 15.0) -> tuple:
    """
    Build a lightweight index from HF dataset.
    
    Returns:
        (speaker_groups, dataset) where dataset is the filtered HF dataset
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
    
    # Build speaker index
    print("Building speaker index...")
    speaker_groups = defaultdict(list)
    
    has_audio_id = 'audio_id' in dataset[0] if len(dataset) > 0 else False
    
    for idx in tqdm(range(len(dataset)), desc="Indexing"):
        sample = dataset[idx]
        
        if has_audio_id:
            audio_id_full = sample['audio_id']
            audio_id_base = audio_id_full.rsplit('_seg_', 1)[0] if '_seg_' in audio_id_full else audio_id_full
            speaker_key = (audio_id_base, sample['speaker_id'])
        else:
            speaker_key = (sample['channel'], sample['speaker_id'])
        
        speaker_groups[speaker_key].append(idx)
    
    print(f"Found {len(speaker_groups)} unique speakers")
    
    return speaker_groups, dataset


def create_pairs_generator(speaker_groups: dict, segment_source, max_pairs_per_speaker: int = None, random_seed: int = 42):
    """
    Generator that yields pairs one at a time without storing all in memory.
    
    Args:
        speaker_groups: Dict mapping speaker -> list of segment indices
        segment_source: Either list of paths (for NPZ) or HF dataset
        max_pairs_per_speaker: Max pairs per speaker
        random_seed: Random seed
        
    Yields:
        Pair dictionaries
    """
    random.seed(random_seed)
    is_npz = isinstance(segment_source, list)
    
    for speaker, indices in speaker_groups.items():
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
        
        # Yield pairs
        for ref_idx, target_idx in possible_pairs:
            # Load segments on-demand
            if is_npz:
                ref_sample = load_segment_from_npz(segment_source[ref_idx])
                target_sample = load_segment_from_npz(segment_source[target_idx])
            else:
                ref_sample = segment_source[ref_idx]
                target_sample = segment_source[target_idx]
            
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
            
            yield pair


def create_paired_dataset(
    segments_path: str = "dataset_cache/segments_dataset/",
    output_path: str = "dataset_cache/paired_dataset/",
    input_format: str = "auto",
    min_duration_sec: float = 0.5,
    max_duration_sec: float = 15.0,
    min_segments_per_speaker: int = 2,
    max_pairs_per_speaker: int = None,
    batch_size: int = 10000,
    force_rebuild: bool = False,
    random_seed: int = 42,
):
    """
    Create paired dataset for voice cloning training with memory-efficient processing.
    
    OPTIMIZATIONS:
    - Two-pass approach: build lightweight index first, load data on-demand
    - Generator-based pair creation: don't store all pairs in memory
    - Batched writing: write to disk in chunks
    
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
        batch_size: Number of pairs to accumulate before writing (lower = less memory)
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
    
    # Build lightweight speaker index
    print("\n" + "="*60)
    print("STEP 1: Building Speaker Index (Memory Efficient)")
    print("="*60)
    
    if input_format == "npz":
        speaker_groups, segment_source = build_speaker_index_from_npz(
            segments_path, min_duration_sec, max_duration_sec
        )
    elif input_format == "hf_dataset":
        speaker_groups, segment_source = build_speaker_index_from_hf_dataset(
            segments_path, min_duration_sec, max_duration_sec
        )
    else:
        raise ValueError(f"Unknown input format: {input_format}. Use 'auto', 'npz', or 'hf_dataset'")
    
    if len(speaker_groups) == 0:
        print("❌ No valid segments found. Check your data and filters.")
        return
    
    # Filter speakers with insufficient segments
    print(f"\nFiltering speakers with < {min_segments_per_speaker} segments...")
    valid_speakers = {
        speaker: indices 
        for speaker, indices in speaker_groups.items() 
        if len(indices) >= min_segments_per_speaker
    }
    print(f"Valid speakers: {len(valid_speakers)}")
    
    if len(valid_speakers) == 0:
        print("❌ No valid speakers found. Check your filtering parameters.")
        return
    
    # Calculate expected pairs
    print("\nCalculating expected pairs...")
    total_expected_pairs = 0
    speaker_pair_counts = []
    
    for speaker, indices in valid_speakers.items():
        n_segments = len(indices)
        n_possible_pairs = n_segments * (n_segments - 1)  # All pairs where ref != target
        
        if max_pairs_per_speaker is not None:
            n_pairs = min(n_possible_pairs, max_pairs_per_speaker)
        else:
            n_pairs = n_possible_pairs
        
        total_expected_pairs += n_pairs
        speaker_pair_counts.append(n_pairs)
    
    print(f"Expected total pairs: {total_expected_pairs:,}")
    print(f"Avg pairs per speaker: {sum(speaker_pair_counts) / len(speaker_pair_counts):.1f}")
    print(f"Min pairs per speaker: {min(speaker_pair_counts)}")
    print(f"Max pairs per speaker: {max(speaker_pair_counts)}")
    
    # Create pairs in batches
    print("\n" + "="*60)
    print(f"STEP 2: Creating Pairs (Batched, batch_size={batch_size})")
    print("="*60)
    
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
    
    # Create pairs using generator and write in batches
    pair_generator = create_pairs_generator(
        valid_speakers, 
        segment_source, 
        max_pairs_per_speaker, 
        random_seed
    )
    
    batch = []
    all_datasets = []
    total_pairs_created = 0
    
    print("Creating pairs in batches...")
    pbar = tqdm(total=total_expected_pairs, desc="Creating pairs")
    
    for pair in pair_generator:
        batch.append(pair)
        total_pairs_created += 1
        pbar.update(1)
        
        # Write batch when it reaches batch_size
        if len(batch) >= batch_size:
            batch_dataset = Dataset.from_list(batch, features=features)
            all_datasets.append(batch_dataset)
            batch = []
    
    # Write remaining pairs
    if len(batch) > 0:
        batch_dataset = Dataset.from_list(batch, features=features)
        all_datasets.append(batch_dataset)
    
    pbar.close()
    
    if len(all_datasets) == 0:
        print("❌ No pairs created. Check your filtering parameters.")
        return
    
    # Concatenate all batches
    print("\nConcatenating batches...")
    from datasets import concatenate_datasets
    paired_dataset = concatenate_datasets(all_datasets)
    
    # Save to disk
    print(f"\nSaving to {output_path}...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    paired_dataset.save_to_disk(output_path)
    
    print(f"\n{'='*60}")
    print(f"✅ Paired dataset saved to: {output_path}")
    print(f"📊 Dataset size: {len(paired_dataset):,} pairs")
    print(f"📊 Speakers with pairs: {len(valid_speakers)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    Fire(create_paired_dataset)

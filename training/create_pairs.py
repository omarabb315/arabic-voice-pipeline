import os
import random
from collections import defaultdict
from datasets import load_from_disk, Dataset, Audio, Features, Value, Sequence
from tqdm import tqdm
from fire import Fire


def create_paired_dataset(
    segments_dataset_path: str = "dataset_cache/segments_dataset/",
    output_path: str = "dataset_cache/paired_dataset/",
    min_duration_sec: float = 0.5,
    max_duration_sec: float = 15.0,
    min_segments_per_speaker: int = 2,
    max_pairs_per_speaker: int = None,
    force_rebuild: bool = False,
    random_seed: int = 42,
):
    """
    Create paired dataset for voice cloning training.
    
    Args:
        segments_dataset_path: Path to segments dataset
        output_path: Path to save paired dataset
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
    
    # Load segments dataset
    print(f"Loading segments dataset from {segments_dataset_path}...")
    dataset = load_from_disk(segments_dataset_path)
    print(f"Loaded {len(dataset)} segments")
    
    # Filter by duration
    print(f"\nFiltering segments by duration ({min_duration_sec}s - {max_duration_sec}s)...")
    dataset = dataset.filter(
        lambda x: min_duration_sec <= x['duration_sec'] <= max_duration_sec
    )
    print(f"Segments after duration filter: {len(dataset)}")
    
    # Filter out invalid codes
    print("Filtering out segments with invalid codes...")
    dataset = dataset.filter(
        lambda x: x['codes'] is not None and len(x['codes']) > 0
    )
    print(f"Segments after code filter: {len(dataset)}")
    
    # Group by (channel, speaker_id)
    print("\nGrouping segments by (channel, speaker_id)...")
    speaker_groups = defaultdict(list)
    
    for idx, sample in enumerate(tqdm(dataset, desc="Grouping")):
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
            ref_sample = dataset[ref_idx]
            target_sample = dataset[target_idx]
            
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


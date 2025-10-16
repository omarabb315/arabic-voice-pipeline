"""
Quick test script to verify the pipeline works with a small subset of data.
Run this before processing your entire dataset.
"""

import os
from build_dataset import build_segments_dataset
from create_pairs import create_paired_dataset


def test_pipeline(
    gcp_bucket_name: str = "audio-data-gemini",
    test_channel: str = "ZadTVchannel",  # Use one channel for testing
    codec_device: str = "cuda",
    audio_channel_name: str = None,  # Set this if audio folder name differs from processed name
):
    """
    Test the pipeline with a small subset of data.
    
    Args:
        gcp_bucket_name: GCP bucket name
        test_channel: Single channel to test with (processed folder name)
        codec_device: Device for codec ("cuda" or "cpu")
        audio_channel_name: Audio folder name if different from test_channel
                           (e.g., test_channel="Atheer_-", audio_channel_name="Atheer - أثير")
    """
    
    print("=" * 70)
    print("PIPELINE TEST - Small Subset")
    print("=" * 70)
    print()
    print(f"Testing with channel: {test_channel}")
    print()
    
    # Create test directories
    test_segments_path = "test_dataset_cache/segments_dataset/"
    test_paired_path = "test_dataset_cache/paired_dataset/"
    
    # Step 1: Build segments dataset
    print("Step 1: Building segments dataset...")
    print("-" * 70)
    
    # Build channel mapping if audio channel name is different
    channel_mapping = None
    if audio_channel_name and audio_channel_name != test_channel:
        channel_mapping = {test_channel: audio_channel_name}
        print(f"Using channel mapping: '{test_channel}' -> '{audio_channel_name}'")
    
    try:
        build_segments_dataset(
            gcp_bucket_name=gcp_bucket_name,
            channels=[test_channel],
            output_path=test_segments_path,
            codec_device=codec_device,
            force_rebuild=True,
            channel_mapping=channel_mapping,
        )
        print("✅ Segments dataset created successfully!")
    except Exception as e:
        print(f"❌ Error building segments dataset: {e}")
        return False
    
    print()
    
    # Step 2: Create paired dataset
    print("Step 2: Creating paired dataset...")
    print("-" * 70)
    
    try:
        create_paired_dataset(
            segments_dataset_path=test_segments_path,
            output_path=test_paired_path,
            min_duration_sec=0.5,
            max_duration_sec=15.0,
            min_segments_per_speaker=2,
            max_pairs_per_speaker=10,  # Limit for testing
            force_rebuild=True,
        )
        print("✅ Paired dataset created successfully!")
    except Exception as e:
        print(f"❌ Error creating paired dataset: {e}")
        return False
    
    print()
    print("=" * 70)
    print("✅ TEST PASSED!")
    print("=" * 70)
    print()
    print("The pipeline works correctly. You can now:")
    print("1. Process all your channels using build_dataset.py")
    print("2. Create the full paired dataset using create_pairs.py")
    print("3. Start training using finetune-neutts.py")
    print()
    print("Test datasets saved at:")
    print(f"  - {test_segments_path}")
    print(f"  - {test_paired_path}")
    print()
    print("You can delete these test datasets after verification:")
    print("  rm -rf test_dataset_cache/")
    
    return True


if __name__ == "__main__":
    import sys
    
    # Parse command line args
    if len(sys.argv) > 1:
        test_channel = sys.argv[1]
    else:
        test_channel = "ZadTVchannel"  # Default
    
    success = test_pipeline(test_channel=test_channel)
    
    if not success:
        print()
        print("=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)
        print()
        print("Please check the error messages above and fix any issues.")
        print("Common issues:")
        print("  - Wrong channel name")
        print("  - Wrong GCP bucket paths")
        print("  - Missing GCP permissions")
        print("  - Missing dependencies")
        sys.exit(1)


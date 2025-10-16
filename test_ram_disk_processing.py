"""
Test Script for RAM Disk Processing

Tests the RAM-optimized workflow with a small sample of segments
before running the full processing pipeline.

Usage:
    python test_ram_disk_processing.py
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, 'scripts')

from ram_disk_utils import (
    get_available_ram_gb,
    get_total_ram_gb,
    get_optimal_temp_config,
    setup_temp_directory,
    validate_ram_for_processing
)
from gcs_utils import GCSClient


def test_ram_detection():
    """Test RAM detection and configuration"""
    print("=" * 70)
    print("TEST 1: RAM Detection")
    print("=" * 70)
    print()
    
    total_ram = get_total_ram_gb()
    available_ram = get_available_ram_gb()
    
    print(f"Total RAM: {total_ram:.1f} GB")
    print(f"Available RAM: {available_ram:.1f} GB")
    print()
    
    if available_ram < 100:
        print("⚠️  WARNING: Less than 100 GB available RAM")
        print("   The full pipeline may require more RAM")
    else:
        print(f"✅ Sufficient RAM detected: {available_ram:.1f} GB")
    
    print()
    return available_ram >= 100


def test_temp_location():
    """Test temporary storage location detection"""
    print("=" * 70)
    print("TEST 2: Temp Storage Location")
    print("=" * 70)
    print()
    
    config = get_optimal_temp_config(
        required_space_gb=300.0,
        data_size_gb=221.0
    )
    
    print(f"Selected location: {config['temp_dir']}")
    print(f"Storage type: {config['storage_type']}")
    print(f"Available space: {config['available_space_gb']:.1f} GB")
    print()
    
    if config['ram_sufficient']:
        print("✅ RAM validation passed")
    else:
        print("⚠️  WARNING: May have insufficient RAM")
    
    print()
    return config['ram_sufficient']


def test_temp_directory_setup():
    """Test temporary directory creation and cleanup"""
    print("=" * 70)
    print("TEST 3: Temp Directory Setup")
    print("=" * 70)
    print()
    
    test_temp_dir = setup_temp_directory(
        base_path="/tmp",
        subdir="neutts_test",
        clean_if_exists=True
    )
    
    print(f"Created: {test_temp_dir}")
    
    if os.path.exists(test_temp_dir):
        print("✅ Directory created successfully")
        
        # Test write
        test_file = os.path.join(test_temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        
        if os.path.exists(test_file):
            print("✅ Write test passed")
        else:
            print("❌ Write test failed")
            return False
        
        # Cleanup
        import shutil
        shutil.rmtree(test_temp_dir, ignore_errors=True)
        print("✅ Cleanup successful")
        print()
        return True
    else:
        print("❌ Failed to create directory")
        print()
        return False


def test_gcs_connection(bucket_name="audio-data-gemini"):
    """Test GCS connection"""
    print("=" * 70)
    print("TEST 4: GCS Connection")
    print("=" * 70)
    print()
    
    try:
        client = GCSClient(bucket_name=bucket_name)
        print(f"✅ GCS client initialized for bucket: {bucket_name}")
        print()
        return True
    except Exception as e:
        print(f"❌ GCS connection failed: {e}")
        print()
        return False


def test_small_sample_processing(
    bucket_name="audio-data-gemini",
    input_prefix="segments_cache/",
    sample_size=10
):
    """
    Test processing a small sample of segments
    
    Args:
        bucket_name: GCS bucket name
        input_prefix: GCS prefix for input segments
        sample_size: Number of files to test with (default: 10)
    """
    print("=" * 70)
    print(f"TEST 5: Small Sample Processing ({sample_size} files)")
    print("=" * 70)
    print()
    
    try:
        # Initialize GCS client
        print("Initializing GCS client...")
        client = GCSClient(bucket_name=bucket_name)
        print("✅ GCS client initialized")
        print()
        
        # List files
        print(f"Listing files in gs://{bucket_name}/{input_prefix}...")
        all_files = client.list_files(prefix=input_prefix, suffix=".npz")
        print(f"Found {len(all_files):,} total files")
        
        if len(all_files) == 0:
            print("❌ No files found in GCS")
            return False
        
        # Select sample
        sample_files = all_files[:sample_size]
        print(f"Testing with {len(sample_files)} files")
        print()
        
        # Setup temp directory
        test_temp_dir = setup_temp_directory(
            base_path="/tmp",
            subdir="neutts_test_sample",
            clean_if_exists=True
        )
        
        # Download sample
        print("Downloading sample files...")
        downloaded_files = []
        for gcs_path in sample_files:
            try:
                relative_path = gcs_path.replace(input_prefix, "")
                local_path = os.path.join(test_temp_dir, relative_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                if client.download_file(gcs_path, local_path, show_progress=False):
                    downloaded_files.append(local_path)
                    print(f"  ✓ {os.path.basename(local_path)}")
            except Exception as e:
                print(f"  ✗ Failed to download {gcs_path}: {e}")
        
        print(f"\n✅ Downloaded {len(downloaded_files)} files")
        print()
        
        # Test loading and processing
        print("Testing codec processing...")
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        
        if device == "cpu":
            print("⚠️  WARNING: CUDA not available, using CPU (slower)")
        
        # Load codec
        from neucodec import NeuCodec
        codec = NeuCodec.from_pretrained("neuphonic/neucodec")
        codec.eval().to(device)
        print("✅ Codec loaded")
        print()
        
        # Process first file as test
        if len(downloaded_files) > 0:
            test_file = downloaded_files[0]
            print(f"Processing test file: {os.path.basename(test_file)}")
            
            try:
                # Load
                data = np.load(test_file, allow_pickle=True)
                print(f"  ✓ Loaded: {data['audio'].shape}")
                
                # Encode
                audio_array = data['audio']
                audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0).unsqueeze(0)
                audio_tensor = audio_tensor.to(device)
                
                with torch.no_grad():
                    vq_codes = codec.encode_code(audio_or_path=audio_tensor)
                    vq_codes = vq_codes.squeeze(0).squeeze(0)
                    if vq_codes.is_cuda:
                        vq_codes = vq_codes.cpu()
                    vq_codes_list = vq_codes.numpy()
                
                print(f"  ✓ Encoded: {vq_codes_list.shape}")
                print("✅ Processing test passed")
                print()
                
                # Cleanup
                del codec
                if device == "cuda":
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ Processing failed: {e}")
                return False
        
        # Cleanup temp directory
        import shutil
        shutil.rmtree(test_temp_dir, ignore_errors=True)
        print("✅ Cleanup complete")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print()
    print("=" * 70)
    print("RAM Disk Processing - Test Suite")
    print("=" * 70)
    print()
    
    results = []
    
    # Test 1: RAM detection
    results.append(("RAM Detection", test_ram_detection()))
    
    # Test 2: Temp location
    results.append(("Temp Location", test_temp_location()))
    
    # Test 3: Temp directory setup
    results.append(("Temp Directory", test_temp_directory_setup()))
    
    # Test 4: GCS connection
    results.append(("GCS Connection", test_gcs_connection()))
    
    # Test 5: Small sample processing (optional - requires GCS setup)
    print("Would you like to test small sample processing? (requires GCS credentials)")
    print("This will download and process 10 sample files.")
    response = input("Run sample processing test? [y/N]: ").strip().lower()
    
    if response == 'y':
        results.append(("Sample Processing", test_small_sample_processing()))
    
    # Summary
    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print()
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
        if not passed:
            all_passed = False
    
    print()
    
    if all_passed:
        print("=" * 70)
        print("🎉 All tests passed!")
        print("=" * 70)
        print()
        print("You're ready to run the full processing pipeline:")
        print()
        print("  python scripts/add_codes_to_dataset_gcs.py \\")
        print("    --gcs_bucket=audio-data-gemini \\")
        print("    --input_prefix=segments_cache/ \\")
        print("    --output_prefix=segments_with_codes/ \\")
        print("    --codec_device=cuda")
        print()
    else:
        print("=" * 70)
        print("⚠️  Some tests failed")
        print("=" * 70)
        print()
        print("Please fix the issues before running the full pipeline.")
        print()
    
    return all_passed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RAM disk processing setup")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only quick tests (skip sample processing)"
    )
    args = parser.parse_args()
    
    if args.quick:
        print("Running quick tests only...")
        print()
        test_ram_detection()
        test_temp_location()
        test_temp_directory_setup()
        test_gcs_connection()
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)


"""
Test script to verify H200 can access GCS bucket.

Run this on H200 machine to test connectivity.
"""

import os
from google.cloud import storage

def test_gcs_access(bucket_name="audio-data-gemini", prefix="audio/"):
    """
    Test GCS access from H200.
    
    Args:
        bucket_name: Your GCS bucket name
        prefix: Optional prefix to list (e.g., "audio/" or "segments_cache/")
    """
    
    print("=" * 70)
    print("Testing GCS Access from H200")
    print("=" * 70)
    print()
    
    # Check for credentials
    creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if creds_path:
        print(f"✅ Using credentials: {creds_path}")
        if os.path.exists(creds_path):
            print(f"✅ Credentials file exists")
        else:
            print(f"❌ Credentials file not found!")
            return False
    else:
        print("⚠️  GOOGLE_APPLICATION_CREDENTIALS not set")
        print("   Will try default credentials...")
    
    print()
    
    try:
        # Initialize client
        print(f"Connecting to GCS bucket: gs://{bucket_name}")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Test bucket access
        if bucket.exists():
            print(f"✅ Bucket '{bucket_name}' found and accessible!")
        else:
            print(f"❌ Bucket '{bucket_name}' not found or not accessible")
            return False
        
        print()
        
        # List some files
        print(f"Listing files with prefix: {prefix}")
        blobs = list(bucket.list_blobs(prefix=prefix, max_results=10))
        
        if blobs:
            print(f"✅ Found {len(blobs)} files (showing first 10):")
            for blob in blobs:
                size_mb = blob.size / (1024 * 1024)
                print(f"   - {blob.name} ({size_mb:.2f} MB)")
        else:
            print(f"⚠️  No files found with prefix '{prefix}'")
            print("   This might be okay if the prefix is wrong")
        
        print()
        print("=" * 70)
        print("✅ GCS Access Test: SUCCESS!")
        print("=" * 70)
        print()
        print("Your H200 can access GCS! You can proceed with:")
        print("1. Upload segments to GCS from GCP T4 (if not already there)")
        print("2. Download from GCS to H200")
        print("3. Process on H200")
        print("4. Optionally upload to S3 for backup")
        print("5. Push final dataset to HuggingFace Hub")
        
        return True
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ GCS Access Test: FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("1. Set credentials: export GOOGLE_APPLICATION_CREDENTIALS='/path/to/key.json'")
        print("2. Verify JSON key has Storage permissions")
        print("3. Check bucket name is correct")
        print("4. Verify H200 has internet access")
        
        return False


if __name__ == "__main__":
    import sys
    
    # Allow command-line arguments
    bucket = sys.argv[1] if len(sys.argv) > 1 else "audio-data-gemini"
    prefix = sys.argv[2] if len(sys.argv) > 2 else "audio/"
    
    print(f"Usage: python test_h200_gcs_access.py [bucket_name] [prefix]")
    print(f"Testing with: bucket='{bucket}', prefix='{prefix}'")
    print()
    
    test_gcs_access(bucket, prefix)


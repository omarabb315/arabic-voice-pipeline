# S3 Setup Guide

Quick reference for setting up S3 credentials and testing connectivity.

## Environment Variables

### On GCP T4 (for uploading)

```bash
# Add to ~/.bashrc or set in terminal
export S3_ENDPOINT_URL="https://AJN-NAS-01.ALJAZEERA.TV:9021"
export S3_ACCESS_KEY="your-access-key-here"
export S3_SECRET_KEY="your-secret-key-here"
export S3_BUCKET_NAME="aifiles"

# Reload
source ~/.bashrc
```

### On H200 (for downloading/processing)

```bash
# Add to ~/.bashrc or set in terminal
export S3_ENDPOINT_URL="https://AJN-NAS-01.ALJAZEERA.TV:9021"
export S3_ACCESS_KEY="your-access-key-here"
export S3_SECRET_KEY="your-secret-key-here"
export S3_BUCKET_NAME="aifiles"

# Reload
source ~/.bashrc
```

## Test Connection

```bash
# Test S3 connectivity
python -c "from scripts.s3_utils import test_connection; test_connection()"

# Expected output:
# ✅ Connection successful! Found X files in bucket.
```

## Common Issues

### SSL Certificate Verification Error

**Problem**: `SSL: CERTIFICATE_VERIFY_FAILED`

**Solution**: The S3 client is already configured with `verify_ssl=False` for self-signed certificates. No action needed.

### Connection Timeout

**Problem**: Cannot connect to endpoint

**Solution**:
1. Check firewall rules
2. Verify endpoint URL is correct (including https/http and port)
3. Try from different network if VPN is required

### Access Denied

**Problem**: `403 Forbidden` or `Access Denied`

**Solution**:
1. Verify credentials are correct
2. Check bucket permissions
3. Ensure bucket name is correct

## Manual Testing with boto3

```python
import boto3
import os

# Create client
s3 = boto3.client(
    's3',
    endpoint_url=os.environ['S3_ENDPOINT_URL'],
    aws_access_key_id=os.environ['S3_ACCESS_KEY'],
    aws_secret_access_key=os.environ['S3_SECRET_KEY'],
    verify=False  # For self-signed certificates
)

# List objects
bucket = os.environ['S3_BUCKET_NAME']
response = s3.list_objects_v2(Bucket=bucket, MaxKeys=10)

print(f"Bucket: {bucket}")
print(f"Files found: {response.get('KeyCount', 0)}")

if 'Contents' in response:
    for obj in response['Contents'][:5]:
        print(f"  - {obj['Key']} ({obj['Size']} bytes)")
```

## S3 Directory Structure

```
s3://aifiles/
└── neutts_segments/
    ├── stage1/                    # Raw segments (no codes)
    │   ├── ZadTVchannel/
    │   │   ├── audio_123_seg_0.npz
    │   │   └── ...
    │   ├── Atheer_-/
    │   └── thmanyahPodcasts/
    │
    └── stage2_with_codes/         # Encoded segments (with VQ codes)
        ├── ZadTVchannel/
        ├── Atheer_-/
        └── thmanyahPodcasts/
```

## Quick Commands Reference

```bash
# Test connection
python -c "from scripts.s3_utils import test_connection; test_connection()"

# Upload segments (T4)
python scripts/upload_to_s3.py \
  --source_dir=segments_cache/ \
  --s3_prefix=neutts_segments/stage1/

# Add codes (H200)
python scripts/add_codes_to_dataset_s3.py \
  --s3_prefix=neutts_segments/stage1/ \
  --output_prefix=neutts_segments/stage2_with_codes/ \
  --batch_size=70000

# Download encoded segments (T4)
python scripts/download_from_s3.py \
  --s3_prefix=neutts_segments/stage2_with_codes/ \
  --output_dir=segments_cache_encoded/

# Create HF dataset from S3 (H200)
python create_hf_dataset_from_npz.py \
  --s3_prefix=neutts_segments/stage2_with_codes/ \
  --push_to_hub=True \
  --hf_repo=omarabb315/arabic-voice-cloning-dataset
```

## Security Best Practices

1. **Never commit credentials** - They're excluded in `.gitignore`
2. **Use environment variables** - Don't hardcode in scripts
3. **Rotate keys periodically** - Update access keys regularly
4. **Restrict permissions** - Only give access to needed buckets/paths
5. **Use secure networks** - VPN when accessing from public networks


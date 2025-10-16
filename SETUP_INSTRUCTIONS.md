# Setup Instructions

Complete guide for setting up the NeuTTS dataset pipeline on different machines.

## Table of Contents

1. [GitHub Repository Setup](#github-repository-setup)
2. [GCP T4 Setup (Stage 1)](#gcp-t4-setup-stage-1)
3. [H200 Setup (Stage 2)](#h200-setup-stage-2)
4. [HuggingFace Setup](#huggingface-setup)

---

## GitHub Repository Setup

### Create Repository

1. Go to GitHub: https://github.com/new
2. Create repository named `neutts` (or your preferred name)
3. Set visibility (public or private)
4. **Do NOT** initialize with README (we already have one)

### Push Existing Code

```bash
cd /home/ubuntu/work/neutts

# Install git if needed
# sudo apt-get update && sudo apt-get install -y git

# Initialize repository
git init

# Configure git (use your details)
git config user.email "your-email@example.com"
git config user.name "Your Name"

# Add files
git add .

# Commit
git commit -m "Initial commit: Complete dataset pipeline

- Stage 1: Build segments from audio (GCP T4)
- Stage 2: Add VQ codes with batch processing (H200)
- S3 integration for cross-machine data transfer
- HuggingFace Hub support
- 656K segments across 3 channels (221GB)"

# Add remote (replace with your username)
git remote add origin https://github.com/your-username/neutts.git

# Push
git branch -M main
git push -u origin main
```

---

## GCP T4 Setup (Stage 1)

### 1. Install Dependencies

```bash
cd /home/ubuntu/work/neutts
source venv_voicecloning/bin/activate
pip install boto3
```

### 2. Set Environment Variables

```bash
# GCS credentials (already set if working)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# S3 credentials (new)
export S3_ENDPOINT_URL="https://AJN-NAS-01.ALJAZEERA.TV:9021"
export S3_ACCESS_KEY="your-access-key"
export S3_SECRET_KEY="your-secret-key"
export S3_BUCKET_NAME="aifiles"

# Make permanent (optional)
echo 'export S3_ENDPOINT_URL="https://AJN-NAS-01.ALJAZEERA.TV:9021"' >> ~/.bashrc
echo 'export S3_ACCESS_KEY="your-access-key"' >> ~/.bashrc
echo 'export S3_SECRET_KEY="your-secret-key"' >> ~/.bashrc
echo 'export S3_BUCKET_NAME="aifiles"' >> ~/.bashrc
source ~/.bashrc
```

### 3. Test S3 Connection

```bash
python -c "from scripts.s3_utils import test_connection; test_connection()"
```

Expected output:
```
✅ Connection successful! Found X files in bucket.
```

### 4. Upload Segments to S3

```bash
# Upload all segments (if not already done)
python scripts/upload_to_s3.py \
  --source_dir=segments_cache/ \
  --s3_prefix=neutts_segments/stage1/ \
  --max_workers=4

# Time estimate: 30-60 minutes for 221GB
# Progress will be shown with overall completion percentage
```

---

## H200 Setup (Stage 2)

### 1. Clone Repository

```bash
# Navigate to your work directory
cd ~

# Clone repository (replace with your username)
git clone https://github.com/your-username/neutts.git
cd neutts
```

### 2. Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Expected installations:
- torch (with CUDA support)
- neucodec
- boto3
- datasets
- transformers
- tqdm, fire, etc.

### 3. Set S3 Credentials

```bash
# Set environment variables
export S3_ENDPOINT_URL="https://AJN-NAS-01.ALJAZEERA.TV:9021"
export S3_ACCESS_KEY="your-access-key"
export S3_SECRET_KEY="your-secret-key"
export S3_BUCKET_NAME="aifiles"

# Make permanent
echo 'export S3_ENDPOINT_URL="https://AJN-NAS-01.ALJAZEERA.TV:9021"' >> ~/.bashrc
echo 'export S3_ACCESS_KEY="your-access-key"' >> ~/.bashrc
echo 'export S3_SECRET_KEY="your-secret-key"' >> ~/.bashrc
echo 'export S3_BUCKET_NAME="aifiles"' >> ~/.bashrc
source ~/.bashrc
```

### 4. Test Connection

```bash
python -c "from scripts.s3_utils import test_connection; test_connection()"
```

### 5. Verify GPU

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

Expected output:
```
CUDA available: True
Device: NVIDIA H200
```

### 6. Run Stage 2 (Add VQ Codes)

```bash
# Recommended: 70K segments per batch
python scripts/add_codes_to_dataset_s3.py \
  --s3_prefix=neutts_segments/stage1/ \
  --output_prefix=neutts_segments/stage2_with_codes/ \
  --batch_size=70000 \
  --temp_dir=/tmp/neutts_batch/ \
  --codec_device=cuda

# Time estimate: 2-3 hours for 656K segments in ~10 batches
```

**Monitor progress:**
- Watch for batch completion messages
- Progress saved to `processing_progress.json`
- Can interrupt and resume anytime

**If interrupted:**
```bash
# Just re-run the same command - it will resume!
python scripts/add_codes_to_dataset_s3.py \
  --s3_prefix=neutts_segments/stage1/ \
  --output_prefix=neutts_segments/stage2_with_codes/
```

---

## HuggingFace Setup

### 1. Install HuggingFace CLI (if not present)

```bash
pip install huggingface-hub
```

### 2. Login

```bash
huggingface-cli login
```

Enter your HuggingFace access token when prompted.
- Get token from: https://huggingface.co/settings/tokens
- Recommended: Create token with "write" permissions

### 3. Create HuggingFace Dataset

**On H200 (stream from S3):**
```bash
python create_hf_dataset_from_npz.py \
  --s3_prefix=neutts_segments/stage2_with_codes/ \
  --output_path=/tmp/segments_dataset/ \
  --push_to_hub=True \
  --hf_repo=omarabb315/arabic-voice-cloning-dataset
```

**On GCP T4 (download first):**
```bash
# Download encoded segments
python scripts/download_from_s3.py \
  --s3_prefix=neutts_segments/stage2_with_codes/ \
  --output_dir=segments_cache_encoded/

# Create and push dataset
python create_hf_dataset_from_npz.py \
  --segments_cache_dir=segments_cache_encoded/ \
  --output_path=dataset_cache/final/ \
  --push_to_hub=True \
  --hf_repo=omarabb315/arabic-voice-cloning-dataset
```

### 4. Access Dataset

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("omarabb315/arabic-voice-cloning-dataset")

# Explore
print(dataset)
print(dataset['train'][0])
```

---

## Verification Checklist

### Stage 1 (GCP T4) ✅
- [ ] Segments built (656K .npz files)
- [ ] S3 credentials configured
- [ ] S3 connection tested
- [ ] Segments uploaded to S3
- [ ] Verified in S3: `neutts_segments/stage1/`

### Stage 2 (H200) ⏳
- [ ] Repository cloned
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] S3 credentials configured
- [ ] GPU verified (CUDA available)
- [ ] Stage 2 processing complete
- [ ] Verified in S3: `neutts_segments/stage2_with_codes/`

### Final Step 🎯
- [ ] HuggingFace CLI installed
- [ ] Logged in to HuggingFace
- [ ] Dataset created
- [ ] Dataset pushed to Hub
- [ ] Dataset accessible at: `omarabb315/arabic-voice-cloning-dataset`

---

## Troubleshooting

### Problem: Git not installed on GCP T4

```bash
sudo apt-get update
sudo apt-get install -y git
```

### Problem: S3 connection timeout

**Check:**
1. Endpoint URL is correct (including https and port)
2. Firewall allows connection
3. Credentials are valid

**Test manually:**
```python
import boto3
s3 = boto3.client('s3', endpoint_url='https://AJN-NAS-01.ALJAZEERA.TV:9021', ...)
```

### Problem: Out of space on H200

**Solutions:**
1. Use smaller batch size: `--batch_size=50000`
2. Check disk usage: `df -h`
3. Clean temp directory: `rm -rf /tmp/neutts_batch/`

### Problem: CUDA out of memory

**Solutions:**
1. Clear GPU cache: `torch.cuda.empty_cache()`
2. Reduce batch size
3. Check GPU memory: `nvidia-smi`

### Problem: HuggingFace push fails

**Solutions:**
1. Check login: `huggingface-cli whoami`
2. Re-login: `huggingface-cli login`
3. Verify token has write permissions
4. Check internet connectivity

---

## Next Steps

After completing setup:

1. **Add more channels** (Stage 1 on GCP T4)
2. **Process new channels** (Stage 2 on H200)
3. **Start training** with the dataset
4. **Share dataset** with team via HuggingFace Hub

See [DATASET_BUILD_WORKFLOW.md](docs/DATASET_BUILD_WORKFLOW.md) for detailed workflow.


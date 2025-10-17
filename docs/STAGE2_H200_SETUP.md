# Stage 2: Adding VQ Codes on H200 Machine (RAM-Optimized)

This guide explains how to process all segments on your H200 machine using 2TB RAM for optimal performance.

## Overview

**Stage 1 (T4 - Completed):** Built audio + metadata segments (~221GB for 656K segments)  
**Stage 2 (H200 - This guide):** Add VQ codes using RAM-optimized processing (~3-5 hours on H200)

### Key Features

- **RAM-Optimized:** Uses 2TB RAM to process all 656K segments at once
- **No Disk Limitation:** Bypasses 100GB disk storage limit using `/tmp` with OS caching
- **Resume Capability:** Three-phase checkpointing survives interruptions
- **Fast I/O:** Automatic RAM caching provides 10-100x faster I/O vs disk

---

## Prerequisites

### Hardware Requirements

- **RAM:** 2TB total (1.9TB+ available recommended for 656K segments)
- **Disk:** 100GB+ for code and logs (data stored in RAM)
- **GPU:** H200 or similar CUDA-capable GPU
- **Network:** Good bandwidth for GCS downloads/uploads

### Software Requirements

- Python 3.8+
- CUDA 12.4+ (for H200)
- GCS credentials with read/write access

---

## Step 1: Setup H200 Machine

### 1.1 Install Dependencies

```bash
# Clone repository
git clone https://github.com/omarabb315/arabic-voice-pipeline.git
cd arabic-voice-pipeline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 1.2 Setup GCS Access

```bash
# Install gcloud SDK (if not already installed)
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Or just install gsutil
sudo apt-get install google-cloud-sdk
```

### 1.3 Configure GCS Credentials

Transfer the GCS credentials file to H200:

```bash
# Option 1: SCP from local machine
scp your-service-account-key.json user@h200-machine:/path/to/

# Option 2: Create on H200 directly (paste content)
nano ~/gcs-credentials.json
```

Set environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-service-account-key.json"

# Add to ~/.bashrc for persistence
echo 'export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-service-account-key.json"' >> ~/.bashrc
source ~/.bashrc
```

---

## Step 2: Verify RAM and System Resources

Before running the full pipeline, verify your system has sufficient resources:

```bash
# Check RAM
free -h

# You should see:
# Total: ~2TB
# Available: ~1.9TB+ (need at least 300GB for 221GB data + overhead)

# Check /dev/shm size
df -h /dev/shm

# Test RAM detection
python test_ram_disk_processing.py --quick
```

**Expected Output:**
```
Total RAM: 2048.0 GB
Available RAM: 1977.0 GB
✅ Sufficient RAM detected
```

---

## Step 3: Test with Small Sample (Recommended)

Before processing all 656K segments, test with a small sample:

```bash
# Run full test suite (interactive)
python test_ram_disk_processing.py

# This will:
# 1. Detect RAM and temp storage
# 2. Test GCS connection  
# 3. Optionally process 10 sample files
```

If all tests pass, you're ready for the full run!

---

## Step 4: Run Full Processing Pipeline

The new RAM-optimized script processes all segments in three phases:

### 4.1 Start Processing in tmux

```bash
# Start tmux session (important for long-running jobs)
tmux new-session -s add_codes

# Activate environment
source venv/bin/activate

# Run RAM-optimized processing
python scripts/add_codes_to_dataset_gcs.py \
  --gcs_bucket=audio-data-gemini \
  --input_prefix=segments_cache/ \
  --output_prefix=segments_with_codes/ \
  --codec_device=cuda \
  --checkpoint_interval=5000

# If you want to train on the same machine, add --skip_cleanup=True
# This keeps the processed data instead of deleting it:
# --skip_cleanup=True \
# --temp_dir=/workspace/segments_with_codes

# Detach: Ctrl+B then D
# Reattach: tmux attach -s add_codes
```

### 4.2 What Happens During Processing

The script runs in three phases with automatic checkpointing:

**Phase 1: Download All Segments** (~30-60 min)
- Downloads all 656K segments (~221GB) from GCS to `/tmp/neutts_processing/`
- Linux automatically caches in RAM (with 1.9TB available)
- Checkpoints every 5,000 files

**Phase 2: Process All Segments** (~2-4 hours)
- Loads NeuCodec on H200 GPU
- Encodes all segments with VQ codes
- Saves processed files in `/tmp` (still in RAM cache)
- Checkpoints every 5,000 files

**Phase 3: Upload All Segments** (~30-60 min)
- Uploads all processed segments to GCS `segments_with_codes/` prefix
- Checkpoints every 5,000 files
- Cleans up temporary files after successful upload

### 4.3 Monitor Progress

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check RAM usage
watch -n 1 'free -h'

# Reattach to see progress
tmux attach -s add_codes
```

### 4.4 Resume After Interruption

If the process is interrupted (network issue, OOM, manual stop), simply re-run the same command:

```bash
python scripts/add_codes_to_dataset_gcs.py \
  --gcs_bucket=audio-data-gemini \
  --input_prefix=segments_cache/ \
  --output_prefix=segments_with_codes/ \
  --codec_device=cuda
```

The script automatically:
- Detects completed phases
- Skips already processed files
- Resumes from the last checkpoint

**Expected Total Time:** 3-5 hours for 656K segments (vs 10-15 hours with batch processing)

---

## Special: Training on Same Machine (RunPod/Cloud)

If you're running on a cloud instance (RunPod, AWS, etc.) and want to train on the **same machine** after processing:

### Use Persistent Storage + Skip Cleanup

```bash
# Use a persistent directory and skip cleanup
python scripts/add_codes_to_dataset_gcs.py \
  --gcs_bucket=audio-data-gemini \
  --input_prefix=segments_cache/ \
  --output_prefix=segments_with_codes/ \
  --codec_device=cuda \
  --temp_dir=/workspace/segments_with_codes \
  --skip_cleanup=True

# Data will be kept at /workspace/segments_with_codes/
# Ready for immediate training!
```

**Benefits:**
- ✅ No need to re-download from GCS for training
- ✅ Saves time and bandwidth
- ✅ Data already on fast local disk
- ✅ Can start training immediately after processing

**After processing completes:**

```bash
# Create training pairs
python training/create_pairs.py \
  --segments_path=/workspace/segments_with_codes/

# Start training
python training/finetune-neutts.py \
  --dataset_path=/workspace/paired_dataset/
```

**Cleanup later (if needed):**

```bash
# After training is done and you've saved checkpoints:
rm -rf /workspace/segments_with_codes/
rm processing_progress.json
```

---

## Step 5: Next Steps - Create HuggingFace Dataset

After all segments are processed and uploaded to GCS, create a HuggingFace dataset:

```bash
# Login to HuggingFace
huggingface-cli login

# Create and push dataset (streams from GCS, no local download needed)
python create_hf_dataset_from_npz.py \
  --gcs_bucket=audio-data-gemini \
  --gcs_prefix=segments_with_codes/ \
  --output_path=/tmp/segments_dataset/ \
  --push_to_hub=True \
  --hf_repo=omarabb315/arabic-voice-cloning-dataset
```

The final dataset contains:

- ✅ Audio (raw waveforms, 16kHz)
- ✅ VQ Codes (encoded representations for training)
- ✅ Text (transcriptions)
- ✅ Metadata (speaker_id, channel, gender, dialect, tone, duration)

---

## Troubleshooting

### Issue: Insufficient RAM

If you see warnings about insufficient RAM:

```bash
# Check available RAM
free -h

# If less than 300GB available, try:
# 1. Close other applications
# 2. Wait for other jobs to finish (shared server)
# 3. Contact admin to free up resources
```

### Issue: /tmp disk full

The script uses `/tmp` with OS RAM caching, but if `/tmp` has size limits:

```bash
# Check /tmp disk usage
df -h /tmp

# Solution: Script automatically uses RAM caching
# With 1.9TB available RAM, Linux will cache everything in memory
```

### Issue: "gsutil: command not found"
```bash
# Install gcloud SDK
curl https://sdk.cloud.google.com | bash
# or
sudo apt-get install google-cloud-sdk
```

### Issue: "Permission denied" on GCS
```bash
# Verify credentials
gcloud auth activate-service-account --key-file=/path/to/key.json
gsutil ls gs://audio-data-gemini/  # Test access

# Check environment variable
echo $GOOGLE_APPLICATION_CREDENTIALS
```

### Issue: Process interrupted (network, OOM, etc.)

Simply re-run the same command - it automatically resumes:

```bash
python scripts/add_codes_to_dataset_gcs.py \
  --gcs_bucket=audio-data-gemini \
  --input_prefix=segments_cache/ \
  --output_prefix=segments_with_codes/ \
  --codec_device=cuda
```

The script checks:
- What's already uploaded to GCS (skips those)
- What's been processed locally (continues upload)
- What's been downloaded (continues processing)

### Issue: CUDA out of memory

```bash
# Check GPU memory
nvidia-smi

# If other processes are using GPU, wait or:
# Use CPU (slower but works)
python scripts/add_codes_to_dataset_gcs.py \
  --codec_device=cpu \
  --gcs_bucket=audio-data-gemini \
  --input_prefix=segments_cache/ \
  --output_prefix=segments_with_codes/
```

---

## Performance Comparison

### Old Batch Approach (100GB disk limit)
- **Download:** 70K batches (~24GB each), ~10 iterations
- **Process:** Each batch separately
- **Upload:** Each batch separately
- **Time:** 10-15 hours total
- **Disk:** 24GB per batch, cleanup between batches

### New RAM-Optimized Approach (2TB RAM)
- **Download:** All 656K segments at once (~221GB) to `/tmp`
- **Process:** All segments in one pass (RAM caching)
- **Upload:** All segments at once
- **Time:** 3-5 hours total (50-70% faster!)
- **Disk:** <10GB (logs only, data in RAM)

### Time Breakdown (656K segments)

| Phase | Old Approach | New Approach | Improvement |
|-------|--------------|--------------|-------------|
| Download | 10×(10 min) = 100 min | 30-60 min | 40-70% faster |
| Process | 10×(40 min) = 400 min | 120-240 min | 40-70% faster |
| Upload | 10×(10 min) = 100 min | 30-60 min | 40-70% faster |
| **Total** | **600-900 min (10-15 hrs)** | **180-360 min (3-6 hrs)** | **50-70% faster** |

### Why It's Faster

1. **No batch overhead:** Download/upload once instead of 10 times
2. **RAM I/O:** 10-100x faster than disk I/O
3. **No cleanup between batches:** Process continuously
4. **Better caching:** OS optimizes RAM usage automatically

---

## System Requirements Summary

| Resource | Minimum | Recommended | Your H200 |
|----------|---------|-------------|-----------|
| RAM | 300 GB available | 500 GB available | 1.9 TB ✅ |
| Disk | 100 GB | 200 GB | 100 GB ✅ |
| GPU VRAM | 40 GB | 80 GB | 141 GB (H200) ✅ |
| Network | 100 Mbps | 1 Gbps | - |

**Your H200 is perfectly suited for this workflow!** 🚀


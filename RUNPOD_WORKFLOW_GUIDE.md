# RunPod Instance Workflow Guide - Arabic Voice Cloning Pipeline

**Date:** October 17, 2025  
**Repository:** https://github.com/omarabb315/arabic-voice-pipeline  
**Purpose:** Complete workflow for processing 656K Arabic audio segments on RunPod H100 instance

---

## 📋 Overview

This document guides you through processing Arabic voice cloning segments on a RunPod instance with the following workflow:

```
GCS (input) → Download → Process (Add VQ Codes) → Upload to GCS → Create HF Dataset → Train
```

**Total Time:** ~4-6 hours (2-3 hours processing + 1-2 hours HF dataset + training time)

---

## 🖥️ RunPod Instance Requirements

### Required Specs:
- **GPU:** H100 80GB (or A100 80GB)
- **RAM:** 128GB+ (more is better, but not critical with 500GB disk)
- **Disk:** 500GB NVMe (minimum) - 1TB recommended
- **Region:** US (for fast GCS access)

### Why These Specs:
- **500GB disk:** Holds all 656K segments (~221GB) + paired dataset (~100GB) + models
- **H100 GPU:** Fast VQ encoding (~2-3 hours vs 10+ hours on slower GPUs)
- **Good network:** Cloud-to-cloud GCS transfer (fast)

---

## 📊 Dataset Information

### Current State:
- **Location:** `gs://audio-data-gemini/segments_cache/`
- **Total Segments:** 656,300 segments
- **Total Size:** ~221GB (raw .npz files)
- **Duration:** ~218 hours
- **Channels:** ZadTVchannel, Atheer, thmanyahPodcasts
- **Status:** Stage 1 complete (audio + metadata), needs Stage 2 (VQ codes)

### What Will Be Created:
- **Processed Segments:** `gs://audio-data-gemini/segments_with_codes/` (~221GB)
- **HuggingFace Dataset:** `omarabb315/arabic-voice-cloning-dataset` (~60GB optimized)
- **Paired Training Dataset:** Local on RunPod for training

---

## 🔑 Required Credentials

### GCS (Google Cloud Storage):
- **File:** Service account JSON key
- **Location:** Upload to RunPod as `~/gcs-credentials.json`
- **Environment Variable:** `GOOGLE_APPLICATION_CREDENTIALS`

### HuggingFace Hub:
- **Token:** Get from https://huggingface.co/settings/tokens
- **Login:** `huggingface-cli login`
- **Permissions:** Write access to create datasets

---

## 🚀 Complete Workflow

### Step 1: Initial Setup (5-10 minutes)

```bash
# SSH into RunPod
ssh root@<runpod-ip>

# Clone repository
cd ~
git clone https://github.com/omarabb315/arabic-voice-pipeline.git
cd arabic-voice-pipeline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Install gcloud for GCS access
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

### Step 2: Setup Credentials

```bash
# Upload GCS credentials (do this from your local machine)
# On local machine:
scp /path/to/gcs-key.json root@<runpod-ip>:~/gcs-credentials.json

# On RunPod, set environment variable
export GOOGLE_APPLICATION_CREDENTIALS=~/gcs-credentials.json
echo 'export GOOGLE_APPLICATION_CREDENTIALS=~/gcs-credentials.json' >> ~/.bashrc

# Test GCS access
gsutil ls gs://audio-data-gemini/segments_cache/ | head -5

# Login to HuggingFace
huggingface-cli login
# Paste your HF token when prompted
```

### Step 3: Verify System Resources

```bash
# Check GPU
nvidia-smi

# Check disk space (need ~500GB free)
df -h /workspace

# Check RAM
free -h

# Test the setup
cd ~/arabic-voice-pipeline
python test_ram_disk_processing.py --quick
```

**Expected Output:**
```
Total RAM: 128.0 GB
Available RAM: 120.0 GB
✅ Sufficient RAM detected

/tmp:
   Space: 450.0 GB available / 500.0 GB total
   ✅ Selected: /tmp with sufficient space
```

### Step 4: Process All Segments (2-3 hours)

This is the main processing step that:
1. Downloads all 656K segments from GCS (~30-60 min)
2. Adds VQ codes using NeuCodec on H100 (~1.5-2.5 hours)
3. Uploads all processed segments back to GCS (~30-60 min)

```bash
# Navigate to project
cd ~/arabic-voice-pipeline
source venv/bin/activate

# Run processing with nohup (survives disconnection)
nohup python scripts/add_codes_to_dataset_gcs.py \
  --gcs_bucket=audio-data-gemini \
  --input_prefix=segments_cache/ \
  --output_prefix=segments_with_codes/ \
  --codec_device=cuda \
  --temp_dir=/workspace/segments_with_codes \
  --skip_cleanup=True \
  --checkpoint_interval=5000 > processing.log 2>&1 &

# Save the process ID
echo $! > processing.pid

# Monitor progress
tail -f processing.log

# To detach and reconnect later: Ctrl+C then tail -f processing.log
```

**Key Parameters Explained:**
- `--skip_cleanup=True`: Keeps processed data locally for training (saves re-download)
- `--temp_dir=/workspace/segments_with_codes`: Uses persistent workspace storage
- `--checkpoint_interval=5000`: Saves progress every 5,000 files (~2-3 minutes)

**Monitoring Progress:**

```bash
# Check progress file
cat processing_progress.json | python -m json.tool | head -20

# Watch GPU usage
watch -n 1 nvidia-smi

# Watch disk usage
watch -n 1 'df -h /workspace'

# Check if still running
ps aux | grep add_codes_to_dataset_gcs.py
```

**Expected Output:**
```
======================================================================
PHASE 1: Downloading All Segments from GCS
======================================================================
Target: 656,300 files (~221.0 GB)
Downloading...
✅ Download complete: 656,300 files

======================================================================
PHASE 2: Processing All Segments (Adding VQ Codes)
======================================================================
Loading NeuCodec on cuda...
✅ Codec loaded on cuda
Processing 656,300 remaining files...
Encoding: 100%|████████████| 656300/656300
✅ Processing complete: 656,300 encoded

======================================================================
PHASE 3: Uploading All Processed Segments to GCS
======================================================================
Uploading...
✅ Upload complete: 656,300 files

======================================================================
✅ Stage 2 Complete!
======================================================================
Total files processed: 656,300
Total files uploaded: 656,300
```

### Step 5: Create HuggingFace Dataset (1-2 hours)

After processing completes, create and push the HuggingFace dataset:

```bash
# Make sure you're in the project directory
cd ~/arabic-voice-pipeline
source venv/bin/activate

# Create HF dataset from local processed segments
python create_hf_dataset_from_npz.py \
  --segments_cache_dir=/workspace/segments_with_codes \
  --output_path=/workspace/hf_dataset \
  --push_to_hub=True \
  --hf_repo=omarabb315/arabic-voice-cloning-dataset

# This will:
# 1. Load all 656K .npz files from local storage
# 2. Convert to HuggingFace dataset format
# 3. Save locally to /workspace/hf_dataset
# 4. Push to HuggingFace Hub
```

**Expected Output:**
```
Creating HuggingFace Dataset from .npz Files
Source: Local (/workspace/segments_with_codes)
Found 656,300 .npz files
Loading segments...
Creating HuggingFace Dataset with 656,300 segments...
Saving to /workspace/hf_dataset...
✅ Dataset created successfully!

📊 Dataset info:
   - Total segments: 656,300
   - Channels: {'ZadTVchannel', 'Atheer', 'thmanyahPodcasts'}
   - Unique speakers: 12,450
   - Total duration: 218.33 hours
   
📤 Pushing to HuggingFace Hub: omarabb315/arabic-voice-cloning-dataset
✅ Dataset uploaded to https://huggingface.co/datasets/omarabb315/arabic-voice-cloning-dataset
```

**Your dataset is now public at:** https://huggingface.co/datasets/omarabb315/arabic-voice-cloning-dataset

### Step 6: Create Training Pairs (30-60 minutes)

Create paired dataset for training (reference/target pairs):

```bash
cd ~/arabic-voice-pipeline
source venv/bin/activate

# Create pairs directly from .npz files (efficient, no HF dataset needed)
python training/create_pairs.py \
  --segments_path=/workspace/segments_with_codes \
  --output_path=/workspace/paired_dataset \
  --min_duration_sec=0.5 \
  --max_duration_sec=15.0 \
  --min_segments_per_speaker=2

# This auto-detects .npz format and loads directly
```

**Expected Output:**
```
Auto-detected input format: npz
Loading segments from .npz files in /workspace/segments_with_codes...
Found 656,300 .npz files
Loaded 656,300 valid segments

Grouping segments by (channel, speaker_id)...
Found 12,450 unique (channel, speaker_id) combinations

Filtering speakers with < 2 segments...
Valid speakers: 11,800

Creating reference/target pairs...
============================================================
Total pairs created: 2,450,000
============================================================

✅ Paired dataset saved to: /workspace/paired_dataset
```

### Step 7: Start Training

```bash
cd ~/arabic-voice-pipeline
source venv/bin/activate

# Start training
python training/finetune-neutts.py \
  --dataset_path=/workspace/paired_dataset \
  --save_root=/workspace/checkpoints \
  --run_name=arabic_voice_cloning_v1 \
  --lr=1.0e-5 \
  --max_steps=10000 \
  --per_device_train_batch_size=4

# Training will run for hours/days depending on settings
# Checkpoints saved to /workspace/checkpoints/
```

---

## 🔄 Resume After Interruption

If the process is interrupted at any point:

### For Processing (Step 4):
```bash
# Just re-run the same command - it automatically resumes!
cd ~/arabic-voice-pipeline
source venv/bin/activate

nohup python scripts/add_codes_to_dataset_gcs.py \
  --gcs_bucket=audio-data-gemini \
  --input_prefix=segments_cache/ \
  --output_prefix=segments_with_codes/ \
  --codec_device=cuda \
  --temp_dir=/workspace/segments_with_codes \
  --skip_cleanup=True > processing.log 2>&1 &
```

The script checks:
- What's already uploaded to GCS (skips those)
- What's been processed locally (just uploads)
- What's been downloaded (processes and uploads)

### For Training:
Training automatically saves checkpoints. Restart with same command and it resumes from last checkpoint.

---

## 📁 Directory Structure After Processing

```
/workspace/
├── segments_with_codes/      # 221GB - Processed .npz files with VQ codes
│   ├── ZadTVchannel/
│   ├── Atheer/
│   └── thmanyahPodcasts/
├── hf_dataset/                # ~60GB - HuggingFace dataset format
├── paired_dataset/            # ~100GB - Training pairs
├── checkpoints/               # Training checkpoints (grows over time)
│   └── arabic_voice_cloning_v1/
└── arabic-voice-pipeline/     # Git repository
    ├── scripts/
    ├── training/
    └── ...
```

**Total Disk Usage:** ~400-450GB (fits in 500GB instance)

---

## 🐛 Troubleshooting

### Issue: GCS connection fails

```bash
# Test GCS access
gsutil ls gs://audio-data-gemini/

# Check credentials
echo $GOOGLE_APPLICATION_CREDENTIALS
cat $GOOGLE_APPLICATION_CREDENTIALS | head -5

# Re-authenticate
gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
```

### Issue: Process killed (OOM)

```bash
# Check what killed it
dmesg | tail -50

# If OOM, the script will resume from last checkpoint
# Just re-run the same command
```

### Issue: Disk full

```bash
# Check disk usage
df -h /workspace

# Clean up if needed
rm -rf /workspace/hf_dataset  # After pushed to HuggingFace
rm -rf /tmp/*                  # Clear temp files

# Or provision larger instance (1TB disk)
```

### Issue: GPU not found

```bash
# Check GPU
nvidia-smi

# If not found, use CPU (slower)
python scripts/add_codes_to_dataset_gcs.py \
  --codec_device=cpu \
  --gcs_bucket=audio-data-gemini \
  ...
```

### Issue: HuggingFace push fails

```bash
# Check if logged in
huggingface-cli whoami

# Re-login if needed
huggingface-cli login

# Make sure token has write permissions
# Create new token at: https://huggingface.co/settings/tokens
```

---

## 💰 Cost Optimization Tips

### 1. Use Spot/Interruptible Instances
- RunPod spot instances are 70-80% cheaper
- Resume capability makes this safe

### 2. Separate Processing and Training
```bash
# Day 1: Process on H100 (fast, 2-3 hours)
# Terminate instance after upload to GCS

# Day 2+: Train on cheaper A100/A6000 (days)
# Download from GCS to new instance
```

### 3. Monitor Costs
- H100: ~$2-3/hour
- Processing time: 2-3 hours = $6-9
- Training time: Hours to days (depends on your needs)

### 4. Clean Up After Completion
```bash
# After training finishes and you've saved models:
# Keep checkpoints, delete large datasets
rm -rf /workspace/segments_with_codes  # 221GB
rm -rf /workspace/hf_dataset           # 60GB (already on HF Hub)
rm -rf /workspace/paired_dataset       # 100GB (can recreate if needed)
```

---

## ✅ Success Criteria

After completing all steps, you should have:

1. ✅ All 656K segments processed with VQ codes
2. ✅ Segments uploaded to: `gs://audio-data-gemini/segments_with_codes/`
3. ✅ HuggingFace dataset at: https://huggingface.co/datasets/omarabb315/arabic-voice-cloning-dataset
4. ✅ Paired dataset ready for training: `/workspace/paired_dataset/`
5. ✅ Training started/completed with checkpoints in `/workspace/checkpoints/`

---

## 🔗 Quick Reference Commands

```bash
# Check processing progress
tail -f processing.log
cat processing_progress.json | python -m json.tool

# Monitor GPU
watch -n 1 nvidia-smi

# Monitor disk
df -h /workspace

# Check if process is running
ps aux | grep python

# Kill process if needed
kill $(cat processing.pid)

# Resume processing (if stopped)
cd ~/arabic-voice-pipeline && source venv/bin/activate
nohup python scripts/add_codes_to_dataset_gcs.py <same args as before> > processing.log 2>&1 &
```

---

## 📞 Support & Resources

- **GitHub Repo:** https://github.com/omarabb315/arabic-voice-pipeline
- **Documentation:** See `docs/STAGE2_H200_SETUP.md` for detailed info
- **HuggingFace Dataset:** https://huggingface.co/datasets/omarabb315/arabic-voice-cloning-dataset

---

## 📝 Notes for AI Assistant

When user starts chat with this document:

1. **Confirm SSH connection** to RunPod instance
2. **Verify credentials** are uploaded (GCS, HuggingFace)
3. **Run setup steps** in order
4. **Monitor progress** and help with any issues
5. **Verify completion** at each stage before moving to next
6. **Assist with training** configuration and monitoring

**Key Points:**
- Resume capability at all stages
- All data backed up to GCS (safe to terminate instance)
- ~4-6 hours for full processing + HF dataset creation
- Training time varies (hours to days)
- Total cost: ~$10-30 depending on training time

---

**Last Updated:** October 17, 2025  
**Repository Version:** Latest (commit 15e0788 and later)


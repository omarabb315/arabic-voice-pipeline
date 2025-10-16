# Stage 2: Adding VQ Codes on H200 Machine

This guide explains how to transfer the dataset from the T4 GCP VM to your on-premise H200 machine and add VQ codes.

## Overview

**Stage 1 (T4 - Completed):** Built audio + metadata (~10 minutes)  
**Stage 2 (H200 - This guide):** Add VQ codes (~1-2 hours on H200, would be 21 hours on T4)

---

## Step 1: Upload Dataset to GCS (from T4 GCP VM)

After Stage 1 completes on the T4 VM:

```bash
# Upload dataset to GCS bucket
cd /home/ubuntu/work/neutts
gsutil -m rsync -r dataset_cache/segments_dataset/ \
  gs://audio-data-gemini/datasets/segments_dataset/

# Verify upload
gsutil ls -lh gs://audio-data-gemini/datasets/segments_dataset/
```

**Expected size:** ~50-100 GB (depending on total audio duration)

---

## Step 2: Setup H200 On-Premise Machine

### 2.1 Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv_h200
source venv_h200/bin/activate

# Install required packages
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install datasets transformers neucodec librosa soundfile fire tqdm

# Install FFmpeg (for dataset audio features)
sudo apt-get update && sudo apt-get install -y ffmpeg
```

### 2.2 Setup GCS Access

```bash
# Install gcloud SDK (if not already installed)
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Or just install gsutil
sudo apt-get install google-cloud-sdk
```

### 2.3 Copy Service Account JSON Key

Transfer the GCS credentials file to H200:

```bash
# Option 1: SCP from local machine
scp ajgc-ai-portal-dev-01-93965d2f6a97.json user@h200-machine:/path/to/

# Option 2: Download from secure storage
# (Copy the JSON content securely)
```

Set environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/ajgc-ai-portal-dev-01-93965d2f6a97.json"

# Add to ~/.bashrc for persistence
echo 'export GOOGLE_APPLICATION_CREDENTIALS="/path/to/ajgc-ai-portal-dev-01-93965d2f6a97.json"' >> ~/.bashrc
```

---

## Step 3: Download Dataset from GCS (on H200)

```bash
# Create project directory
mkdir -p /path/to/neutts
cd /path/to/neutts

# Download dataset from GCS
gsutil -m rsync -r \
  gs://audio-data-gemini/datasets/segments_dataset/ \
  dataset_cache/segments_dataset/

# Verify download
ls -lh dataset_cache/segments_dataset/
```

**Download speed:** Depends on your internet connection  
**Time:** 10-60 minutes depending on bandwidth

---

## Step 4: Copy Scripts to H200

Transfer the encoding script:

```bash
# From T4 VM, upload scripts to GCS
cd /home/ubuntu/work/neutts
gsutil cp add_codes_to_dataset.py gs://audio-data-gemini/scripts/
gsutil cp requirements.txt gs://audio-data-gemini/scripts/

# On H200, download scripts
gsutil cp gs://audio-data-gemini/scripts/add_codes_to_dataset.py .
gsutil cp gs://audio-data-gemini/scripts/requirements.txt .
```

---

## Step 5: Run Stage 2 Encoding (on H200)

### 5.1 Test GPU

```bash
nvidia-smi  # Verify H200 is detected
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 5.2 Run Encoding in tmux

```bash
# Start tmux session
tmux new-session -s add_codes

# Run encoding
source venv_h200/bin/activate
python add_codes_to_dataset.py \
  --input_dataset_path='dataset_cache/segments_dataset/' \
  --output_dataset_path='dataset_cache/segments_with_codes/' \
  --codec_device=cuda \
  --save_every=100

# Detach: Ctrl+B then D
# Reattach: tmux attach -s add_codes
```

### 5.3 Monitor Progress

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check progress
tmux attach -s add_codes
```

**Expected time on H200:** 1-2 hours for ~10,000-30,000 segments  
**Much faster than T4:** ~10x speedup

---

## Step 6: Upload Final Dataset (Optional)

If you want to keep the final dataset in GCS:

```bash
# Upload enhanced dataset back to GCS
cd /path/to/neutts
gsutil -m rsync -r dataset_cache/segments_with_codes/ \
  gs://audio-data-gemini/datasets/segments_with_codes/
```

---

## Step 7: Use Dataset for Training

The final dataset at `dataset_cache/segments_with_codes/` contains:

- ✅ Audio (raw waveforms)
- ✅ VQ Codes (encoded representations)
- ✅ Text (transcriptions)
- ✅ Metadata (speaker, channel, gender, dialect, tone)

You can now use this for:
1. Building paired dataset (`create_pairs.py`)
2. Training the voice cloning model (`finetune-neutts.py`)

---

## Troubleshooting

### Issue: "gsutil: command not found"
```bash
pip install gsutil
# or
sudo apt-get install google-cloud-sdk
```

### Issue: "Permission denied" on GCS
```bash
# Verify credentials
gcloud auth activate-service-account --key-file=/path/to/key.json
gsutil ls gs://audio-data-gemini/  # Test access
```

### Issue: Out of memory on H200
```bash
# Reduce batch_size (though default is 1)
python add_codes_to_dataset.py --batch_size=1 --save_every=50
```

### Issue: Dataset too large to download
```bash
# Download in chunks (by channel)
# You'll need to modify the approach or download specific arrow files
```

---

## Performance Comparison

| Stage | Machine | Time | Bottleneck |
|-------|---------|------|------------|
| Stage 1 (Audio + Metadata) | T4 GCP | ~10 min | GCS download |
| Stage 2 (Add Codes) - T4 | T4 GCP | ~21 hours | Slow encoding |
| Stage 2 (Add Codes) - H200 | H200 On-prem | ~1-2 hours | 10x faster! |

**Total time saved:** ~19 hours! 🚀


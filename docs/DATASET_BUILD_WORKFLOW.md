# Complete Dataset Build Workflow

## Overview

This document explains the optimized two-stage workflow for building your voice cloning dataset.

---

## Architecture

```
Stage 1 (T4 GCP VM) - Fast!
  ↓ Downloads audio from GCS
  ↓ Segments based on transcripts  
  ↓ Saves as .npz files (compressed)
  → segments_cache/*.npz (221 GB, 656K files)
  ↓ Upload to S3
  → S3: neutts_segments/stage1/

Stage 2 (H200 On-Prem) - Powerful!
  ↓ Batch download from S3 (70K segments = 24GB)
  ↓ Encodes to VQ codes (GPU)
  ↓ Upload encoded to S3
  ↓ Clean local batch
  → S3: neutts_segments/stage2_with_codes/

Final Step (Either Machine)
  ↓ Stream from S3 or download
  ↓ Creates HuggingFace Dataset
  ↓ Push to HuggingFace Hub
  → HF: omarabb315/arabic-voice-cloning-dataset (~60 GB)
```

---

## Stage 1: Build Audio + Metadata (T4 - ~1 hour)

### Run the Build

```bash
cd /home/ubuntu/work/neutts
source venv_voicecloning/bin/activate

# Process single channel
python scripts/build_dataset_robust.py \
  --channels='["ZadTVchannel"]' \
  --skip_codec_encoding=True \
  --create_hf_dataset=False

# Process multiple channels
python scripts/build_dataset_robust.py \
  --channels='["ZadTVchannel", "Atheer_-"]' \
  --channel_mapping='{"Atheer_-": "Atheer - أثير"}' \
  --skip_codec_encoding=True \
  --create_hf_dataset=False
```

### What Gets Created

```
segments_cache/
├── ZadTVchannel/
│   ├── video123_seg_0.npz  (500 KB - audio + metadata)
│   ├── video123_seg_1.npz
│   ├── video456_seg_0.npz
│   └── ... (~150,000 files, ~40 GB total)
└── Atheer/
    └── ...
```

### Each .npz Contains:
- `audio`: Raw waveform (numpy array)
- `codes`: Empty list [] (added in Stage 2)
- `text`: Transcription
- `speaker_id`, `gender`, `dialect`, `tone`, `duration_sec`, `audio_id`

### Resume/Continue:
```bash
# If interrupted, just run again - skips existing segments!
python build_dataset_robust.py --channels='["ZadTVchannel"]' --skip_codec_encoding=True

# Add new channel
python build_dataset_robust.py --channels='["ZadTVchannel", "NewChannel"]' --skip_codec_encoding=True
# Skips ZadTVchannel, processes only NewChannel
```

---

## Transfer to H200 (via S3)

### Upload to S3 (from T4 VM)

```bash
# Set S3 credentials
export S3_ENDPOINT_URL="https://AJN-NAS-01.ALJAZEERA.TV:9021"
export S3_ACCESS_KEY="your-access-key"
export S3_SECRET_KEY="your-secret-key"
export S3_BUCKET_NAME="aifiles"

# Test connection
python -c "from scripts.s3_utils import test_connection; test_connection()"

# Upload all segments
python scripts/upload_to_s3.py \
  --source_dir=segments_cache/ \
  --s3_prefix=neutts_segments/stage1/ \
  --max_workers=4

# Expected time: ~30-60 minutes for 221GB (depends on bandwidth)
# Uploads in parallel for speed
# Automatically skips existing files
```

**Result in S3:**
```
s3://aifiles/neutts_segments/stage1/
├── ZadTVchannel/
│   ├── audio_123_seg_0.npz
│   └── ... (279,000 files)
├── Atheer_-/
│   └── ... (171,000 files)
└── thmanyahPodcasts/
    └── ... (205,000 files)
```

---

## Stage 2: Add VQ Codes (H200 - ~2-3 hours)

### Setup H200 Environment

```bash
# Clone repository
git clone https://github.com/your-username/neutts.git
cd neutts
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set S3 credentials
export S3_ENDPOINT_URL="https://AJN-NAS-01.ALJAZEERA.TV:9021"
export S3_ACCESS_KEY="your-access-key"
export S3_SECRET_KEY="your-secret-key"
export S3_BUCKET_NAME="aifiles"

# Test connection
python -c "from scripts.s3_utils import test_connection; test_connection()"
```

### Run Batch Encoding

**Recommended: 70K segments per batch (~24GB)**
```bash
python scripts/add_codes_to_dataset_s3.py \
  --s3_prefix=neutts_segments/stage1/ \
  --output_prefix=neutts_segments/stage2_with_codes/ \
  --batch_size=70000 \
  --temp_dir=/tmp/neutts_batch/ \
  --codec_device=cuda
```

**Conservative: 50K segments per batch (~17GB)**
```bash
python scripts/add_codes_to_dataset_s3.py \
  --batch_size=50000 \
  --codec_device=cuda
```

### How It Works:
1. **List** all .npz files in S3 (656K files)
2. **Download** batch of 70K segments to `/tmp` (~24GB)
3. **Encode** audio → VQ codes using NeuCodec (GPU)
4. **Upload** encoded segments to S3 (new prefix)
5. **Clean** local batch, clear GPU memory
6. **Repeat** for next batch (~10 batches total)
7. **Resume** automatically if interrupted (JSON progress tracking)

### Resume After Interruption:
```bash
# Just re-run the same command!
# Automatically skips already-processed files
python scripts/add_codes_to_dataset_s3.py \
  --s3_prefix=neutts_segments/stage1/ \
  --output_prefix=neutts_segments/stage2_with_codes/
```

### Storage Requirements:
- **Available space**: 33GB (overlay filesystem)
- **Per batch**: 24GB (70K segments)
- **Safe margin**: 9GB for temp files
- **Total processed**: 656K segments in ~10 batches

---

## Final Step: Create HuggingFace Dataset

### Option 1: On H200 (Stream from S3, push to HF Hub)

```bash
# Login to HuggingFace (one-time)
huggingface-cli login

# Create dataset and push to Hub
python create_hf_dataset_from_npz.py \
  --s3_prefix=neutts_segments/stage2_with_codes/ \
  --output_path=/tmp/segments_dataset/ \
  --push_to_hub=True \
  --hf_repo=omarabb315/arabic-voice-cloning-dataset

# Benefits:
# - No need to download all 230GB
# - Streams directly from S3
# - Pushes to HF Hub for easy access anywhere
```

### Option 2: On T4 (Download from S3, then create)

```bash
# Download encoded segments
python scripts/download_from_s3.py \
  --s3_prefix=neutts_segments/stage2_with_codes/ \
  --output_dir=segments_cache_encoded/ \
  --max_workers=4

# Create dataset locally
python create_hf_dataset_from_npz.py \
  --segments_cache_dir=segments_cache_encoded/ \
  --output_path=dataset_cache/final/ \
  --push_to_hub=True \
  --hf_repo=omarabb315/arabic-voice-cloning-dataset

# Benefits:
# - Have local copy for validation
# - Can create multiple dataset versions
# - T4 has more storage space
```

### Option 3: Local only (no codes, for testing)

```bash
# Create dataset without codes (for testing pipeline)
python create_hf_dataset_from_npz.py \
  --segments_cache_dir=segments_cache/ \
  --output_path=dataset_cache/segments_no_codes/
```

---

## Incremental Workflow Examples

### Example 1: Add New Channel

```bash
# Already have: ZadTVchannel (processed)

# Add new channel
python build_dataset_robust.py \
  --channels='["ZadTVchannel", "Atheer_-"]' \
  --channel_mapping='{"Atheer_-": "Atheer - أثير"}' \
  --skip_codec_encoding=True

# Result:
# - segments_cache/ZadTVchannel/  (unchanged)
# - segments_cache/Atheer/        (new!)
```

### Example 2: New Audio in Existing Channel

```bash
# New videos added to ZadTVchannel in GCS

python build_dataset_robust.py \
  --channels='["ZadTVchannel"]' \
  --skip_codec_encoding=True

# Only processes new audio files
# Skips existing .npz files
```

### Example 3: Updated Transcript

```bash
# Transcript updated with better segmentation

# Currently: File-level checking (skips entire audio if any segment exists)
# To reprocess: Delete the specific .npz files
rm segments_cache/ZadTVchannel/video123_seg_*.npz

# Then run
python build_dataset_robust.py --channels='["ZadTVchannel"]' --skip_codec_encoding=True
```

---

## Storage Management

### Disk Usage

| Stage | Location | Size | Can Delete? |
|-------|----------|------|-------------|
| Segments (.npz) | `segments_cache/` | ~40 GB | After creating HF dataset |
| Segments with codes | `segments_cache/` | ~45 GB | After creating HF dataset |
| Final HF dataset | `dataset_cache/` | ~60 GB | Never (for training) |

### Cleanup Strategies

**Strategy 1: Keep Everything (Safest)**
```bash
# Keep both:
# - segments_cache/  (source of truth, can rebuild)
# - dataset_cache/   (for training)
# Total: ~105 GB
```

**Strategy 2: Delete After Success (Save Space)**
```bash
# After creating HF dataset successfully:
rm -rf segments_cache/
# Keep only: dataset_cache/ (~60 GB)
```

**Strategy 3: Backup to GCS**
```bash
# Upload segments to GCS for backup
gsutil -m rsync -r segments_cache/ gs://audio-data-gemini/segments_backup/

# Delete local copy
rm -rf segments_cache/

# Can restore anytime
```

---

## Performance Metrics

### Stage 1 (T4):
- **GCS Download:** 80-200 MB/s (excellent!)
- **Librosa Load:** ~0.5-5s per file
- **Segment Extraction:** ~0.001s per segment
- **Total Time:** ~1 hour for 1367 audio files

### Stage 2 (H200):
- **Load .npz:** ~0.001s per segment
- **VQ Encoding:** ~0.1s per segment (10x faster than T4!)
- **Total Time:** ~1-2 hours for 300K segments

### Final HF Creation:
- **Load .npz:** ~5-10 minutes for 300K files
- **Create Dataset:** ~2-3 minutes
- **Save:** ~5-10 minutes
- **Total:** ~15-20 minutes

---

## Troubleshooting

### Issue: "Segment already exists but needs updating"

```bash
# Delete specific segments
rm segments_cache/ZadTVchannel/video123_seg_*.npz

# Or delete entire audio file's segments
rm segments_cache/ZadTVchannel/video123_*.npz

# Then rerun
python build_dataset_robust.py --channels='["ZadTVchannel"]'
```

### Issue: "Dataset creation failed"

```bash
# Check segments
find segments_cache/ -name "*.npz" | wc -l

# Try creating with fewer channels
python create_hf_dataset_from_npz.py --channels='["ZadTVchannel"]'

# Check one segment manually
python -c "import numpy as np; d=np.load('segments_cache/ZadTVchannel/xxx_seg_0.npz'); print(d.files)"
```

### Issue: "Out of memory during HF creation"

```bash
# Process in batches (not yet implemented, but can be added)
# Or upgrade RAM
# Or create separate datasets per channel and concatenate later
```

---

## Best Practices

1. **Always use `--create_hf_dataset=False` during Stage 1 and 2**
   - Creates HF dataset only once at the end
   - Faster processing

2. **Keep segments_cache until training succeeds**
   - Can recreate HF dataset anytime
   - Easy to debug issues

3. **Use detailed logs to monitor GCS performance**
   - Check `build_dataset_detailed.log`
   - Verify download speeds

4. **Monitor /tmp space**
   - Current code cleans up automatically
   - But verify: `df -h /tmp`

5. **Use tmux for long-running builds**
   - Can detach and reattach
   - Survives SSH disconnects

---

## Quick Reference

```bash
# Stage 1: Build segments on T4
python build_dataset_robust.py --channels='["ZadTVchannel"]' --skip_codec_encoding=True

# Upload to GCS
gsutil -m rsync -r segments_cache/ gs://audio-data-gemini/segments_cache/

# Stage 2: Add codes on H200
python add_codes_to_dataset.py --segments_cache_dir='segments_cache/'

# Create final dataset
python create_hf_dataset_from_npz.py

# Use for training
python create_pairs.py --segments_dataset_path='dataset_cache/segments_dataset/'
python finetune-neutts.py --config config.yaml
```


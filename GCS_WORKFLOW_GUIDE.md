# GCS Workflow Guide

Complete guide for processing dataset using GCS (instead of S3).

---

## Overview

Since H200 can access GCS and T4 encoding is too slow (66 hours), we use:

```
GCP T4 → GCS → H200 (batch processing) → GCS (encoded) → HuggingFace Hub
```

**Benefits:**
- No S3 needed (simpler!)
- H200 processes ~60x faster than T4
- Batch processing fits in H200's 33GB storage
- Final dataset on HF Hub for easy access

---

## Stage 1: Upload to GCS (GCP T4) - Optional

If segments aren't already in GCS:

```bash
cd /home/ubuntu/work/neutts
source venv_voicecloning/bin/activate

# Check if already in GCS
python -c "from scripts.gcs_utils import GCSClient; \
  c = GCSClient('audio-data-gemini'); \
  files = c.list_files(prefix='segments_cache/', suffix='.npz'); \
  print(f'{len(files):,} files in GCS')"

# If not, upload
python scripts/upload_to_gcs.py \
  --source_dir=segments_cache/ \
  --gcs_bucket=audio-data-gemini \
  --gcs_prefix=segments_cache/ \
  --max_workers=4
```

**Time:** ~30-60 minutes for 221GB

---

## Stage 2: Process on H200 - Main Step

### Setup H200 (One-time)

```bash
# Clone repo (or pull latest)
cd ~
git clone https://github.com/your-username/neutts.git
cd neutts

# Create environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set GCS credentials
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
export GCS_BUCKET_NAME="audio-data-gemini"

# Test connection
python scripts/test_h200_gcs_access.py
```

### Run Batch Processing

```bash
# Activate environment
cd ~/neutts
source venv/bin/activate

# Set credentials (if not permanent)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
export GCS_BUCKET_NAME="audio-data-gemini"

# Run processing (recommended: 70K per batch)
python scripts/add_codes_to_dataset_gcs.py \
  --gcs_bucket=audio-data-gemini \
  --input_prefix=segments_cache/ \
  --output_prefix=segments_with_codes/ \
  --batch_size=70000 \
  --temp_dir=/tmp/neutts_batch/ \
  --codec_device=cuda
```

**What happens:**
1. Lists 656K segments in GCS
2. Downloads batch of 70K (~24GB) to `/tmp`
3. Encodes with NeuCodec on GPU
4. Uploads encoded to GCS
5. Cleans up, repeats for next batch
6. ~10 batches total

**Time:** ~2-3 hours (vs 66 hours on T4!)

**Storage used:** Only 24GB at a time (72% of 33GB available)

### If Interrupted

Just re-run the same command - it automatically resumes!

```bash
python scripts/add_codes_to_dataset_gcs.py \
  --gcs_bucket=audio-data-gemini \
  --input_prefix=segments_cache/ \
  --output_prefix=segments_with_codes/
```

Progress is saved in `processing_progress.json`

---

## Stage 3: Create HuggingFace Dataset

### Option A: On H200 (Stream from GCS)

```bash
# Login to HuggingFace (one-time)
pip install huggingface-hub
huggingface-cli login

# Create and push dataset
python create_hf_dataset_from_npz.py \
  --gcs_bucket=audio-data-gemini \
  --gcs_prefix=segments_with_codes/ \
  --output_path=/home/jovyan/segments_dataset/ \
  --push_to_hub=True \
  --hf_repo=omarabb315/arabic-voice-cloning-dataset
```

**Time:** ~20-30 minutes

### Option B: On GCP T4 (Download first)

```bash
# Download encoded segments
python scripts/download_from_gcs.py \
  --gcs_bucket=audio-data-gemini \
  --gcs_prefix=segments_with_codes/ \
  --output_dir=segments_encoded/

# Create dataset
python create_hf_dataset_from_npz.py \
  --segments_cache_dir=segments_encoded/ \
  --output_path=dataset_cache/final/ \
  --push_to_hub=True \
  --hf_repo=omarabb315/arabic-voice-cloning-dataset
```

---

## Stage 4: Use for Training

```python
from datasets import load_dataset

# Load from anywhere!
dataset = load_dataset("omarabb315/arabic-voice-cloning-dataset", streaming=True)

# Train
for batch in dataset['train']:
    # Your training code
    pass
```

---

## Batch Size Options

### Recommended (70K segments)
```bash
--batch_size=70000  # ~24GB per batch, 72% of 33GB
```
- **Good balance** of speed and safety
- ~10 batches for 656K segments
- 9GB margin for overhead

### Conservative (50K segments)
```bash
--batch_size=50000  # ~17GB per batch, 51% of 33GB
```
- **Most safe** - lots of margin
- ~13 batches for 656K segments
- 16GB margin for overhead

### Aggressive (100K segments)
```bash
--batch_size=100000  # ~34GB per batch - TOO RISKY!
```
- ❌ **Not recommended** - uses all 33GB
- No margin for overhead
- May fail

---

## Storage Summary

### H200 Storage

```
/home/jovyan/     98GB persistent storage
  ├── neutts/      ~500MB (code)
  └── free:        ~97.5GB

/tmp/             33GB overlay filesystem
  ├── batch:       24GB (during processing)
  └── free:        9GB margin
```

### GCS Storage

```
gs://audio-data-gemini/
├── segments_cache/           221GB (raw, no codes)
└── segments_with_codes/      ~230GB (with VQ codes)
                              
Total: ~450GB
```

### HuggingFace Hub

```
omarabb315/arabic-voice-cloning-dataset
Size: ~60GB (optimized format)
```

---

## Performance Stats

| Operation | Machine | Time | Speed |
|-----------|---------|------|-------|
| Stage 1 (build) | GCP T4 | Complete | ✅ Done |
| Upload to GCS | GCP T4 | 30-60 min | ~5 GB/min |
| Stage 2 (encode) | **T4** | **66 hours** | 2.7 seg/sec |
| Stage 2 (encode) | **H200** | **2-3 hours** | ~60 seg/sec |
| Create HF dataset | Either | 20-30 min | Fast |

**Conclusion:** H200 is ~22x faster than T4 for encoding!

---

## Commands Cheatsheet

```bash
# Test GCS connection
python scripts/test_h200_gcs_access.py audio-data-gemini

# Upload to GCS (if needed)
python scripts/upload_to_gcs.py \
  --source_dir=segments_cache/ \
  --gcs_bucket=audio-data-gemini \
  --gcs_prefix=segments_cache/

# Process on H200
python scripts/add_codes_to_dataset_gcs.py \
  --gcs_bucket=audio-data-gemini \
  --input_prefix=segments_cache/ \
  --output_prefix=segments_with_codes/ \
  --batch_size=70000 \
  --codec_device=cuda

# Create HF dataset
python create_hf_dataset_from_npz.py \
  --gcs_bucket=audio-data-gemini \
  --gcs_prefix=segments_with_codes/ \
  --push_to_hub=True \
  --hf_repo=omarabb315/arabic-voice-cloning-dataset

# Load dataset
from datasets import load_dataset
dataset = load_dataset("omarabb315/arabic-voice-cloning-dataset")
```

---

## Troubleshooting

### GCS Connection Failed

```bash
# Check credentials
echo $GOOGLE_APPLICATION_CREDENTIALS
ls -l $GOOGLE_APPLICATION_CREDENTIALS

# Test manually
python -c "from google.cloud import storage; \
  client = storage.Client(); \
  bucket = client.bucket('audio-data-gemini'); \
  print('✅ Connected!' if bucket.exists() else '❌ Failed')"
```

### Out of Space on H200

```bash
# Check disk usage
df -h /tmp
df -h /home/jovyan

# Use smaller batches
python scripts/add_codes_to_dataset_gcs.py --batch_size=50000

# Clean temp directory
rm -rf /tmp/neutts_batch/
```

### GPU Out of Memory

```bash
# Check GPU
nvidia-smi

# Clear cache
python -c "import torch; torch.cuda.empty_cache()"

# Use CPU (slower)
python scripts/add_codes_to_dataset_gcs.py --codec_device=cpu
```

---

## Next Steps

After completing the dataset:

1. **Train on H200** with streaming from HF Hub
2. **Add more channels** (re-run Stage 1 on T4)
3. **Share dataset** with team via HF Hub
4. **Iterate** on model training

---

**Ready to start?** Begin with Stage 2 on H200! 🚀


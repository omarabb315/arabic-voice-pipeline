# Quick Start Guide

**TL;DR**: Complete the dataset pipeline in 3 steps across 2 machines.

---

## Current Status

✅ **Stage 1 Complete**: 656,300 segments (221 GB) built on GCP T4  
⏳ **Next**: Upload to S3, process on H200, push to HF Hub

---

## Step 1: Upload to S3 (GCP T4) - ~30-60 min

```bash
cd /home/ubuntu/work/neutts
source venv_voicecloning/bin/activate

# Set credentials (get these from your team)
export S3_ENDPOINT_URL="https://AJN-NAS-01.ALJAZEERA.TV:9021"
export S3_ACCESS_KEY="your-access-key"
export S3_SECRET_KEY="your-secret-key"
export S3_BUCKET_NAME="aifiles"

# Test connection
python -c "from scripts.s3_utils import test_connection; test_connection()"

# Upload (221GB, ~30-60 minutes)
python scripts/upload_to_s3.py \
  --source_dir=segments_cache/ \
  --s3_prefix=neutts_segments/stage1/ \
  --max_workers=4
```

**Expected output**: Progress bars showing upload completion

---

## Step 2: Process on H200 (H200) - ~2-3 hours

```bash
# Clone repo
git clone https://github.com/your-username/neutts.git
cd neutts

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set same S3 credentials as above
export S3_ENDPOINT_URL="https://AJN-NAS-01.ALJAZEERA.TV:9021"
export S3_ACCESS_KEY="your-access-key"
export S3_SECRET_KEY="your-secret-key"
export S3_BUCKET_NAME="aifiles"

# Test
python -c "from scripts.s3_utils import test_connection; test_connection()"

# Process (656K segments in 10 batches, ~2-3 hours)
python scripts/add_codes_to_dataset_s3.py \
  --s3_prefix=neutts_segments/stage1/ \
  --output_prefix=neutts_segments/stage2_with_codes/ \
  --batch_size=70000 \
  --codec_device=cuda
```

**What happens**: Downloads 70K segments → Encodes → Uploads → Repeats × 10

**If interrupted**: Just re-run the same command (auto-resumes!)

---

## Step 3: Push to HuggingFace Hub (H200 or GCP T4) - ~20-30 min

```bash
# Login (one-time)
huggingface-cli login

# Create and push dataset
python create_hf_dataset_from_npz.py \
  --s3_prefix=neutts_segments/stage2_with_codes/ \
  --output_path=/tmp/segments_dataset/ \
  --push_to_hub=True \
  --hf_repo=omarabb315/arabic-voice-cloning-dataset
```

**Result**: Dataset available at https://huggingface.co/datasets/omarabb315/arabic-voice-cloning-dataset

---

## Use the Dataset

```python
from datasets import load_dataset

# Load from anywhere
dataset = load_dataset("omarabb315/arabic-voice-cloning-dataset")

# Start training!
print(f"Segments: {len(dataset['train']):,}")
print(f"Sample: {dataset['train'][0]}")
```

---

## Troubleshooting

### S3 connection fails
```bash
# Check credentials
echo $S3_ENDPOINT_URL
echo $S3_BUCKET_NAME

# Test manually
python -c "from scripts.s3_utils import test_connection; test_connection()"
```

### H200 out of space
```bash
# Check disk
df -h

# Use smaller batches
python scripts/add_codes_to_dataset_s3.py --batch_size=50000
```

### HuggingFace push fails
```bash
# Check login
huggingface-cli whoami

# Re-login
huggingface-cli login
```

---

## Need More Details?

- **Complete setup**: See `SETUP_INSTRUCTIONS.md`
- **Full workflow**: See `docs/DATASET_BUILD_WORKFLOW.md`
- **S3 help**: See `docs/S3_SETUP_GUIDE.md`
- **Architecture**: See `README.md`

---

## Time Estimates

| Step | Machine | Time | What |
|------|---------|------|------|
| Upload | GCP T4 | 30-60 min | 221GB to S3 |
| Process | H200 | 2-3 hours | Add VQ codes (10 batches) |
| Push to HF | Either | 20-30 min | Create & upload dataset |
| **Total** | | **~3-4 hours** | **End-to-end** |

---

## Storage Requirements

| Machine | Available | Needed | Usage |
|---------|-----------|--------|-------|
| GCP T4 | ~200GB | 221GB | Stage 1 (done) |
| S3 | 10TB | 450GB | Stage 1 + 2 |
| H200 | 33GB | 24GB | Per batch only |
| HF Hub | ∞ | 60GB | Final dataset |

---

**Ready?** Start with Step 1! 🚀


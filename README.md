# NeuTTS Arabic Voice Cloning Dataset Pipeline

Production-ready pipeline for building large-scale Arabic voice cloning datasets across multiple machines with different capabilities.

## 🎯 Overview

This repository contains the complete workflow for:
1. **Stage 1 (GCP T4)**: Extract audio segments from videos with transcripts
2. **Stage 2 (H200)**: Add VQ codes using NeuCodec (GPU-accelerated)
3. **Final Step**: Create HuggingFace dataset and push to Hub

**Current Status**: 656K segments (221GB) processed across 3 channels ✅

## 🏗️ Architecture

```
┌─────────────────┐
│   GCP T4 VM     │  Stage 1: Extract segments
│  ─────────────  │  • Download from GCS
│  Audio segments │  • Segment based on transcripts
│    221GB .npz   │  • Save as .npz (audio + metadata)
└────────┬────────┘
         │
         │ Upload via boto3
         ▼
┌─────────────────┐
│   S3 Bucket     │  Intermediate Storage
│  ─────────────  │  • aifiles bucket (10TB)
│  stage1/        │  • Custom endpoint
│  stage2_with_   │  • 221GB → 230GB with codes
│    codes/       │
└────────┬────────┘
         │
         │ Batch download (70K segments = 24GB)
         ▼
┌─────────────────┐
│  H200 Machine   │  Stage 2: Add VQ codes
│  ─────────────  │  • Batch processing (33GB available)
│  NeuCodec GPU   │  • Encode → Upload → Clean
│  encoding       │  • Resume capability
└────────┬────────┘
         │
         │ Create HF dataset
         ▼
┌─────────────────┐
│ HuggingFace Hub │  Final Dataset
│  ─────────────  │  • ~60GB optimized format
│  Ready for      │  • Load from anywhere
│  training       │  • Public/private repo
└─────────────────┘
```

## 📦 Repository Structure

```
neutts/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.yaml                  # Training configuration
├── .gitignore                   # Git exclusions
│
├── scripts/                     # Main pipeline scripts
│   ├── build_dataset_robust.py       # Stage 1: Extract segments (GCP)
│   ├── s3_utils.py                   # S3 client utilities
│   ├── upload_to_s3.py               # Upload segments to S3
│   ├── add_codes_to_dataset_s3.py    # Stage 2: Add codes (H200)
│   ├── download_from_s3.py           # Download from S3
│   └── add_codes_to_dataset.py       # Original (local processing)
│
├── training/                    # Training scripts
│   ├── finetune-neutts.py           # Fine-tune NeuTTS
│   ├── create_pairs.py              # Create training pairs
│   └── neutts.py                    # Model code
│
├── docs/                        # Documentation
│   ├── DATASET_BUILD_WORKFLOW.md    # Detailed workflow
│   └── STAGE2_H200_SETUP.md         # H200 setup guide
│
└── create_hf_dataset_from_npz.py   # Create HF dataset (supports S3)
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Stage 1: Build Segments (GCP T4)

```bash
# Extract segments from videos
python scripts/build_dataset_robust.py \
  --channels='["ZadTVchannel", "Atheer_-", "thmanyahPodcasts"]' \
  --skip_codec_encoding=True \
  --create_hf_dataset=False

# Upload to S3
export S3_ENDPOINT_URL="https://your-endpoint:9021"
export S3_ACCESS_KEY="your-access-key"
export S3_SECRET_KEY="your-secret-key"
export S3_BUCKET_NAME="aifiles"

python scripts/upload_to_s3.py \
  --source_dir=segments_cache/ \
  --s3_prefix=neutts_segments/stage1/
```

### Stage 2: Add VQ Codes (H200)

```bash
# Clone repo on H200
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

# Process in batches
python scripts/add_codes_to_dataset_s3.py \
  --s3_prefix=neutts_segments/stage1/ \
  --output_prefix=neutts_segments/stage2_with_codes/ \
  --batch_size=70000 \
  --codec_device=cuda
```

**Options:**
- `--batch_size=70000` → ~24GB per batch (recommended, 72% of 33GB)
- `--batch_size=50000` → ~17GB per batch (safer, 51% of 33GB)

### Final Step: Create HuggingFace Dataset

**Option A: On H200 (streaming from S3)**
```bash
# Login to HuggingFace
huggingface-cli login

# Create and push dataset
python create_hf_dataset_from_npz.py \
  --s3_prefix=neutts_segments/stage2_with_codes/ \
  --output_path=/tmp/segments_dataset/ \
  --push_to_hub=True \
  --hf_repo=omarabb315/arabic-voice-cloning-dataset
```

**Option B: On GCP T4 (download first)**
```bash
# Download encoded segments
python scripts/download_from_s3.py \
  --s3_prefix=neutts_segments/stage2_with_codes/ \
  --output_dir=segments_cache_encoded/

# Create dataset
python create_hf_dataset_from_npz.py \
  --segments_cache_dir=segments_cache_encoded/ \
  --output_path=dataset_cache/final/ \
  --push_to_hub=True \
  --hf_repo=omarabb315/arabic-voice-cloning-dataset
```

### Use for Training

```python
from datasets import load_dataset

# Load from HuggingFace Hub
dataset = load_dataset("omarabb315/arabic-voice-cloning-dataset")

# Ready for training!
# Each sample contains:
# - audio: waveform (16kHz)
# - codes: VQ codes for training
# - text: transcription
# - speaker_id, channel, gender, dialect, tone, duration_sec
```

## 📊 Current Dataset Stats

| Channel | Segments | Duration | Size |
|---------|----------|----------|------|
| ZadTVchannel | 279,000 | ~93 hours | 94 GB |
| Atheer - أثير | 171,000 | ~57 hours | 58 GB |
| Thmanyah Podcasts | 205,000 | ~68 hours | 69 GB |
| **Total** | **656,300** | **~218 hours** | **221 GB** |

## 🔧 Key Features

### Batch Processing
- **Smart batching**: Process 70K segments at a time (~24GB)
- **Resume capability**: Automatically resumes after interruptions
- **Progress tracking**: JSON-based checkpoint system

### S3 Integration
- **Custom endpoint support**: Works with any S3-compatible storage
- **Parallel uploads**: Multi-threaded for speed
- **Progress bars**: Real-time transfer monitoring

### HuggingFace Hub
- **Direct push**: Upload dataset to Hub from anywhere
- **Streaming support**: Create dataset without downloading all files
- **Public/private**: Choose visibility

## 📝 Environment Variables

### S3 Configuration (Required for Stage 2)
```bash
export S3_ENDPOINT_URL="https://your-endpoint:9021"
export S3_ACCESS_KEY="your-access-key"
export S3_SECRET_KEY="your-secret-key"
export S3_BUCKET_NAME="aifiles"
```

### GCS Configuration (Required for Stage 1)
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

### HuggingFace (Required for Hub upload)
```bash
# Login once
huggingface-cli login
```

## 🐛 Troubleshooting

### S3 Connection Issues
```bash
# Test connection
python -c "from scripts.s3_utils import test_connection; test_connection()"
```

### Out of Space on H200
```bash
# Check disk usage
df -h /tmp

# Use smaller batches
python scripts/add_codes_to_dataset_s3.py --batch_size=50000
```

### Resume After Interruption
```bash
# Just re-run the same command - it automatically resumes!
python scripts/add_codes_to_dataset_s3.py \
  --s3_prefix=neutts_segments/stage1/ \
  --output_prefix=neutts_segments/stage2_with_codes/
```

## 📚 Documentation

- [**DATASET_BUILD_WORKFLOW.md**](docs/DATASET_BUILD_WORKFLOW.md) - Complete workflow guide
- [**STAGE2_H200_SETUP.md**](docs/STAGE2_H200_SETUP.md) - H200 setup instructions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.




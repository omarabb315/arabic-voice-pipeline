# Commands Cheatsheet

Quick copy-paste commands for the dataset pipeline.

---

## Setup (One-Time)

### Set S3 Credentials
```bash
export S3_ENDPOINT_URL="https://AJN-NAS-01.ALJAZEERA.TV:9021"
export S3_ACCESS_KEY="your-access-key-here"
export S3_SECRET_KEY="your-secret-key-here"
export S3_BUCKET_NAME="aifiles"
```

### Make Permanent (Optional)
```bash
cat >> ~/.bashrc << 'EOF'
export S3_ENDPOINT_URL="https://AJN-NAS-01.ALJAZEERA.TV:9021"
export S3_ACCESS_KEY="your-access-key-here"
export S3_SECRET_KEY="your-secret-key-here"
export S3_BUCKET_NAME="aifiles"
EOF
source ~/.bashrc
```

---

## GCP T4 Commands

### Test S3 Connection
```bash
cd /home/ubuntu/work/neutts
source venv_voicecloning/bin/activate
python -c "from scripts.s3_utils import test_connection; test_connection()"
```

### Upload Segments to S3
```bash
python scripts/upload_to_s3.py \
  --source_dir=segments_cache/ \
  --s3_prefix=neutts_segments/stage1/ \
  --max_workers=4
```

### Dry Run (Test Without Uploading)
```bash
python scripts/upload_to_s3.py \
  --source_dir=segments_cache/ \
  --s3_prefix=neutts_segments/stage1/ \
  --dry_run=True
```

### Push Code to GitHub
```bash
git init
git add .
git commit -m "Complete dataset pipeline with S3 batch processing"
git remote add origin https://github.com/your-username/neutts.git
git push -u origin main
```

---

## H200 Commands

### Initial Setup
```bash
cd ~
git clone https://github.com/your-username/neutts.git
cd neutts
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Set Credentials (Same as T4)
```bash
export S3_ENDPOINT_URL="https://AJN-NAS-01.ALJAZEERA.TV:9021"
export S3_ACCESS_KEY="your-access-key-here"
export S3_SECRET_KEY="your-secret-key-here"
export S3_BUCKET_NAME="aifiles"
```

### Test Connection
```bash
python -c "from scripts.s3_utils import test_connection; test_connection()"
```

### Check GPU
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### Run Stage 2 (Recommended)
```bash
python scripts/add_codes_to_dataset_s3.py \
  --s3_prefix=neutts_segments/stage1/ \
  --output_prefix=neutts_segments/stage2_with_codes/ \
  --batch_size=70000 \
  --temp_dir=/tmp/neutts_batch/ \
  --codec_device=cuda
```

### Run Stage 2 (Conservative)
```bash
python scripts/add_codes_to_dataset_s3.py \
  --batch_size=50000 \
  --codec_device=cuda
```

### Resume After Interruption
```bash
# Just re-run the same command - it auto-resumes!
python scripts/add_codes_to_dataset_s3.py \
  --s3_prefix=neutts_segments/stage1/ \
  --output_prefix=neutts_segments/stage2_with_codes/
```

---

## HuggingFace Commands

### Login (One-Time)
```bash
pip install huggingface-hub
huggingface-cli login
```

### Check Login
```bash
huggingface-cli whoami
```

### Create HF Dataset from S3 (H200)
```bash
python create_hf_dataset_from_npz.py \
  --s3_prefix=neutts_segments/stage2_with_codes/ \
  --output_path=/tmp/segments_dataset/ \
  --push_to_hub=True \
  --hf_repo=omarabb315/arabic-voice-cloning-dataset
```

### Create HF Dataset from Local (GCP T4)
```bash
# First download from S3
python scripts/download_from_s3.py \
  --s3_prefix=neutts_segments/stage2_with_codes/ \
  --output_dir=segments_cache_encoded/

# Then create dataset
python create_hf_dataset_from_npz.py \
  --segments_cache_dir=segments_cache_encoded/ \
  --output_path=dataset_cache/final/ \
  --push_to_hub=True \
  --hf_repo=omarabb315/arabic-voice-cloning-dataset
```

---

## Monitoring & Debugging

### Check Disk Space
```bash
df -h
df -h /tmp
du -sh segments_cache/
```

### Check GPU Memory
```bash
nvidia-smi
watch -n 1 nvidia-smi
```

### Count Segments
```bash
# Local
find segments_cache/ -name "*.npz" | wc -l

# By channel
find segments_cache/ZadTVchannel/ -name "*.npz" | wc -l
```

### Check S3 Upload Progress
```bash
# If upload script is running, it shows progress bars
# To manually check S3:
python -c "
from scripts.s3_utils import S3Client
s3 = S3Client()
files = s3.list_files(prefix='neutts_segments/stage1/', suffix='.npz')
print(f'Files in S3: {len(files):,}')
"
```

### View Processing Progress
```bash
# Check progress file (created during Stage 2)
cat processing_progress.json | python -m json.tool

# Or count processed files
python -c "
import json
with open('processing_progress.json') as f:
    data = json.load(f)
    print(f\"Processed: {len(data['processed']):,} files\")
"
```

---

## Utilities

### Test with Small Batch
```bash
# Create test directory
mkdir -p test_segments/ZadTVchannel
find segments_cache/ZadTVchannel -name "*.npz" | head -10 | xargs -I {} cp {} test_segments/ZadTVchannel/

# Upload test
python scripts/upload_to_s3.py \
  --source_dir=test_segments/ \
  --s3_prefix=neutts_segments/test/

# Clean up
rm -rf test_segments/
```

### Download Specific Channel
```bash
python scripts/download_from_s3.py \
  --s3_prefix=neutts_segments/stage2_with_codes/ \
  --output_dir=segments_encoded/ \
  --channels='["ZadTVchannel"]'
```

### Download with Limit
```bash
python scripts/download_from_s3.py \
  --s3_prefix=neutts_segments/stage2_with_codes/ \
  --output_dir=test_download/ \
  --limit=100
```

---

## Clean Up

### Remove Local Segments (After Upload)
```bash
# BE CAREFUL! Only after confirming upload to S3
rm -rf segments_cache/
```

### Remove Temp Files (H200)
```bash
rm -rf /tmp/neutts_batch/
rm -f processing_progress.json
```

### Remove Test Files
```bash
rm -rf test_segments/ test_download/ test_upload/
```

---

## Pro Tips

1. **Always test connection first**: `python -c "from scripts.s3_utils import test_connection; test_connection()"`
2. **Use tmux for long runs**: `tmux new -s neutts` (can disconnect and reconnect)
3. **Monitor with watch**: `watch -n 5 'nvidia-smi; df -h'`
4. **Parallel uploads**: Default is 4 workers, increase if you have bandwidth: `--max_workers=8`
5. **Resume is automatic**: Don't delete `processing_progress.json` during Stage 2

---

**Copy, paste, profit!** 🚀


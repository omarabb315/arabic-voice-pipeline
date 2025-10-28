# Orpheus Quick Reference Card

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Choose Training Mode

#### LoRA with Unsloth (Recommended - 16-40GB GPU)
```bash
# Edit config-orpheus.yaml
use_lora: true

# Train
./training/run_orpheus_training.sh
```

#### Standard HF Training (Like NeuTTS - 80GB+ GPU)
```bash
# Use standard training (no Unsloth)
python training/finetune-orpheus-standard.py \
    --config_fpath="training/config-orpheus-standard.yaml"
```

#### Full Fine-tuning with Unsloth (80GB+ GPU)
```bash
# Use full fine-tuning config
python training/finetune-orpheus.py \
    --config_fpath="training/config-orpheus-full-finetune.yaml"
```

---

## 📊 Training Modes Comparison

| Feature | LoRA (Unsloth) | Standard HF | Full FT (Unsloth) |
|---------|----------------|-------------|-------------------|
| **Script** | `finetune-orpheus.py` | `finetune-orpheus-standard.py` ⭐ NEW | `finetune-orpheus.py` |
| **GPU Memory** | 16-40GB | 80GB+ | 80GB+ |
| **Speed** | 2x faster | Standard | Standard |
| **Quality** | Very good (9/10) | Excellent (9.5/10) | Excellent (9.5/10) |
| **Parameters Trained** | ~15% | 100% | 100% |
| **Dependencies** | Unsloth + PEFT | Standard HF only | Unsloth |
| **Like NeuTTS** | ❌ No | ✅ Yes | ❌ No |
| **Recommended** | ✅ For most users | For NeuTTS-style | For max quality |

---

## 🔧 Configuration Files

### LoRA Training (Unsloth)
**File:** `training/config-orpheus.yaml`
```yaml
use_lora: true
lora_r: 64
lora_alpha: 64
per_device_train_batch_size: 4
lr: 1e-4
```

### Standard HF Training (No Unsloth) ⭐ NEW
**File:** `training/config-orpheus-standard.yaml`
```yaml
# No LoRA settings - pure HuggingFace
per_device_train_batch_size: 2
lr: 5e-5
# Simple like NeuTTS!
```

### Full Fine-tuning (Unsloth)
**File:** `training/config-orpheus-full-finetune.yaml`
```yaml
use_lora: false
per_device_train_batch_size: 2
lr: 5e-5
```

---

## 📝 Common Commands

### Training
```bash
# LoRA with Unsloth (recommended)
./training/run_orpheus_training.sh

# Standard HF (like NeuTTS) ⭐ NEW
python training/finetune-orpheus-standard.py \
    --config_fpath="training/config-orpheus-standard.yaml"

# Full fine-tuning with Unsloth
python training/finetune-orpheus.py \
    --config_fpath="training/config-orpheus-full-finetune.yaml"
```

### Inference
```bash
python training/test_orpheus_inference.py \
    --model_path="checkpoints/orpheus/checkpoint-1000" \
    --ref_audio_path="reference.wav" \
    --ref_text="مرحباً" \
    --target_text="كيف حالك؟" \
    --output_path="output.wav"
```

### Compare Models
```bash
python training/compare_models.py
```

---

## 🎯 Memory Requirements

| Configuration | GPU Memory | Example GPU |
|--------------|------------|-------------|
| LoRA + 4bit | 16 GB | RTX 4090, A10 |
| LoRA + FP16 | 32-40 GB | A100 40GB |
| Full FT | 80+ GB | A100 80GB, H100 |

---

## 📖 Documentation

| Document | Purpose |
|----------|---------|
| `ORPHEUS_IMPLEMENTATION_SUMMARY.md` | Quick overview |
| `training/ORPHEUS_FINETUNING_GUIDE.md` | Complete guide |
| `training/README.md` | Training overview |
| `training/IMPLEMENTATION_ORPHEUS.md` | Technical details |

---

## 🔄 Switching Between Modes

### From LoRA to Full Fine-tuning
1. Copy your data paths from `config-orpheus.yaml`
2. Edit `config-orpheus-full-finetune.yaml`
3. Set `use_lora: false`
4. Reduce batch size to 2
5. Train with the new config

### From Full Fine-tuning to LoRA
1. Edit `config-orpheus.yaml`
2. Set `use_lora: true`
3. Increase batch size to 4
4. Train with LoRA config

---

## ⚡ Performance Tips

### For LoRA Training
- Use `lora_r: 128` for better quality
- Enable `load_in_4bit: true` if OOM
- Batch size 4-8 works well

### For Full Fine-tuning
- Use `warmup_ratio: 0.1`
- Lower learning rate: `5e-5`
- Batch size 1-2 recommended
- May need gradient checkpointing

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

**LoRA:**
```yaml
load_in_4bit: true
per_device_train_batch_size: 2
lora_r: 32
```

**Full FT:**
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
```

### Poor Quality

**Try:**
1. More training epochs (5-10)
2. More data per speaker
3. Better reference audio (clean, 3-10s)
4. Increase LoRA rank (128) or use full FT

---

## 📦 GitHub Repository

**Repository:** https://github.com/omarabb315/arabic-voice-pipeline

**Latest Commit:** fe3439f - "Add Orpheus voice cloning implementation with optional LoRA"

**Clone:**
```bash
git clone https://github.com/omarabb315/arabic-voice-pipeline.git
cd arabic-voice-pipeline
```

---

## 🎓 Learning Path

1. **Start:** Read `ORPHEUS_IMPLEMENTATION_SUMMARY.md`
2. **Setup:** Follow `training/ORPHEUS_FINETUNING_GUIDE.md`
3. **Train:** Use LoRA mode first (easier)
4. **Evaluate:** Test with inference script
5. **Optimize:** Try full fine-tuning if needed
6. **Compare:** Run `compare_models.py`

---

## 💡 Pro Tips

✅ **Always start with LoRA** - faster iteration, good quality  
✅ **Use 4-bit** if you have limited GPU memory  
✅ **Monitor W&B** for training progress  
✅ **Test checkpoints** regularly during training  
✅ **Use full fine-tuning** only if quality is critical  

---

## 🔗 Quick Links

- **Main Config (LoRA):** `training/config-orpheus.yaml`
- **Full FT Config:** `training/config-orpheus-full-finetune.yaml`
- **Training Script:** `training/finetune-orpheus.py`
- **Inference Script:** `training/test_orpheus_inference.py`
- **Helper Script:** `training/run_orpheus_training.sh`

---

## ✨ Key Feature: Optional LoRA

```yaml
# Enable LoRA (memory efficient)
use_lora: true
lora_r: 64
lora_alpha: 64

# Disable LoRA (full fine-tuning)
use_lora: false
```

**Your request:** ✅ Implemented!

You can now easily switch between LoRA and full fine-tuning by changing the `use_lora` flag in the config file.

---

**Need help?** Check the documentation or run: `python training/compare_models.py`


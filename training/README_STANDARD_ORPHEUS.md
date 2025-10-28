# Standard Orpheus Training (Without Unsloth)

This guide explains how to train Orpheus using standard HuggingFace Transformers, similar to NeuTTS training, without Unsloth or LoRA.

## Overview

**Script:** `finetune-orpheus-standard.py`

This is a simplified version that:
- ✅ Uses standard HuggingFace Transformers (like NeuTTS)
- ✅ No Unsloth dependency
- ✅ No LoRA (full fine-tuning only)
- ✅ Same training approach as `finetune-neutts.py`
- ✅ SNAC codec for Orpheus audio
- ✅ Compatible with existing paired datasets

---

## Why This Script?

### Use Standard Orpheus Training When:

1. **You want simplicity** - No additional dependencies beyond standard HF
2. **You prefer traditional training** - Full fine-tuning like NeuTTS
3. **You don't want Unsloth** - Avoid proprietary optimizations
4. **You have sufficient GPU** - 80GB+ GPU available
5. **You want consistency** - Same training pattern as NeuTTS

### Comparison: Standard vs Unsloth

| Feature | Standard (`finetune-orpheus-standard.py`) | Unsloth (`finetune-orpheus.py`) |
|---------|------------------------------------------|----------------------------------|
| **Dependencies** | Standard HF Transformers | Unsloth + PEFT |
| **Training Speed** | Standard | 2x faster |
| **Memory Usage** | 80GB+ | 40GB with LoRA |
| **Simplicity** | ✅ Simple | More complex |
| **LoRA Support** | ❌ No | ✅ Yes |
| **Like NeuTTS** | ✅ Yes | Different |

---

## Installation

### Minimal Dependencies

```bash
# Core dependencies (no Unsloth needed)
pip install torch transformers datasets
pip install snac  # For Orpheus audio codec
pip install phonemizer loguru fire omegaconf
pip install wandb tensorboard  # Optional but recommended

# NOT needed for standard training:
# pip install unsloth peft bitsandbytes  # Skip these!
```

---

## Quick Start

### 1. Prepare Dataset

Use the same paired dataset as NeuTTS:

```bash
python training/create_pairs.py \
    --segments_path="dataset_cache/segments_dataset/" \
    --output_path="dataset_cache/paired_dataset/"
```

### 2. Configure Training

Edit `training/config-orpheus-standard.yaml`:

```yaml
# Model
restore_from: "canopylabs/orpheus-3b-0.1"  # Standard model
max_seq_len: 4096

# Dataset
paired_dataset_path: "dataset_cache/paired_dataset/"

# Training
lr: 5e-5
per_device_train_batch_size: 2
gradient_accumulation_steps: 16
num_train_epochs: 3

# Simple - no LoRA settings needed!
```

### 3. Train

```bash
python training/finetune-orpheus-standard.py \
    --config_fpath="training/config-orpheus-standard.yaml"
```

That's it! Simple and straightforward, just like NeuTTS.

---

## Detailed Comparison

### Code Structure

Both scripts follow the same pattern:

```python
# finetune-neutts.py
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(restore_from)
model = AutoModelForCausalLM.from_pretrained(restore_from)
# ... train with standard Trainer

# finetune-orpheus-standard.py (NEW!)
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(restore_from)
model = AutoModelForCausalLM.from_pretrained(restore_from)
# ... train with standard Trainer (same pattern!)
```

### Key Differences from NeuTTS

Only the **audio codec** differs:

| Component | NeuTTS | Orpheus Standard |
|-----------|--------|------------------|
| **Model** | Custom Transformer | Llama-based |
| **Tokenizer** | HF AutoTokenizer | HF AutoTokenizer |
| **Trainer** | HF Trainer | HF Trainer |
| **Audio Codec** | Mimi (16kHz) | SNAC (24kHz) ⭐ |
| **Training** | Full fine-tuning | Full fine-tuning |
| **Dependencies** | Standard HF | Standard HF + SNAC |

---

## Configuration

### Default Config (80GB GPU)

```yaml
restore_from: "canopylabs/orpheus-3b-0.1"
max_seq_len: 4096
lr: 5e-5
per_device_train_batch_size: 2
gradient_accumulation_steps: 16
num_train_epochs: 3
warmup_ratio: 0.1
weight_decay: 0.01
```

### Memory-Constrained (40GB GPU)

```yaml
restore_from: "canopylabs/orpheus-3b-0.1"
max_seq_len: 2048  # Reduce sequence length
lr: 5e-5
per_device_train_batch_size: 1  # Smaller batch
gradient_accumulation_steps: 32  # More accumulation
num_train_epochs: 3
warmup_ratio: 0.1
weight_decay: 0.01
```

**Note:** If 40GB is still not enough, use the LoRA version (`finetune-orpheus.py`) instead.

---

## Memory Requirements

### Standard Training (This Script)

| GPU | Batch Size | Gradient Accumulation | Feasible? |
|-----|------------|----------------------|-----------|
| 24GB (RTX 3090) | 1 | 32 | ❌ Not enough |
| 40GB (A100) | 1 | 32 | ⚠️ Tight |
| 80GB (A100) | 2 | 16 | ✅ Recommended |
| 80GB (H100) | 2 | 16 | ✅ Recommended |

### Alternative: LoRA Training

If you don't have 80GB GPU, use `finetune-orpheus.py` with LoRA:

```bash
python training/finetune-orpheus.py \
    --config_fpath="training/config-orpheus.yaml"
```

LoRA works with 16-40GB GPUs.

---

## Training Command

### Basic Usage

```bash
python training/finetune-orpheus-standard.py \
    --config_fpath="training/config-orpheus-standard.yaml"
```

### Multi-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    training/finetune-orpheus-standard.py \
    --config_fpath="training/config-orpheus-standard.yaml"
```

### Resume from Checkpoint

```yaml
# In config file
restore_from: "checkpoints/orpheus-arabic-voice-cloning-standard/checkpoint-1000"
```

---

## Inference

Inference is the same for all Orpheus models:

```bash
python training/test_orpheus_inference.py \
    --model_path="checkpoints/orpheus-arabic-voice-cloning-standard/checkpoint-1000" \
    --ref_audio_path="reference.wav" \
    --ref_text="مرحباً، كيف حالك؟" \
    --target_text="أهلاً وسهلاً بك في هذا البرنامج" \
    --output_path="output.wav"
```

---

## Comparison: All Three Approaches

| Feature | NeuTTS | Orpheus (Standard) | Orpheus (Unsloth) |
|---------|--------|-------------------|-------------------|
| **Script** | `finetune-neutts.py` | `finetune-orpheus-standard.py` | `finetune-orpheus.py` |
| **Model** | Custom 1B | Llama 3B | Llama 3B |
| **Codec** | Mimi 16kHz | SNAC 24kHz | SNAC 24kHz |
| **Training** | Full FT | Full FT | LoRA or Full FT |
| **Dependencies** | Standard HF | Standard HF + SNAC | Unsloth + PEFT |
| **GPU Memory** | 16-32GB | 80GB+ | 16-40GB (LoRA) |
| **Speed** | Standard | Standard | 2x faster |
| **Quality** | Good | Excellent | Excellent |
| **Simplicity** | ✅ Simple | ✅ Simple | Complex |

---

## FAQ

### Q: Why not just use the Unsloth version?

**A:** 
- Some users prefer standard HuggingFace without proprietary optimizations
- Simpler dependencies and setup
- More transparent training process
- Consistency with NeuTTS training approach

### Q: Is this slower than the Unsloth version?

**A:** Yes, Unsloth is ~2x faster. But this version is simpler and uses standard tools.

### Q: Can I use 4-bit quantization?

**A:** Not recommended for full fine-tuning. Use LoRA version (`finetune-orpheus.py`) if you need quantization.

### Q: Which should I use?

**Choose Standard Orpheus if:**
- ✅ You have 80GB+ GPU
- ✅ You prefer simplicity
- ✅ You want standard HF training
- ✅ You like the NeuTTS approach

**Choose Unsloth Orpheus if:**
- ✅ You have limited GPU (16-40GB)
- ✅ You want faster training
- ✅ You need LoRA support

---

## Troubleshooting

### Out of Memory

**Solution 1:** Reduce batch size
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
```

**Solution 2:** Reduce sequence length
```yaml
max_seq_len: 2048
```

**Solution 3:** Use LoRA version instead
```bash
python training/finetune-orpheus.py --config_fpath="training/config-orpheus.yaml"
```

### Training Too Slow

**Solution:** Use the Unsloth version for 2x speedup
```bash
python training/finetune-orpheus.py --config_fpath="training/config-orpheus.yaml"
```

### Model Quality Issues

- Train longer (5-10 epochs)
- Use better reference audio
- Increase training data
- Try the Unsloth version with higher LoRA rank

---

## Summary

**Standard Orpheus Training:**
- ✅ Simple and straightforward
- ✅ Uses standard HuggingFace (like NeuTTS)
- ✅ No Unsloth or LoRA complexity
- ✅ Full fine-tuning only
- ⚠️ Requires 80GB+ GPU
- ⚠️ Slower than Unsloth version

**Perfect for:**
- Users who want simplicity
- Teams with large GPUs
- Those who prefer standard tools
- Consistency with NeuTTS workflow

**For memory-constrained setups:**
Use `finetune-orpheus.py` with LoRA instead.

---

## Quick Reference

```bash
# Standard training (no Unsloth)
python training/finetune-orpheus-standard.py \
    --config_fpath="training/config-orpheus-standard.yaml"

# Unsloth training with LoRA (memory efficient)
python training/finetune-orpheus.py \
    --config_fpath="training/config-orpheus.yaml"

# Inference (works for both)
python training/test_orpheus_inference.py \
    --model_path="checkpoints/.../checkpoint-1000" \
    --ref_audio_path="ref.wav" \
    --target_text="..."
```

---

**Choose the approach that fits your setup and preferences!**


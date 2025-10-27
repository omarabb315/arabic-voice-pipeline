# Orpheus Voice Cloning Fine-tuning Guide

This guide explains how to finetune the Orpheus model for Arabic voice cloning using paired datasets.

## Overview

The Orpheus finetuning pipeline adapts the Orpheus TTS model for voice cloning by training it on paired samples where:
- **Reference audio + text** provides the voice characteristics to clone
- **Target text** is what we want the model to synthesize in the reference voice

The model learns to:
1. Extract voice characteristics from reference audio
2. Generate target speech that maintains the reference voice while saying the target text
3. Respect conditioning attributes (gender, dialect, tone)

## Architecture Differences

### NeuTTS vs Orpheus

| Feature | NeuTTS | Orpheus |
|---------|--------|---------|
| Base Model | Custom architecture | Llama-based (3B params) |
| Audio Codec | Mimi (encodec) | SNAC |
| Sample Rate | 16kHz | 24kHz (SNAC) |
| Token Format | `<|speech_{i}|>` | Interleaved 7-token frames |
| Special Tokens | Custom prompts | Llama token IDs |
| Training Method | Standard transformer | LoRA adapters |

### Orpheus Token Format

Orpheus uses specific token IDs for structure:

```
[SOH=128259] [SOT=128000] text [EOT=128009] [EOH=128260]
[SOA=128261] [SOS=128257] audio_codes [EOS=128258] [EOA=128262]
```

Audio codes start at `128266` and are interleaved from 3 SNAC layers (7 tokens per frame).

## Installation

### 1. Install Required Packages

```bash
# Install Unsloth (for efficient training)
pip install unsloth

# Install SNAC codec
pip install snac

# Install audio libraries
pip install librosa soundfile
pip uninstall torchaudio -y
pip install torchaudio==2.7.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies (if not already installed)
pip install phonemizer datasets transformers omegaconf fire loguru wandb
```

### 2. Verify Installation

```python
from unsloth import FastLanguageModel
from snac import SNAC
import phonemizer

print("✅ All dependencies installed successfully")
```

## Workflow

### Step 1: Prepare Paired Dataset

Use the `create_pairs.py` script to generate paired training data:

```bash
python training/create_pairs.py \
    --segments_path="dataset_cache/segments_dataset/" \
    --output_path="dataset_cache/paired_dataset/" \
    --min_duration_sec=0.5 \
    --max_duration_sec=15.0 \
    --min_segments_per_speaker=2 \
    --max_pairs_per_speaker=100 \
    --batch_size=10000
```

**Expected output:**
```
dataset_cache/paired_dataset/
├── dataset_info.json
├── state.json
└── data-00000-of-00001.arrow
```

Each pair contains:
- Reference: audio, text, codes, gender, dialect, tone
- Target: audio, text, codes, gender, dialect, tone

### Step 2: Configure Training

You can choose between **LoRA fine-tuning** (efficient, recommended) or **full fine-tuning** (higher quality, more memory).

#### Option A: LoRA Fine-tuning (Recommended)

Edit `training/config-orpheus.yaml`:

```yaml
# Model settings
restore_from: "unsloth/orpheus-3b-0.1-ft"  # Base model
max_seq_len: 4096

# Dataset
paired_dataset_path: "dataset_cache/paired_dataset/"

# Training
save_root: "checkpoints/"
run_name: "orpheus-arabic-voice-cloning"
lr: 1.0e-4
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
num_train_epochs: 3

# LoRA (efficient fine-tuning)
use_lora: true  # Enable LoRA
lora_r: 64
lora_alpha: 64
load_in_4bit: false

# Optimization
warmup_ratio: 0.05
weight_decay: 0.01
lr_scheduler_type: "cosine"

# Checkpointing
save_steps: 500
logging_steps: 1
save_total_limit: 2

# Wandb
wandb_project: "Orpheus-Arabic-Voice-Cloning"
```

**Memory requirements:** 16-40GB GPU

#### Option B: Full Fine-tuning (Higher Quality)

Use `training/config-orpheus-full-finetune.yaml`:

```yaml
# Model settings
restore_from: "unsloth/orpheus-3b-0.1-ft"
max_seq_len: 4096

# Training (adjusted for full fine-tuning)
lr: 5.0e-5  # Lower learning rate
per_device_train_batch_size: 2  # Smaller batch
gradient_accumulation_steps: 16  # Compensate for smaller batch
num_train_epochs: 3

# LoRA - DISABLED for full fine-tuning
use_lora: false  # Disable LoRA for full fine-tuning
load_in_4bit: false

# Optimization
warmup_ratio: 0.1  # More warmup
weight_decay: 0.01
lr_scheduler_type: "cosine"
```

**Memory requirements:** 80GB+ GPU (A100/H100)

**Comparison:**

| Feature | LoRA | Full Fine-tuning |
|---------|------|------------------|
| **Memory** | 16-40GB | 80GB+ |
| **Speed** | 2x faster | Slower |
| **Quality** | Very good | Slightly better |
| **GPU** | 1x A100 40GB | 1-2x A100 80GB |
| **Recommended** | ✅ Yes | Only if needed |

### Step 3: Run Fine-tuning

```bash
python training/finetune-orpheus.py \
    --config_fpath="training/config-orpheus.yaml"
```

**What happens during training:**

1. ✅ Loads Orpheus base model with LoRA adapters
2. ✅ Loads SNAC model for audio tokenization
3. ✅ Initializes Arabic phonemizer
4. ✅ Preprocesses paired dataset:
   - Phonemizes Arabic text
   - Tokenizes audio using SNAC
   - Creates training sequences
   - Masks labels (only target codes are trained)
5. ✅ Trains with Unsloth optimizations
6. ✅ Saves checkpoints to `checkpoints/orpheus-arabic-voice-cloning/`

**Training logs:**

```
🦥 Unsloth: Loading model...
Loading checkpoint from unsloth/orpheus-3b-0.1-ft
Loading SNAC model...
Initializing phonemizer...
Loading paired dataset from dataset_cache/paired_dataset/
Loaded 10,000 training pairs
Preprocessing dataset...
[████████████████████] 10000/10000 [00:15<00:00]
Filtered out 127 failed samples. 9,873 samples remaining.
Starting training...
Step 1/500 | Loss: 2.456
Step 2/500 | Loss: 2.231
...
```

### Step 4: Test Inference

Use the inference script to test your fine-tuned model:

```bash
python training/test_orpheus_inference.py \
    --model_path="checkpoints/orpheus-arabic-voice-cloning/checkpoint-1000" \
    --ref_audio_path="path/to/reference_audio.wav" \
    --ref_text="مرحباً، كيف حالك؟" \
    --ref_gender="male" \
    --ref_dialect="MSA" \
    --ref_tone="neutral" \
    --target_text="أهلاً وسهلاً بك في هذا البرنامج" \
    --target_gender="male" \
    --target_dialect="MSA" \
    --target_tone="formal" \
    --output_path="output_audio.wav"
```

**Expected output:**

```
============================================================
Orpheus Voice Cloning Inference
============================================================

1. Loading model from checkpoints/...
2. Loading SNAC model...
3. Initializing phonemizer...
4. Loading reference audio from path/to/reference_audio.wav...
   Reference audio encoded to 147 tokens
5. Phonemizing texts...
   Ref phones: mˈaɾħabˈaː, kˈajfa ħˈaːluk?
   Target phones: ʔˈahlan wasˈahlan bˈika fˈiː hˈaːðaː ʔalbˈaɾnaːmˈadʒ
6. Creating input prompt...
7. Generating target audio codes...
   Generated 1247 tokens
   Extracted 840 target audio tokens
8. Decoding audio with SNAC...
9. Saving audio to output_audio.wav...

============================================================
✅ Voice cloning complete!
📁 Output saved to: output_audio.wav
============================================================
```

## Key Implementation Details

### 1. Audio Tokenization (SNAC)

```python
# SNAC produces 3 layers of codes
# Layer 1: Coarse (4096 tokens)
# Layer 2: Medium (4096 tokens)  
# Layer 3: Fine (4096 tokens)

# Interleaving pattern (7 tokens per frame):
for i in range(num_frames):
    codes = [
        layer_1[i] + 128266,
        layer_2[2*i] + 128266 + 4096,
        layer_3[4*i] + 128266 + (2*4096),
        layer_3[4*i+1] + 128266 + (3*4096),
        layer_2[2*i+1] + 128266 + (4*4096),
        layer_3[4*i+2] + 128266 + (5*4096),
        layer_3[4*i+3] + 128266 + (6*4096),
    ]
```

### 2. Conditioning Format

```python
# Reference and target have independent conditioning
ref_conditioned = f"[{gender}|{dialect}|{tone}] {phonemes}"
target_conditioned = f"[{gender}|{dialect}|{tone}] {phonemes}"
combined = f"{ref_conditioned} | {target_conditioned}"
```

Supported values:
- **Gender:** male, female, unknown
- **Dialect:** MSA, Egyptian, Levantine, Gulf, Iraqi, Maghrebi, Yemeni, Sudanese, unknown
- **Tone:** neutral, formal, conversational, enthusiastic, analytical, empathetic, serious, humorous

### 3. Label Masking

Only target audio codes are trained (reference codes are masked):

```python
labels = [-100] * len(input_ids)  # Start with all masked

# Find where target codes start
target_start = len(text_sequence) + 2 + len(ref_codes)

# Unmask target codes and end tokens
labels[target_start:] = input_ids[target_start:]
```

This ensures:
- Model learns to generate based on reference
- Doesn't overfit to copying reference exactly
- Focuses on voice cloning capability

### 4. LoRA Training

Uses LoRA for efficient fine-tuning:

```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj"      # MLP
]
```

Benefits:
- 85%+ parameters remain frozen
- Much faster training
- Lower memory requirements
- Easy to merge or swap adapters

## Monitoring Training

### Weights & Biases

Training automatically logs to W&B:

```
https://wandb.ai/your-username/Orpheus-Arabic-Voice-Cloning
```

Metrics tracked:
- Training loss
- Learning rate
- GPU memory usage
- Samples processed
- Time per step

### TensorBoard

Also logs to TensorBoard:

```bash
tensorboard --logdir checkpoints/orpheus-arabic-voice-cloning/
```

### Key Metrics to Watch

1. **Loss curve:** Should decrease steadily
   - Initial: ~2.5-3.0
   - After 1 epoch: ~1.5-2.0
   - After 3 epochs: ~1.0-1.5

2. **Sample quality:** Test periodically
   - Checkpoint every 500 steps
   - Run inference on validation samples
   - Check voice similarity and text accuracy

## Memory Requirements

| Configuration | GPU Memory | Training Speed |
|--------------|------------|----------------|
| 4bit + LoRA r=32 | ~16 GB | Fast |
| FP16 + LoRA r=64 | ~40 GB | Faster |
| BF16 + LoRA r=64 | ~40 GB | Fastest |

Recommendations:
- **16GB GPU:** Use `load_in_4bit: true`, `lora_r: 32`, `batch_size: 2`
- **40GB GPU:** Use `load_in_4bit: false`, `lora_r: 64`, `batch_size: 4`
- **80GB GPU:** Use `load_in_4bit: false`, `lora_r: 64`, `batch_size: 8`

## Troubleshooting

### Issue: Out of Memory

**Solutions:**
1. Enable 4-bit quantization: `load_in_4bit: true`
2. Reduce batch size: `per_device_train_batch_size: 2`
3. Reduce LoRA rank: `lora_r: 32`
4. Reduce sequence length: `max_seq_len: 2048`

### Issue: Poor Voice Cloning Quality

**Solutions:**
1. Train longer: `num_train_epochs: 5`
2. More pairs per speaker: `max_pairs_per_speaker: 200`
3. Increase LoRA rank: `lora_r: 128`
4. Better reference audio (clean, 3-10s duration)

### Issue: Slow Preprocessing

**Solutions:**
1. Reduce workers: `num_preprocessing_workers: 20`
2. Use smaller batch size during map: `batch_size: 5000`
3. Pre-tokenize audio and save dataset

### Issue: Model Generates Nonsense

**Possible causes:**
1. Learning rate too high → Try `lr: 5e-5`
2. Not enough training → Increase epochs
3. Bad data filtering → Check dataset quality

## Advanced Usage

### Resume from Checkpoint

```yaml
restore_from: "checkpoints/orpheus-arabic-voice-cloning/checkpoint-1000"
```

### Multi-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    training/finetune-orpheus.py \
    --config_fpath="training/config-orpheus.yaml"
```

### Custom Conditioning Tokens

Edit `finetune-orpheus.py` to add new conditioning categories:

```python
# Add new categories
VALID_EMOTIONS = ['happy', 'sad', 'angry', 'neutral']

# Update preprocessing
emotion_conditioned = f"[{gender}|{dialect}|{tone}|{emotion}] {phonemes}"
```

### Merge LoRA Weights

After training, merge LoRA back into base model:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="checkpoints/orpheus-arabic-voice-cloning/checkpoint-1000",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=False,
)

# Merge and save
model.save_pretrained_merged(
    "merged_model",
    tokenizer,
    save_method="merged_16bit"
)
```

## Comparison with NeuTTS

| Aspect | NeuTTS | Orpheus |
|--------|--------|---------|
| Model Size | ~1B params | 3B params |
| Quality | Good | Better |
| Speed | Faster | Slower |
| Memory | Less | More |
| Training | Standard | LoRA |
| Codec | Mimi | SNAC |

**When to use NeuTTS:**
- Limited compute resources
- Need fastest inference
- Smaller model size required

**When to use Orpheus:**
- Want best quality
- Have sufficient GPU memory
- Can afford slower inference
- Want to leverage Llama architecture

## Citation

If you use this work, please cite:

```bibtex
@software{orpheus_voice_cloning,
  title={Orpheus Arabic Voice Cloning},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo}
}
```

## Support

For issues and questions:
- GitHub Issues: [your-repo/issues]
- Discord: [Unsloth Discord](https://discord.gg/unsloth)
- Email: your-email@example.com


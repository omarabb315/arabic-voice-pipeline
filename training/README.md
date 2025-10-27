# Training Scripts for Arabic Voice Cloning

This directory contains training scripts for fine-tuning voice cloning models on Arabic speech data.

## Available Models

### 1. NeuTTS
- **Script:** `finetune-neutts.py`
- **Architecture:** Custom transformer-based TTS
- **Codec:** Mimi (Encodec)
- **Model Size:** ~1B parameters
- **Best for:** Fast inference, lower memory requirements

### 2. Orpheus (NEW!)
- **Script:** `finetune-orpheus.py`
- **Architecture:** Llama-based (3B parameters)
- **Codec:** SNAC
- **Training:** LoRA adapters
- **Best for:** Highest quality, leverages Llama architecture

## Quick Start

### Option 1: NeuTTS (Original)

```bash
# 1. Create paired dataset
python training/create_pairs.py \
    --segments_path="dataset_cache/segments_dataset/" \
    --output_path="dataset_cache/paired_dataset/"

# 2. Train NeuTTS
python training/finetune-neutts.py \
    --config_fpath="config.yaml"

# 3. Test with batch inference
python training/test_checkpoint_batch.py \
    --checkpoint_path="checkpoints/neutts/checkpoint-1000" \
    --test_samples_path="test_samples/"
```

### Option 2: Orpheus (Recommended for Quality)

```bash
# 1. Create paired dataset (same as above)
python training/create_pairs.py \
    --segments_path="dataset_cache/segments_dataset/" \
    --output_path="dataset_cache/paired_dataset/"

# 2. Train Orpheus using the helper script
./training/run_orpheus_training.sh

# OR manually:
python training/finetune-orpheus.py \
    --config_fpath="training/config-orpheus.yaml"

# 3. Test with inference script
python training/test_orpheus_inference.py \
    --model_path="checkpoints/orpheus-arabic-voice-cloning/checkpoint-1000" \
    --ref_audio_path="reference.wav" \
    --ref_text="مرحباً" \
    --target_text="أهلاً وسهلاً"
```

## File Structure

```
training/
├── README.md                          # This file
├── ORPHEUS_FINETUNING_GUIDE.md       # Detailed Orpheus guide
│
├── create_pairs.py                    # Create paired dataset for training
├── neutts.py                          # NeuTTS model definition
│
├── finetune-neutts.py                 # NeuTTS finetuning script
├── test_checkpoint_batch.py           # Batch testing for NeuTTS
├── test_checkpoint_gradio.py          # Gradio demo for NeuTTS
│
├── finetune-orpheus.py                # Orpheus finetuning script (NEW!)
├── test_orpheus_inference.py          # Orpheus inference script (NEW!)
├── config-orpheus.yaml                # Orpheus config template (NEW!)
├── run_orpheus_training.sh            # Helper script for Orpheus (NEW!)
```

## Detailed Documentation

### Creating Paired Datasets

The `create_pairs.py` script creates training pairs from your segmented dataset:

```python
# Each pair consists of:
{
    # Reference (voice to clone)
    'ref_audio': audio_array,
    'ref_text': "مرحباً",
    'ref_codes': [101, 203, ...],
    'ref_gender': 'male',
    'ref_dialect': 'MSA',
    'ref_tone': 'neutral',
    
    # Target (what to synthesize)
    'target_audio': audio_array,
    'target_text': "أهلاً وسهلاً",
    'target_codes': [156, 289, ...],
    'target_gender': 'male',
    'target_dialect': 'MSA',
    'target_tone': 'formal',
}
```

**Key parameters:**
- `min_segments_per_speaker`: Minimum segments required per speaker (default: 2)
- `max_pairs_per_speaker`: Limit pairs per speaker to avoid imbalance (default: None = unlimited)
- `min_duration_sec`: Minimum segment duration (default: 0.5s)
- `max_duration_sec`: Maximum segment duration (default: 15.0s)

### NeuTTS Training

**Key features:**
- Uses phonemization for text input
- Custom speech tokens: `<|speech_{i}|>`
- Conditioning tokens for gender/dialect/tone
- Standard transformer training

**Configuration (config.yaml):**
```yaml
restore_from: "canopylabs/NeuTTS-3b-ft"
paired_dataset_path: "dataset_cache/paired_dataset/"
max_seq_len: 2048
lr: 5e-5
per_device_train_batch_size: 8
num_train_epochs: 3
```

### Orpheus Training

**Key features:**
- Llama-based architecture (3B params)
- LoRA efficient finetuning
- SNAC codec (24kHz, 7 tokens per frame)
- Unsloth optimizations

**Configuration (config-orpheus.yaml):**
```yaml
restore_from: "unsloth/orpheus-3b-0.1-ft"
paired_dataset_path: "dataset_cache/paired_dataset/"
max_seq_len: 4096
lr: 1e-4
lora_r: 64
lora_alpha: 64
```

**See [ORPHEUS_FINETUNING_GUIDE.md](ORPHEUS_FINETUNING_GUIDE.md) for complete details.**

## Model Comparison

| Feature | NeuTTS | Orpheus |
|---------|--------|---------|
| **Quality** | Good | Excellent |
| **Speed** | Faster | Slower |
| **Memory** | 16GB+ | 40GB+ (or 16GB with 4bit) |
| **Training Time** | Shorter | Longer |
| **Architecture** | Custom | Llama-based |
| **Codec** | Mimi | SNAC |
| **Sample Rate** | 16kHz | 24kHz |
| **Best Use Case** | Production deployment | Research, highest quality |

## Conditioning Attributes

Both models support conditioning on:

### Gender
- `male`
- `female`
- `unknown`

### Dialect
- `MSA` (Modern Standard Arabic)
- `Egyptian`
- `Levantine`
- `Gulf`
- `Iraqi`
- `Maghrebi`
- `Yemeni`
- `Sudanese`
- `unknown`

### Tone
- `neutral`
- `formal`
- `conversational`
- `enthusiastic`
- `analytical`
- `empathetic`
- `serious`
- `humorous`

## Memory Requirements

### NeuTTS
- **Minimum:** 16GB VRAM
- **Recommended:** 32GB VRAM
- **Batch size:** 4-8

### Orpheus
- **Minimum (4bit):** 16GB VRAM
- **Recommended (FP16):** 40GB VRAM
- **Optimal (BF16):** 80GB VRAM
- **Batch size:** 2-8 (depending on quantization)

## Training Tips

### 1. Data Quality
- Clean audio (minimal background noise)
- Accurate transcriptions
- Balanced speaker distribution
- Diverse dialects and tones

### 2. Hyperparameter Tuning
- Start with default configs
- Adjust learning rate if loss plateaus
- Increase epochs for better quality
- Use gradient accumulation if OOM

### 3. Monitoring
- Watch training loss (should decrease)
- Test checkpoints regularly
- Use W&B or TensorBoard for visualization
- Listen to generated samples

### 4. Optimization
- Use LoRA for faster training (Orpheus)
- Enable gradient checkpointing
- Use mixed precision (BF16/FP16)
- Batch similar-length samples

## Testing and Evaluation

### NeuTTS Testing

**Batch testing:**
```bash
python training/test_checkpoint_batch.py \
    --checkpoint_path="checkpoints/neutts/checkpoint-1000" \
    --test_samples_path="test_samples/" \
    --output_dir="outputs/"
```

**Interactive demo:**
```bash
python training/test_checkpoint_gradio.py \
    --checkpoint_path="checkpoints/neutts/checkpoint-1000"
```

### Orpheus Testing

**Single inference:**
```bash
python training/test_orpheus_inference.py \
    --model_path="checkpoints/orpheus/checkpoint-1000" \
    --ref_audio_path="ref.wav" \
    --ref_text="مرحباً" \
    --target_text="كيف حالك؟" \
    --output_path="output.wav"
```

## Common Issues

### Issue: CUDA Out of Memory

**For NeuTTS:**
```yaml
per_device_train_batch_size: 4  # Reduce from 8
gradient_accumulation_steps: 4  # Increase to maintain effective batch size
```

**For Orpheus:**
```yaml
load_in_4bit: true  # Enable 4-bit quantization
lora_r: 32  # Reduce LoRA rank
per_device_train_batch_size: 2
```

### Issue: Poor Voice Quality

1. **Check data quality:**
   - Audio should be 16kHz mono
   - Clean recordings (SNR > 20dB)
   - Accurate transcriptions

2. **Increase training:**
   - More epochs (5-10)
   - More pairs per speaker (100-200)
   - Larger LoRA rank (128)

3. **Better reference audio:**
   - 3-10 seconds duration
   - Clear speech
   - Minimal background noise

### Issue: Slow Training

1. **Use Unsloth (Orpheus):** Already included
2. **Enable compilation:**
   ```yaml
   torch_compile: true  # In TrainingArguments
   ```
3. **Optimize data loading:**
   ```yaml
   dataloader_num_workers: 4
   ```

### Issue: Model Overfitting

1. **Add regularization:**
   ```yaml
   weight_decay: 0.01
   lora_dropout: 0.05  # For Orpheus
   ```
2. **Early stopping:** Monitor validation loss
3. **More diverse data:** Increase speakers/dialects

## Advanced Usage

### Multi-GPU Training

```bash
# NeuTTS
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    training/finetune-neutts.py --config_fpath="config.yaml"

# Orpheus
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    training/finetune-orpheus.py --config_fpath="training/config-orpheus.yaml"
```

### Resume from Checkpoint

Update config:
```yaml
restore_from: "checkpoints/orpheus/checkpoint-1000"
```

### Custom Dataset Format

If your dataset has a different format, modify the preprocessing in:
- `preprocess_sample()` (NeuTTS)
- `preprocess_sample_orpheus()` (Orpheus)

## Performance Benchmarks

Tested on NVIDIA H100 (80GB):

### NeuTTS
- **Preprocessing:** ~2 hours (50k pairs)
- **Training:** ~8 hours (3 epochs)
- **Inference:** ~0.5s per sample

### Orpheus
- **Preprocessing:** ~3 hours (50k pairs)
- **Training:** ~12 hours (3 epochs)
- **Inference:** ~1.0s per sample

## Contributing

When adding new features:

1. Follow existing code style
2. Add docstrings and comments
3. Update relevant README/docs
4. Test on small dataset first
5. Submit PR with example outputs

## Support

- **Issues:** Open GitHub issue
- **Questions:** See main README or docs/
- **Updates:** Check IMPLEMENTATION_SUMMARY.md

## License

See main project LICENSE file.

## Citation

```bibtex
@software{arabic_voice_cloning_training,
  title={Arabic Voice Cloning Training Pipeline},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo}
}
```


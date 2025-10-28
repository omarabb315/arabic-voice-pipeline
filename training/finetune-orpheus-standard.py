import warnings
import re
import os
import torch
import phonemizer
import wandb
import locale
import torchaudio.transforms as T

from fire import Fire
from omegaconf import OmegaConf
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from loguru import logger as LOGGER
from datasets import load_from_disk
from snac import SNAC

warnings.filterwarnings("ignore")
locale.getpreferredencoding = lambda: "UTF-8"

# Orpheus special token IDs
TOKENIZER_LENGTH = 128256
START_OF_TEXT = 128000
END_OF_TEXT = 128009
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
END_OF_AI = 128262
PAD_TOKEN = 128263
AUDIO_TOKENS_START = 128266


def tokenise_audio_snac(waveform, snac_model, ds_sample_rate=16000):
    """
    Tokenize audio using SNAC model.
    
    Args:
        waveform: Audio waveform (numpy array)
        snac_model: SNAC model instance
        ds_sample_rate: Input sample rate
        
    Returns:
        List of interleaved audio token IDs
    """
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)
    
    # Resample to 24kHz (SNAC's expected rate)
    resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=24000)
    waveform = resample_transform(waveform)
    waveform = waveform.unsqueeze(0).to(snac_model.device)
    
    # Generate codes from SNAC
    with torch.inference_mode():
        codes = snac_model.encode(waveform)
    
    # Interleave codes from 3 layers (7 tokens per frame)
    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item() + AUDIO_TOKENS_START)
        all_codes.append(codes[1][0][2*i].item() + AUDIO_TOKENS_START + 4096)
        all_codes.append(codes[2][0][4*i].item() + AUDIO_TOKENS_START + (2*4096))
        all_codes.append(codes[2][0][(4*i)+1].item() + AUDIO_TOKENS_START + (3*4096))
        all_codes.append(codes[1][0][(2*i)+1].item() + AUDIO_TOKENS_START + (4*4096))
        all_codes.append(codes[2][0][(4*i)+2].item() + AUDIO_TOKENS_START + (5*4096))
        all_codes.append(codes[2][0][(4*i)+3].item() + AUDIO_TOKENS_START + (6*4096))
    
    return all_codes


def remove_duplicate_frames(codes_list):
    """
    Remove duplicate consecutive frames from SNAC codes.
    Frames are identified by their first token (every 7 tokens).
    """
    if len(codes_list) % 7 != 0:
        raise ValueError("Codes list length must be divisible by 7")
    
    result = codes_list[:7]
    
    for i in range(7, len(codes_list), 7):
        current_first = codes_list[i]
        previous_first = result[-7]
        
        if current_first != previous_first:
            result.extend(codes_list[i:i+7])
    
    return result


def preprocess_sample_orpheus(sample, tokenizer, max_len, g2p, snac_model):
    """
    Preprocess paired sample for Orpheus voice cloning training (standard HF style).
    
    Format:
    [SOH] [SOT] ref_conditioning ref_phones | target_conditioning target_phones [EOT] [EOH] 
    [SOA] [SOS] ref_codes target_codes [EOS] [EOA]
    """
    
    # Get special token ID
    ignore_index = -100
    
    # Define valid values for validation
    VALID_GENDERS = ['male', 'female', 'unknown']
    VALID_DIALECTS = ['MSA', 'Egyptian', 'Levantine', 'Gulf', 'Iraqi', 'Maghrebi', 'Yemeni', 'Sudanese', 'unknown']
    VALID_TONES = ['neutral', 'formal', 'conversational', 'enthusiastic', 'analytical', 'empathetic', 'serious', 'humorous']
    
    # Unpack paired sample
    ref_text = sample["ref_text"]
    ref_gender = sample["ref_gender"]
    ref_dialect = sample["ref_dialect"]
    ref_tone = sample["ref_tone"]
    ref_audio = sample["ref_audio"]["array"]
    
    target_text = sample["target_text"]
    target_gender = sample["target_gender"]
    target_dialect = sample["target_dialect"]
    target_tone = sample["target_tone"]
    target_audio = sample["target_audio"]["array"]
    
    # Validate and map invalid values to "unknown"
    if ref_gender not in VALID_GENDERS:
        LOGGER.warning(f"⚠️ Invalid ref_gender '{ref_gender}', mapping to 'unknown'")
        ref_gender = "unknown"
    if ref_dialect not in VALID_DIALECTS:
        LOGGER.warning(f"⚠️ Invalid ref_dialect '{ref_dialect}', mapping to 'unknown'")
        ref_dialect = "unknown"
    if ref_tone not in VALID_TONES:
        LOGGER.warning(f"⚠️ Invalid ref_tone '{ref_tone}', mapping to 'unknown'")
        ref_tone = "unknown"
    
    if target_gender not in VALID_GENDERS:
        LOGGER.warning(f"⚠️ Invalid target_gender '{target_gender}', mapping to 'unknown'")
        target_gender = "unknown"
    if target_dialect not in VALID_DIALECTS:
        LOGGER.warning(f"⚠️ Invalid target_dialect '{target_dialect}', mapping to 'unknown'")
        target_dialect = "unknown"
    if target_tone not in VALID_TONES:
        LOGGER.warning(f"⚠️ Invalid target_tone '{target_tone}', mapping to 'unknown'")
        target_tone = "unknown"
    
    # Dictionary to return on failure
    invalid_sample_output = {"input_ids": [], "labels": [], "valid": False}
    
    # Phonemize reference and target
    try:
        ref_phones = g2p.phonemize([ref_text])
        target_phones = g2p.phonemize([target_text])
    except Exception as e:
        LOGGER.warning(f"⚠️ Phonemization failed: {e}")
        return invalid_sample_output
    
    # Safety check
    if not ref_phones or not ref_phones[0]:
        LOGGER.warning(f"⚠️ Empty phonemization output for ref_text: {ref_text}")
        return invalid_sample_output
    
    if not target_phones or not target_phones[0]:
        LOGGER.warning(f"⚠️ Empty phonemization output for target_text: {target_text}")
        return invalid_sample_output
    
    ref_phones = ' '.join(ref_phones[0].split())
    target_phones = ' '.join(target_phones[0].split())
    
    # Add conditioning tokens (using format: gender|dialect|tone)
    ref_conditioned = f"[{ref_gender}|{ref_dialect}|{ref_tone}] {ref_phones}"
    target_conditioned = f"[{target_gender}|{target_dialect}|{target_tone}] {target_phones}"
    
    # Combine text with separator
    combined_text = f"{ref_conditioned} | {target_conditioned}"
    
    # Tokenize audio using SNAC
    try:
        ref_codes_list = tokenise_audio_snac(ref_audio, snac_model, ds_sample_rate=16000)
        target_codes_list = tokenise_audio_snac(target_audio, snac_model, ds_sample_rate=16000)
        
        # Remove duplicate frames
        ref_codes_list = remove_duplicate_frames(ref_codes_list)
        target_codes_list = remove_duplicate_frames(target_codes_list)
    except Exception as e:
        LOGGER.warning(f"⚠️ SNAC tokenization failed: {e}")
        return invalid_sample_output
    
    if len(ref_codes_list) == 0 or len(target_codes_list) == 0:
        LOGGER.warning(f"⚠️ Empty audio codes")
        return invalid_sample_output
    
    # Create input sequence following Orpheus format
    # Text part: [SOH] [SOT] text [EOT] [EOH]
    text_ids = tokenizer.encode(combined_text, add_special_tokens=True)
    text_sequence = [START_OF_HUMAN, START_OF_TEXT] + text_ids + [END_OF_TEXT, END_OF_HUMAN]
    
    # Audio part: [SOA] [SOS] ref_codes target_codes [EOS] [EOA]
    audio_sequence = [START_OF_AI, START_OF_SPEECH] + ref_codes_list + target_codes_list + [END_OF_SPEECH, END_OF_AI]
    
    # Combine
    input_ids = text_sequence + audio_sequence
    
    # Check length
    if len(input_ids) >= max_len:
        LOGGER.warning(f"⚠️ Sample exceeds max_len ({len(input_ids)} >= {max_len}), skipping")
        return invalid_sample_output
    
    # Label masking: only train on TARGET audio codes
    labels = [ignore_index] * len(input_ids)
    
    # Find where target codes start
    # Target codes start after: text_sequence + [SOA, SOS] + ref_codes
    target_start_idx = len(text_sequence) + 2 + len(ref_codes_list)
    
    # Unmask target codes and end tokens
    if target_start_idx < len(input_ids):
        labels[target_start_idx:] = input_ids[target_start_idx:]
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "valid": True,
    }


def main(config_fpath: str):
    
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Load config
    print(f"Loading config from {config_fpath}")
    config = OmegaConf.load(config_fpath)
    checkpoints_dir = os.path.join(config.save_root, config.run_name)
    LOGGER.info(f"Logging to: {checkpoints_dir}")
    
    restore_from = config.restore_from
    paired_dataset_path = config.paired_dataset_path
    
    # Load model and tokenizer using standard HuggingFace (no Unsloth)
    print(f"Loading Orpheus model from {restore_from}")
    print("Using standard HuggingFace Transformers (no Unsloth)")
    
    tokenizer = AutoTokenizer.from_pretrained(restore_from)
    model = AutoModelForCausalLM.from_pretrained(
        restore_from,
        torch_dtype="auto",
        # attn_implementation="flash_attention_2" if config.get("use_flash_attn", False) else "eager"
    )
    
    print(f"Model loaded: {model.config.model_type}")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B")
    
    # Initialize SNAC model for audio tokenization
    print("Loading SNAC model...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model = snac_model.to("cuda")
    
    # Initialize phonemizer
    print("Initializing phonemizer...")
    g2p = phonemizer.backend.EspeakBackend(
        language='ar',  # Arabic
        preserve_punctuation=True,
        with_stress=True,
        words_mismatch="ignore",
        language_switch="remove-flags"
    )
    
    # Create partial preprocessing function
    partial_preprocess = partial(
        preprocess_sample_orpheus,
        tokenizer=tokenizer,
        max_len=config.max_seq_len,
        g2p=g2p,
        snac_model=snac_model,
    )
    
    # Load paired dataset
    print(f"Loading paired dataset from {paired_dataset_path}")
    paired_dataset = load_from_disk(paired_dataset_path)
    print(f"Loaded {len(paired_dataset)} training pairs")
    
    # Preprocess dataset
    print("Preprocessing dataset...")
    original_columns = paired_dataset.column_names
    paired_dataset = paired_dataset.map(
        partial_preprocess,
        remove_columns=original_columns,
        num_proc=config.get("num_preprocessing_workers", 40),
        desc="Preprocessing"
    )
    
    # Filter out invalid samples
    original_size = len(paired_dataset)
    paired_dataset = paired_dataset.filter(
        lambda x: x["valid"] == True,
        num_proc=config.get("num_preprocessing_workers", 40),
        desc="Filtering valid samples"
    )
    filtered_size = len(paired_dataset)
    if filtered_size < original_size:
        print(f"Filtered out {original_size - filtered_size} failed samples. {filtered_size} samples remaining.")
    
    # Remove the 'valid' column
    paired_dataset = paired_dataset.remove_columns(["valid"])
    
    # Initialize wandb
    wandb.init(
        project=config.get("wandb_project", "Orpheus-Voice-Cloning-Standard"),
        name=config.run_name,
        config={
            "model": config.restore_from,
            "paired_dataset": config.paired_dataset_path,
            "train_size": len(paired_dataset),
            "learning_rate": config.lr,
            "batch_size": config.per_device_train_batch_size,
            "epochs": config.num_train_epochs,
            "training_method": "full_finetuning_standard_hf",
            "no_lora": True,
            "no_unsloth": True,
        }
    )
    
    # Training arguments (similar to NeuTTS)
    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        do_train=True,
        learning_rate=config.lr,
        max_steps=config.get("max_steps", -1),
        bf16=True,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_total_limit=config.get("save_total_limit", 2),
        ignore_data_skip=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        num_train_epochs=config.num_train_epochs,
        report_to=["tensorboard", "wandb"],
        optim="adamw_torch",
        weight_decay=config.get("weight_decay", 0.01),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        seed=3407,
    )
    
    # Create data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    
    # Create trainer (standard HuggingFace Trainer, like NeuTTS)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=paired_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    print("⚠️  Full fine-tuning mode: All parameters will be trained")
    print("⚠️  This requires significantly more GPU memory than LoRA")
    trainer.train()
    
    # Save model
    print(f"Saving model to {checkpoints_dir}")
    trainer.save_model(checkpoints_dir)
    tokenizer.save_pretrained(checkpoints_dir)
    
    print("Training complete!")


if __name__ == "__main__":
    Fire(main)


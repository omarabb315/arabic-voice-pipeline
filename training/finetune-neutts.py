import warnings

import re
import os
import torch
import phonemizer

from fire import Fire
from omegaconf import OmegaConf
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from loguru import logger as LOGGER
from datasets import load_dataset, load_from_disk


warnings.filterwarnings("ignore")


ACRONYM = re.compile(r"(?:[a-zA-Z]\.){2,}")
ACRONYM_NO_PERIOD = re.compile(r"(?:[A-Z]){2,}")


def data_filter(sample):
    text = sample["text"]

    if len(text) == 0:
        return False

    if re.search(r'\d', text):
        return False

    if re.search(ACRONYM, text) or re.search(ACRONYM_NO_PERIOD, text):
        return False

    if text[-1] not in ".,?!":
        return False

    if '£' in text or '$' in text:
        return False

    return True


def preprocess_sample(sample, tokenizer, max_len, g2p):

    # get special tokens
    speech_gen_start = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
    ignore_index = -100  # this is from LLaMA

    # Define valid values for validation
    VALID_GENDERS = ['male', 'female', 'unknown']
    VALID_DIALECTS = ['MSA', 'Egyptian', 'Levantine', 'Gulf', 'Iraqi', 'Maghrebi', 'Yemeni', 'Sudanese', 'unknown']
    VALID_TONES = ['neutral', 'formal', 'conversational', 'enthusiastic', 'analytical', 'empathetic', 'serious', 'humorous']

    # unpack paired sample
    ref_codes = sample["ref_codes"]
    ref_text = sample["ref_text"]
    ref_gender = sample["ref_gender"]
    ref_dialect = sample["ref_dialect"]
    ref_tone = sample["ref_tone"]
    
    target_codes = sample["target_codes"]
    target_text = sample["target_text"]
    target_gender = sample["target_gender"]
    target_dialect = sample["target_dialect"]
    target_tone = sample["target_tone"]
    
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

    # phonemize reference and target
    ref_phones = g2p.phonemize([ref_text])
    target_phones = g2p.phonemize([target_text])

    # SAFE CHECK
    if not ref_phones or not ref_phones[0]:
        LOGGER.warning(f"⚠️ Empty phonemization output for ref_text: {ref_text}")
        return {"valid": False}
    
    if not target_phones or not target_phones[0]:
        LOGGER.warning(f"⚠️ Empty phonemization output for target_text: {target_text}")
        return {"valid": False}

    ref_phones = ' '.join(ref_phones[0].split())
    target_phones = ' '.join(target_phones[0].split())
    
    # Add conditioning tokens
    ref_conditioned = f"<|{ref_gender}|><|{ref_dialect}|><|{ref_tone}|> {ref_phones}"
    target_conditioned = f"<|{target_gender}|><|{target_dialect}|><|{target_tone}|> {target_phones}"
    
    # Concatenate ref and target text
    combined_text = f"{ref_conditioned} {target_conditioned}"
    
    # Create codes strings
    ref_codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes])
    target_codes_str = "".join([f"<|speech_{i}|>" for i in target_codes])
    combined_codes_str = ref_codes_str + target_codes_str

    # get chat format (matching _infer_ggml style)
    chat = f"""<|TEXT_PROMPT_START|>{combined_text}<|TEXT_PROMPT_END|><|SPEECH_GENERATION_START|>{combined_codes_str}<|SPEECH_GENERATION_END|>"""
    
    # Use tokenizer with truncation to avoid warnings
    encoding = tokenizer(
        chat,
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    ids = encoding['input_ids']
    
    # If sequence was too long and got truncated, mark as invalid
    # (We want complete samples, not truncated ones)
    if len(ids) >= max_len:
        LOGGER.warning(f"⚠️ Sample exceeds max_len ({len(ids)} >= {max_len}), skipping")
        return {"valid": False}

    # Label masking: only train on TARGET codes
    labels = [ignore_index] * len(ids)
    
    # Find speech generation start token position
    try:
        speech_gen_start_idx = ids.index(speech_gen_start)
        
        # Find where target codes start by encoding ref codes
        ref_codes_ids = tokenizer.encode(ref_codes_str, add_special_tokens=False)
        target_start_idx = speech_gen_start_idx + 1 + len(ref_codes_ids)
        
        # Only unmask target codes (and end token)
        if target_start_idx < len(ids):
            labels[target_start_idx:] = ids[target_start_idx:]
    except ValueError:
        # speech_gen_start not found, skip this sample
        LOGGER.warning(f"⚠️ Could not find SPEECH_GENERATION_START token, skipping sample")
        return {"valid": False}

    # return in hf format as lists (not tensors)
    # DataCollatorForSeq2Seq will handle padding and attention masks dynamically
    return {
        "input_ids": ids,
        "labels": labels,
        "valid": True,
    }


def add_conditioning_tokens(tokenizer, dataset_path):
    """Add conditioning tokens from predefined attributes to the tokenizer."""
    
    # Define supported conditioning values
    genders = ['male', 'female', 'unknown']
    dialects = ['MSA', 'Egyptian', 'Levantine', 'Gulf', 'Iraqi', 'Maghrebi', 'Yemeni', 'Sudanese', 'unknown']
    tones = ['neutral', 'formal', 'conversational', 'enthusiastic', 'analytical', 'empathetic', 'serious', 'humorous']
    
    # Create special tokens
    special_tokens = []
    for gender in genders:
        special_tokens.append(f"<|{gender}|>")
    for dialect in dialects:
        special_tokens.append(f"<|{dialect}|>")
    for tone in tones:
        special_tokens.append(f"<|{tone}|>")
    
    # Add to tokenizer
    num_added = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    print(f"Added {num_added} conditioning tokens to tokenizer")
    print(f"Tokens: {special_tokens}")
    
    return num_added


def main(config_fpath: str):
    
    # Set environment variable to suppress tokenizers parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # load config
    print(f"Loading config from {config_fpath}")
    config = OmegaConf.load(config_fpath)
    checkpoints_dir = os.path.join(config.save_root, config.run_name)
    LOGGER.info(f"Logging to: {checkpoints_dir}")

    restore_from = config.restore_from
    paired_dataset_path = config.paired_dataset_path

    print(f"Loading checkpoint from {restore_from}")
    tokenizer = AutoTokenizer.from_pretrained(restore_from)
    
    # Add conditioning tokens to tokenizer
    num_added = add_conditioning_tokens(tokenizer, paired_dataset_path)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(restore_from, torch_dtype="auto")
    
    # Resize model embeddings if tokens were added
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized model embeddings to {len(tokenizer)}")

    g2p = phonemizer.backend.EspeakBackend(
        language='ar',  # Arabic
        preserve_punctuation=True,
        with_stress=True,
        words_mismatch="ignore",
        language_switch="remove-flags"
    )
    partial_preprocess = partial(
        preprocess_sample,
        tokenizer=tokenizer,
        max_len=config.max_seq_len,
        g2p=g2p,
    )

    # Load paired dataset
    print(f"Loading paired dataset from {paired_dataset_path}")
    paired_dataset = load_from_disk(paired_dataset_path)
    print(f"Loaded {len(paired_dataset)} training pairs")
    
    # Preprocess dataset (remove audio columns to save memory during training)
    remove_cols = ["ref_audio", "target_audio"]
    paired_dataset = paired_dataset.map(
        partial_preprocess, 
        remove_columns=remove_cols,
        num_proc=40,
        desc="Preprocessing"
    )
    
    # Filter out invalid samples (samples that failed preprocessing)
    original_size = len(paired_dataset)
    paired_dataset = paired_dataset.filter(
        lambda x: x["valid"] == True,
        num_proc=40,
        desc="Filtering valid samples"
    )
    filtered_size = len(paired_dataset)
    if filtered_size < original_size:
        print(f"Filtered out {original_size - filtered_size} failed samples. {filtered_size} samples remaining.")
    
    # Remove the 'valid' column as it's no longer needed
    paired_dataset = paired_dataset.remove_columns(["valid"])

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        do_train=True,
        learning_rate=config.lr,
        max_steps=config.max_steps,
        bf16=True,
        per_device_train_batch_size=config.per_device_train_batch_size,
        warmup_ratio=config.warmup_ratio,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        ignore_data_skip=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        torch_compile=True,
        dataloader_num_workers=64,
    )

    # Create data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,  # For better GPU efficiency
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=paired_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(checkpoints_dir)
    tokenizer.save_pretrained(checkpoints_dir)


if __name__ == "__main__":
    Fire(main)

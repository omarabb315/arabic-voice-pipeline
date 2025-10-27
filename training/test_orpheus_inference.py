"""
Inference script for Orpheus voice cloning model.

Usage:
    python test_orpheus_inference.py \
        --model_path="checkpoints/orpheus-arabic-voice-cloning/checkpoint-1000" \
        --ref_text="مرحباً، كيف حالك؟" \
        --ref_gender="male" \
        --ref_dialect="MSA" \
        --ref_tone="neutral" \
        --target_text="أهلاً وسهلاً بك في هذا البرنامج" \
        --target_gender="male" \
        --target_dialect="MSA" \
        --target_tone="formal" \
        --ref_audio_path="path/to/reference_audio.wav" \
        --output_path="output_audio.wav"
"""

import warnings
import torch
import torchaudio
import torchaudio.transforms as T
import phonemizer
import numpy as np
from fire import Fire
from unsloth import FastLanguageModel
from snac import SNAC
from pathlib import Path

warnings.filterwarnings("ignore")

# Orpheus special token IDs
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


def load_and_preprocess_audio(audio_path: str, target_sr: int = 16000):
    """Load audio file and resample to target sample rate."""
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    
    return waveform.squeeze().numpy()


def tokenise_audio_snac(waveform, snac_model, ds_sample_rate=16000):
    """Tokenize audio using SNAC model."""
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


def redistribute_codes(code_list):
    """Redistribute codes back to 3 layers for SNAC decoding."""
    layer_1 = []
    layer_2 = []
    layer_3 = []
    
    for i in range((len(code_list) + 1) // 7):
        layer_1.append(code_list[7*i] - AUDIO_TOKENS_START)
        layer_2.append(code_list[7*i+1] - AUDIO_TOKENS_START - 4096)
        layer_3.append(code_list[7*i+2] - AUDIO_TOKENS_START - (2*4096))
        layer_3.append(code_list[7*i+3] - AUDIO_TOKENS_START - (3*4096))
        layer_2.append(code_list[7*i+4] - AUDIO_TOKENS_START - (4*4096))
        layer_3.append(code_list[7*i+5] - AUDIO_TOKENS_START - (5*4096))
        layer_3.append(code_list[7*i+6] - AUDIO_TOKENS_START - (6*4096))
    
    codes = [
        torch.tensor(layer_1).unsqueeze(0),
        torch.tensor(layer_2).unsqueeze(0),
        torch.tensor(layer_3).unsqueeze(0)
    ]
    
    return codes


def infer_voice_cloning(
    model_path: str = "checkpoints/orpheus-arabic-voice-cloning/checkpoint-1000",
    ref_audio_path: str = None,
    ref_text: str = "مرحباً، كيف حالك؟",
    ref_gender: str = "male",
    ref_dialect: str = "MSA",
    ref_tone: str = "neutral",
    target_text: str = "أهلاً وسهلاً بك في هذا البرنامج",
    target_gender: str = "male",
    target_dialect: str = "MSA",
    target_tone: str = "formal",
    output_path: str = "output_audio.wav",
    max_new_tokens: int = 1200,
    temperature: float = 0.6,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
):
    """
    Perform voice cloning inference with Orpheus model.
    
    Args:
        model_path: Path to finetuned model checkpoint
        ref_audio_path: Path to reference audio file
        ref_text: Reference text (Arabic)
        ref_gender: Reference gender (male/female/unknown)
        ref_dialect: Reference dialect (MSA/Egyptian/etc)
        ref_tone: Reference tone (neutral/formal/etc)
        target_text: Target text to synthesize
        target_gender: Target gender
        target_dialect: Target dialect
        target_tone: Target tone
        output_path: Path to save output audio
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling
        repetition_penalty: Repetition penalty
    """
    
    print("="*60)
    print("Orpheus Voice Cloning Inference")
    print("="*60)
    
    # Load model
    print(f"\n1. Loading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)  # Enable faster inference
    
    # Load SNAC model
    print("2. Loading SNAC model...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model = snac_model.to("cuda")
    
    # Initialize phonemizer
    print("3. Initializing phonemizer...")
    g2p = phonemizer.backend.EspeakBackend(
        language='ar',
        preserve_punctuation=True,
        with_stress=True,
        words_mismatch="ignore",
        language_switch="remove-flags"
    )
    
    # Load and tokenize reference audio
    if ref_audio_path:
        print(f"4. Loading reference audio from {ref_audio_path}...")
        ref_audio = load_and_preprocess_audio(ref_audio_path, target_sr=16000)
        ref_codes = tokenise_audio_snac(ref_audio, snac_model, ds_sample_rate=16000)
        print(f"   Reference audio encoded to {len(ref_codes)} tokens")
    else:
        # If no reference audio, we can't do true voice cloning
        # You might want to use a default or raise an error
        raise ValueError("Reference audio is required for voice cloning")
    
    # Phonemize texts
    print("5. Phonemizing texts...")
    ref_phones = g2p.phonemize([ref_text])
    target_phones = g2p.phonemize([target_text])
    
    if not ref_phones or not ref_phones[0] or not target_phones or not target_phones[0]:
        raise ValueError("Phonemization failed")
    
    ref_phones = ' '.join(ref_phones[0].split())
    target_phones = ' '.join(target_phones[0].split())
    
    print(f"   Ref phones: {ref_phones}")
    print(f"   Target phones: {target_phones}")
    
    # Create conditioned text
    ref_conditioned = f"[{ref_gender}|{ref_dialect}|{ref_tone}] {ref_phones}"
    target_conditioned = f"[{target_gender}|{target_dialect}|{target_tone}] {target_phones}"
    combined_text = f"{ref_conditioned} | {target_conditioned}"
    
    print(f"\n6. Creating input prompt...")
    print(f"   Combined text: {combined_text}")
    
    # Tokenize text
    text_ids = tokenizer.encode(combined_text, add_special_tokens=True)
    
    # Create input sequence: [SOH] [SOT] text [EOT] [EOH] [SOA] [SOS] ref_codes
    input_ids = [START_OF_HUMAN, START_OF_TEXT] + text_ids + [END_OF_TEXT, END_OF_HUMAN]
    input_ids += [START_OF_AI, START_OF_SPEECH] + ref_codes
    
    # Don't add end tokens - let model generate target codes and end tokens
    
    # Convert to tensor
    input_ids = torch.tensor([input_ids]).to("cuda")
    attention_mask = torch.ones_like(input_ids)
    
    print(f"   Input length: {input_ids.shape[1]} tokens")
    
    # Generate
    print(f"\n7. Generating target audio codes...")
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=1,
            eos_token_id=END_OF_SPEECH,
            use_cache=True,
        )
    
    print(f"   Generated {generated_ids.shape[1]} tokens")
    
    # Extract generated codes
    # Find START_OF_SPEECH token in generated sequence
    generated_codes = generated_ids[0].tolist()
    
    # Find the last occurrence of START_OF_SPEECH (the one model generated)
    try:
        # Find where our ref codes ended and target codes begin
        # This is after the input length
        target_codes_start = input_ids.shape[1]
        target_codes = generated_codes[target_codes_start:]
        
        # Remove end tokens
        if END_OF_SPEECH in target_codes:
            eos_idx = target_codes.index(END_OF_SPEECH)
            target_codes = target_codes[:eos_idx]
        if END_OF_AI in target_codes:
            eoa_idx = target_codes.index(END_OF_AI)
            target_codes = target_codes[:eoa_idx]
        
        print(f"   Extracted {len(target_codes)} target audio tokens")
        
        # Trim to multiple of 7
        trim_length = (len(target_codes) // 7) * 7
        target_codes = target_codes[:trim_length]
        
        if len(target_codes) == 0:
            raise ValueError("No audio codes generated")
        
    except Exception as e:
        print(f"   Error extracting codes: {e}")
        raise
    
    # Decode audio using SNAC
    print(f"\n8. Decoding audio with SNAC...")
    snac_model = snac_model.to("cpu")  # Move to CPU for decoding
    codes_redistributed = redistribute_codes(target_codes)
    
    with torch.inference_mode():
        audio_hat = snac_model.decode(codes_redistributed)
    
    # Save audio
    print(f"9. Saving audio to {output_path}...")
    audio_np = audio_hat.detach().squeeze().cpu().numpy()
    
    # Save as WAV file
    import soundfile as sf
    sf.write(output_path, audio_np, 24000)  # SNAC outputs 24kHz
    
    print(f"\n{'='*60}")
    print(f"✅ Voice cloning complete!")
    print(f"📁 Output saved to: {output_path}")
    print(f"{'='*60}")
    
    return output_path


if __name__ == "__main__":
    Fire(infer_voice_cloning)


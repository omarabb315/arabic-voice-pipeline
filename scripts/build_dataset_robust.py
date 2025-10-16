"""
Robust dataset builder with disk-based storage.

Saves each segment immediately to disk to avoid RAM issues.
Can resume from any point - checks which segments already exist.
"""

import os
import json
import tempfile
import numpy as np
import torch
import librosa
import time
from pathlib import Path
from tqdm import tqdm
from google.cloud import storage
from datasets import Dataset, Audio, Features, Value, Sequence, concatenate_datasets
from neucodec import NeuCodec
from fire import Fire


def build_segments_dataset_robust(
    gcp_bucket_name: str = "audio-data-gemini",
    channels: list[str] = None,
    segments_cache_dir: str = "segments_cache/",
    output_path: str = "dataset_cache/segments_dataset/",
    codec_device: str = "cuda" if torch.cuda.is_available() else "cpu",
    force_rebuild: bool = False,
    channel_mapping: dict[str, str] = None,
    detailed_log_file: str = "build_dataset_detailed.log",
    skip_codec_encoding: bool = False,
    create_hf_dataset: bool = True,
):
    """
    Build dataset with disk-based segment storage (more robust).
    
    Args:
        segments_cache_dir: Directory to save individual segment files
        create_hf_dataset: If True, create HuggingFace dataset at the end
                          If False, just save segments (can create dataset later)
    """
    
    print("=" * 70)
    print("Robust Dataset Builder (Disk-Based)")
    print("=" * 70)
    print()
    
    # Create segments cache directory
    os.makedirs(segments_cache_dir, exist_ok=True)
    
    # Initialize GCP client
    print("Connecting to GCP bucket...")
    client = storage.Client()
    bucket = client.bucket(gcp_bucket_name)
    
    # Load codec model (only if encoding is needed)
    codec = None
    if not skip_codec_encoding:
        print(f"Loading NeuCodec on {codec_device}...")
        codec = NeuCodec.from_pretrained("neuphonic/neucodec")
        codec.eval().to(codec_device)
    else:
        print(f"⚡ Skipping codec loading (skip_codec_encoding=True)")
    
    # Open detailed log file
    detailed_log = open(detailed_log_file, 'w', buffering=1)
    detailed_log.write("=== Build Dataset Detailed Log ===\n\n")
    
    # Initialize channel mapping
    if channel_mapping is None:
        channel_mapping = {}
    
    # Track statistics
    total_files_processed = 0
    total_segments_saved = 0
    
    # Process each channel
    for channel in tqdm(channels, desc="Processing channels"):
        print(f"\n{'='*60}")
        print(f"Processing channel: {channel}")
        print(f"{'='*60}")
        
        # Get audio channel name from mapping
        audio_channel_name = channel_mapping.get(channel, channel)
        if channel in channel_mapping:
            print(f"Using channel mapping: '{channel}' -> '{audio_channel_name}'")
        
        # Create channel directory in cache
        channel_cache_dir = os.path.join(segments_cache_dir, channel)
        os.makedirs(channel_cache_dir, exist_ok=True)
        
        # List all transcription JSONs
        transcripts_prefix = f"processed_runs/processed_{channel}/transcripts/"
        transcript_blobs = list(bucket.list_blobs(prefix=transcripts_prefix))
        transcript_blobs = [b for b in transcript_blobs if b.name.endswith('.json')]
        
        print(f"Found {len(transcript_blobs)} transcript files")
        
        for transcript_blob in tqdm(transcript_blobs, desc=f"Processing {channel} transcripts"):
            try:
                # Download and parse transcript JSON
                transcript_json = json.loads(transcript_blob.download_as_text())
                
                # Handle both list and dict formats
                if isinstance(transcript_json, list):
                    segments_data = transcript_json
                elif isinstance(transcript_json, dict) and "segments" in transcript_json:
                    segments_data = transcript_json["segments"]
                else:
                    continue
                
                if not segments_data:
                    continue
                
                # Extract audio filename
                transcript_filename = Path(transcript_blob.name).name
                audio_id_base = transcript_filename.replace('audio_', '').replace('.json', '')
                
                # Note: No longer skipping problematic files - will convert with ffmpeg below
                
                # Try to find the actual audio file
                audio_dir_prefix = f"{audio_channel_name}/audio/{audio_id_base}"
                possible_audio_blobs = list(bucket.list_blobs(prefix=audio_dir_prefix))
                
                if not possible_audio_blobs:
                    continue
                
                # Use the first matching audio file
                audio_blob = possible_audio_blobs[0]
                audio_filename = Path(audio_blob.name).name
                
                # Check which segments need to be processed AND will pass validation
                segments_to_process = []
                for seg_idx, segment in enumerate(segments_data):
                    segment_id = f"{audio_id_base}_seg_{seg_idx}"
                    segment_file = os.path.join(channel_cache_dir, f"{segment_id}.npz")
                    
                    # Skip if already exists (unless force rebuild)
                    if os.path.exists(segment_file) and not force_rebuild:
                        continue
                    
                    # Pre-validate segment before downloading audio
                    text = segment.get('text', '')
                    if not text or not text.strip():
                        continue  # Skip empty text
                    
                    start_ms = segment.get('start_ms', 0)
                    end_ms = segment.get('end_ms', 0)
                    duration_sec = (end_ms - start_ms) / 1000.0
                    
                    if duration_sec < 0.3:
                        continue  # Skip too short segments
                    
                    # This segment needs processing and will pass validation
                    segments_to_process.append((seg_idx, segment))
                
                # Skip if no segments will actually be processed
                if not segments_to_process:
                    continue
                
                log_msg = f"\n📄 {audio_filename}: {len(segments_to_process)} new segments to process\n"
                print(log_msg, end='')
                detailed_log.write(log_msg)
                
                # Download audio (use main disk, not /tmp to avoid space issues)
                temp_dir = "temp_audio_downloads"
                os.makedirs(temp_dir, exist_ok=True)
                temp_audio_path = None
                try:
                    download_start = time.time()
                    temp_audio_path = os.path.join(temp_dir, f"{audio_id_base}_{os.getpid()}.wav")
                    audio_blob.download_to_filename(temp_audio_path)
                    download_time = time.time() - download_start
                    
                    audio_size_mb = os.path.getsize(temp_audio_path) / (1024 * 1024)
                    download_speed = audio_size_mb / download_time if download_time > 0 else 0
                    
                    dl_msg = f"   ⬇️  Downloaded: {audio_size_mb:.2f} MB in {download_time:.2f}s ({download_speed:.2f} MB/s)\n"
                    print(dl_msg, end='')
                    detailed_log.write(dl_msg)
                    
                    # Load audio (use ffmpeg for .webm/.part files, librosa for others)
                    file_ext = Path(temp_audio_path).suffix.lower()
                    audio_format = Path(audio_filename).suffix.lower()
                    
                    # Force ffmpeg conversion for known problematic formats
                    use_ffmpeg = '.webm' in audio_filename.lower() or '.part' in audio_filename.lower() or '.m4a' in audio_filename.lower()
                    
                    if use_ffmpeg:
                        # Convert with ffmpeg for .webm, .part, .m4a files
                        try:
                            import subprocess
                            convert_msg = f"   🔄 Converting {audio_format} with ffmpeg...\n"
                            print(convert_msg, end='')
                            detailed_log.write(convert_msg)
                            
                            converted_path = temp_audio_path + ".converted.wav"
                            
                            # Convert to standard wav with ffmpeg (with timeout for large files)
                            result = subprocess.run([
                                'ffmpeg', '-i', temp_audio_path,
                                '-ar', '16000', '-ac', '1',
                                '-acodec', 'pcm_s16le',  # Force PCM codec for compatibility
                                '-y', converted_path
                            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout for very large files
                            
                            if result.returncode != 0 or not os.path.exists(converted_path):
                                error_msg = f"   ⚠️ FFmpeg conversion failed (rc={result.returncode})\n"
                                if result.stderr:
                                    error_msg += f"   Error: {result.stderr[:200]}\n"
                                error_msg += "   Skipping this file...\n"
                                print(error_msg, end='')
                                detailed_log.write(error_msg)
                                continue
                            
                            # Load the converted file
                            load_start = time.time()
                            audio_data, sr = librosa.load(converted_path, sr=16000, mono=True)
                            load_time = time.time() - load_start
                            
                            # Clean up converted file
                            if os.path.exists(converted_path):
                                os.unlink(converted_path)
                            
                            load_msg = f"   ✅ Converted and loaded: {len(audio_data)/sr:.1f}s audio in {load_time:.2f}s\n"
                            print(load_msg, end='')
                            detailed_log.write(load_msg)
                            
                        except Exception as convert_error:
                            error_msg = f"   ⚠️ Conversion error: {convert_error}\n   Skipping...\n"
                            print(error_msg, end='')
                            detailed_log.write(error_msg)
                            continue
                    else:
                        # Load directly with librosa for standard formats
                        try:
                            load_start = time.time()
                            audio_data, sr = librosa.load(temp_audio_path, sr=16000, mono=True)
                            load_time = time.time() - load_start
                            
                            load_msg = f"   🎵 Loaded: {len(audio_data)/sr:.1f}s audio in {load_time:.2f}s\n"
                            print(load_msg, end='')
                            detailed_log.write(load_msg)
                        except Exception as load_error:
                            error_msg = f"   ⚠️ Load failed: {load_error}\n   Skipping...\n"
                            print(error_msg, end='')
                            detailed_log.write(error_msg)
                            continue
                    
                    # Process segments - collect in list for this audio file
                    audio_segments_list = []
                    for seg_idx, segment in segments_to_process:
                        try:
                            # Extract segment info (already validated)
                            start_ms = segment.get('start_ms', 0)
                            end_ms = segment.get('end_ms', 0)
                            text = segment.get('text', '')
                            
                            # Extract audio
                            start_sample = int(start_ms * sr / 1000)
                            end_sample = int(end_ms * sr / 1000)
                            audio_segment = audio_data[start_sample:end_sample]
                            
                            # Encode if needed
                            if skip_codec_encoding:
                                vq_codes = []
                            else:
                                with torch.no_grad():
                                    audio_tensor = torch.from_numpy(audio_segment).float().unsqueeze(0).unsqueeze(0)
                                    codes = codec.encode_code(audio_or_path=audio_tensor)
                                    codes = codes.squeeze(0).squeeze(0)
                                    if codes.is_cuda:
                                        codes = codes.cpu()
                                    vq_codes = codes.numpy().tolist()
                                
                                if codec_device == "cuda":
                                    torch.cuda.empty_cache()
                            
                            # Save segment immediately as .npz file
                            segment_data = {
                                'audio': audio_segment,
                                'codes': vq_codes,
                                'text': text,
                                'speaker_id': segment.get('speaker_id', 'UNKNOWN'),
                                'channel': channel,
                                'gender': segment.get('gender', 'unknown'),
                                'dialect': segment.get('dialect', 'unknown'),
                                'tone': segment.get('tone', 'neutral'),
                                'duration_sec': (end_ms - start_ms) / 1000.0,
                                'audio_id': f"{audio_id_base}_seg_{seg_idx}",
                            }
                            
                            # Save as compressed .npz file
                            segment_file = os.path.join(channel_cache_dir, f"{audio_id_base}_seg_{seg_idx}.npz")
                            np.savez_compressed(segment_file, **segment_data)
                            audio_segments_list.append(segment_data)  # Just for counting
                            
                        except Exception as e:
                            print(f"⚠️ Error processing segment {seg_idx}: {e}")
                            continue
                    
                    # Summary for this audio file
                    if audio_segments_list:
                        segments_saved = len(audio_segments_list)
                        total_segments_saved += segments_saved
                        
                        summary_msg = f"   ✅ Saved: {segments_saved} segments to disk\n"
                        print(summary_msg, end='')
                        detailed_log.write(summary_msg)
                    
                    total_files_processed += 1
                    
                finally:
                    if temp_audio_path and os.path.exists(temp_audio_path):
                        os.unlink(temp_audio_path)
                
            except KeyboardInterrupt:
                print("\n⚠️ Interrupted by user")
                raise
            except Exception as e:
                import traceback
                print(f"\n⚠️ Error processing transcript {transcript_blob.name}: {e}")
                print(f"   Traceback: {traceback.format_exc()}")
                continue
    
    detailed_log.close()
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  - Files processed: {total_files_processed}")
    print(f"  - Segments saved: {total_segments_saved}")
    print(f"  - Segments cache: {segments_cache_dir}")
    print(f"{'='*60}\n")
    
    # Create HuggingFace dataset if requested
    if create_hf_dataset:
        print("Creating HuggingFace Dataset from cached segments...")
        create_dataset_from_cache(segments_cache_dir, output_path)
    else:
        print(f"Segments saved to {segments_cache_dir}")
        print(f"Run with --create_hf_dataset=True to create HuggingFace dataset")


def create_dataset_from_cache(segments_cache_dir: str, output_path: str):
    """Concatenate all mini HF datasets into final dataset."""
    from datasets import concatenate_datasets, load_from_disk
    
    print("Loading mini-datasets from cache...")
    all_mini_datasets = []
    total_segments = 0
    
    # Walk through all channel directories
    for channel_dir in Path(segments_cache_dir).glob("*"):
        if not channel_dir.is_dir():
            continue
        
        # Each subdirectory is a mini-dataset for one audio file
        audio_dataset_dirs = [d for d in channel_dir.iterdir() if d.is_dir()]
        print(f"  - {channel_dir.name}: {len(audio_dataset_dirs)} audio files")
        
        for audio_dir in tqdm(audio_dataset_dirs, desc=f"Loading {channel_dir.name}"):
            try:
                mini_ds = load_from_disk(str(audio_dir))
                all_mini_datasets.append(mini_ds)
                total_segments += len(mini_ds)
            except Exception as e:
                print(f"⚠️ Error loading {audio_dir}: {e}")
                continue
    
    if len(all_mini_datasets) == 0:
        print("❌ No datasets found in cache!")
        return
    
    print(f"\nConcatenating {len(all_mini_datasets)} mini-datasets ({total_segments} total segments)...")
    final_dataset = concatenate_datasets(all_mini_datasets)
    
    print(f"Saving final dataset to {output_path}...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    final_dataset.save_to_disk(output_path)
    
    print(f"✅ Dataset saved to: {output_path}")
    print(f"📊 Dataset info:")
    print(f"   - Total segments: {len(final_dataset)}")
    print(f"   - Channels: {set(final_dataset['channel'])}")
    print(f"   - Unique speakers: {len(set(final_dataset['speaker_id']))}")
    print(f"   - Total duration: {sum(final_dataset['duration_sec']) / 3600:.2f} hours")


if __name__ == "__main__":
    Fire(build_segments_dataset_robust)


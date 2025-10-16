"""
Upload segments_cache/ directory to S3 bucket.

Run this on GCP T4 after Stage 1 to transfer segments to S3
for processing on H200.
"""

import os
from pathlib import Path
from fire import Fire
from tqdm import tqdm
import concurrent.futures
from s3_utils import S3Client
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def upload_segments(
    source_dir: str = "segments_cache/",
    s3_prefix: str = "neutts_segments/stage1/",
    max_workers: int = 4,
    dry_run: bool = False,
):
    """
    Upload all .npz segment files to S3.
    
    Args:
        source_dir: Local directory containing segments
        s3_prefix: S3 prefix (path in bucket)
        max_workers: Number of parallel upload threads
        dry_run: If True, only show what would be uploaded
    """
    
    print("=" * 70)
    print("Upload Segments to S3")
    print("=" * 70)
    print(f"Source: {source_dir}")
    print(f"S3 Prefix: {s3_prefix}")
    print(f"Parallel workers: {max_workers}")
    print()
    
    # Initialize S3 client
    if not dry_run:
        s3_client = S3Client()
    
    # Find all .npz files
    source_path = Path(source_dir)
    if not source_path.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return
    
    npz_files = list(source_path.rglob("*.npz"))
    print(f"Found {len(npz_files):,} .npz files to upload")
    
    if len(npz_files) == 0:
        logger.error("No .npz files found!")
        return
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in npz_files)
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    print()
    
    if dry_run:
        print("DRY RUN - No files will be uploaded")
        for npz_file in npz_files[:10]:
            relative_path = npz_file.relative_to(source_path)
            s3_key = os.path.join(s3_prefix, str(relative_path))
            print(f"  Would upload: {npz_file} -> {s3_key}")
        if len(npz_files) > 10:
            print(f"  ... and {len(npz_files) - 10} more files")
        return
    
    # Check for existing files in S3
    print("Checking for existing files in S3...")
    existing_files = set(s3_client.list_files(prefix=s3_prefix, suffix=".npz"))
    print(f"Found {len(existing_files):,} existing files in S3")
    print()
    
    # Filter files to upload (skip existing)
    files_to_upload = []
    for npz_file in npz_files:
        relative_path = npz_file.relative_to(source_path)
        s3_key = os.path.join(s3_prefix, str(relative_path))
        
        if s3_key in existing_files:
            logger.debug(f"Skipping existing file: {s3_key}")
        else:
            files_to_upload.append((npz_file, s3_key))
    
    print(f"Files to upload: {len(files_to_upload):,}")
    print(f"Files to skip (already in S3): {len(npz_files) - len(files_to_upload):,}")
    print()
    
    if len(files_to_upload) == 0:
        print("✅ All files already uploaded!")
        return
    
    # Upload files
    uploaded = 0
    failed = 0
    
    def upload_file(args):
        local_path, s3_key = args
        try:
            success = s3_client.upload_file(
                str(local_path),
                s3_key,
                show_progress=False  # Disable individual progress for parallel uploads
            )
            return (success, s3_key)
        except Exception as e:
            logger.error(f"Error uploading {local_path}: {e}")
            return (False, s3_key)
    
    print(f"Uploading {len(files_to_upload):,} files with {max_workers} workers...")
    print()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(files_to_upload), desc="Overall progress") as pbar:
            for success, s3_key in executor.map(upload_file, files_to_upload):
                if success:
                    uploaded += 1
                else:
                    failed += 1
                    logger.error(f"Failed to upload: {s3_key}")
                pbar.update(1)
    
    print()
    print("=" * 70)
    print("Upload Complete!")
    print("=" * 70)
    print(f"✅ Uploaded: {uploaded:,} files")
    if failed > 0:
        print(f"❌ Failed: {failed:,} files")
    print()
    print("Next step: Run add_codes_to_dataset_s3.py on H200")
    print()


if __name__ == "__main__":
    Fire(upload_segments)


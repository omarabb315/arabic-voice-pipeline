"""
Upload segments_cache/ directory to GCS bucket.

Run this on GCP T4 if segments aren't already in GCS.
"""

import os
from pathlib import Path
from fire import Fire
from tqdm import tqdm
import concurrent.futures
from gcs_utils import GCSClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def upload_segments(
    source_dir: str = "segments_cache/",
    gcs_bucket: str = "audio-data-gemini",
    gcs_prefix: str = "segments_cache/",
    max_workers: int = 4,
    dry_run: bool = False,
):
    """
    Upload all .npz segment files to GCS.
    
    Args:
        source_dir: Local directory containing segments
        gcs_bucket: GCS bucket name
        gcs_prefix: GCS prefix (path in bucket)
        max_workers: Number of parallel upload threads
        dry_run: If True, only show what would be uploaded
    """
    
    print("=" * 70)
    print("Upload Segments to GCS")
    print("=" * 70)
    print(f"Source: {source_dir}")
    print(f"GCS Bucket: {gcs_bucket}")
    print(f"GCS Prefix: {gcs_prefix}")
    print(f"Parallel workers: {max_workers}")
    print()
    
    # Initialize GCS client
    if not dry_run:
        gcs_client = GCSClient(bucket_name=gcs_bucket)
    
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
            gcs_path = os.path.join(gcs_prefix, str(relative_path))
            print(f"  Would upload: {npz_file} -> gs://{gcs_bucket}/{gcs_path}")
        if len(npz_files) > 10:
            print(f"  ... and {len(npz_files) - 10} more files")
        return
    
    # Check for existing files in GCS
    print("Checking for existing files in GCS...")
    existing_files = set(gcs_client.list_files(prefix=gcs_prefix, suffix=".npz"))
    print(f"Found {len(existing_files):,} existing files in GCS")
    print()
    
    # Filter files to upload (skip existing)
    files_to_upload = []
    for npz_file in npz_files:
        relative_path = npz_file.relative_to(source_path)
        gcs_path = os.path.join(gcs_prefix, str(relative_path))
        
        if gcs_path in existing_files:
            logger.debug(f"Skipping existing file: {gcs_path}")
        else:
            files_to_upload.append((npz_file, gcs_path))
    
    print(f"Files to upload: {len(files_to_upload):,}")
    print(f"Files to skip (already in GCS): {len(npz_files) - len(files_to_upload):,}")
    print()
    
    if len(files_to_upload) == 0:
        print("✅ All files already uploaded!")
        return
    
    # Upload files
    uploaded = 0
    failed = 0
    
    def upload_file(args):
        local_path, gcs_path = args
        try:
            success = gcs_client.upload_file(
                str(local_path),
                gcs_path,
                show_progress=False  # Disable individual progress for parallel uploads
            )
            return (success, gcs_path)
        except Exception as e:
            logger.error(f"Error uploading {local_path}: {e}")
            return (False, gcs_path)
    
    print(f"Uploading {len(files_to_upload):,} files with {max_workers} workers...")
    print()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(files_to_upload), desc="Overall progress") as pbar:
            for success, gcs_path in executor.map(upload_file, files_to_upload):
                if success:
                    uploaded += 1
                else:
                    failed += 1
                    logger.error(f"Failed to upload: {gcs_path}")
                pbar.update(1)
    
    print()
    print("=" * 70)
    print("Upload Complete!")
    print("=" * 70)
    print(f"✅ Uploaded: {uploaded:,} files")
    if failed > 0:
        print(f"❌ Failed: {failed:,} files")
    print()
    print(f"Files are now in gs://{gcs_bucket}/{gcs_prefix}")
    print()
    print("Next step: Run add_codes_to_dataset_gcs.py on H200")
    print()


if __name__ == "__main__":
    Fire(upload_segments)


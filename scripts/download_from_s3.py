"""
Download encoded segments from S3 to local storage.

Useful for:
- Downloading encoded segments to GCP T4 for HF dataset creation
- Validation and spot-checking
- Creating backups
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


def download_segments(
    s3_prefix: str = "neutts_segments/stage2_with_codes/",
    output_dir: str = "segments_cache_encoded/",
    max_workers: int = 4,
    channels: list = None,
    limit: int = None,
):
    """
    Download encoded segment files from S3.
    
    Args:
        s3_prefix: S3 prefix where encoded segments are stored
        output_dir: Local directory to save segments
        max_workers: Number of parallel download threads
        channels: Optional list of channels to download (e.g., ['ZadTVchannel'])
        limit: Optional limit on number of files to download (for testing)
    """
    
    print("=" * 70)
    print("Download Segments from S3")
    print("=" * 70)
    print(f"S3 Prefix: {s3_prefix}")
    print(f"Output: {output_dir}")
    print(f"Parallel workers: {max_workers}")
    if channels:
        print(f"Channels filter: {channels}")
    if limit:
        print(f"Limit: {limit} files")
    print()
    
    # Initialize S3 client
    s3_client = S3Client()
    
    # List all files
    print("Listing files in S3...")
    all_files = s3_client.list_files(prefix=s3_prefix, suffix=".npz")
    print(f"Found {len(all_files):,} files in S3")
    
    # Filter by channels if specified
    if channels:
        filtered_files = []
        for f in all_files:
            for channel in channels:
                if f"/{channel}/" in f:
                    filtered_files.append(f)
                    break
        all_files = filtered_files
        print(f"After channel filter: {len(all_files):,} files")
    
    # Apply limit if specified
    if limit:
        all_files = all_files[:limit]
        print(f"After limit: {len(all_files):,} files")
    
    if len(all_files) == 0:
        logger.error("No files to download!")
        return
    
    print()
    
    # Check for existing files
    output_path = Path(output_dir)
    existing_files = set()
    if output_path.exists():
        print("Checking for existing files...")
        for s3_key in all_files:
            relative_path = s3_key.replace(s3_prefix, "")
            local_path = output_path / relative_path
            if local_path.exists():
                existing_files.add(s3_key)
        print(f"Found {len(existing_files):,} existing files")
    
    # Filter files to download
    files_to_download = [f for f in all_files if f not in existing_files]
    print(f"Files to download: {len(files_to_download):,}")
    print(f"Files to skip: {len(existing_files):,}")
    print()
    
    if len(files_to_download) == 0:
        print("✅ All files already downloaded!")
        return
    
    # Download files
    downloaded = 0
    failed = 0
    
    def download_file(s3_key):
        try:
            relative_path = s3_key.replace(s3_prefix, "")
            local_path = output_path / relative_path
            
            success = s3_client.download_file(
                s3_key,
                str(local_path),
                show_progress=False
            )
            return (success, s3_key)
        except Exception as e:
            logger.error(f"Error downloading {s3_key}: {e}")
            return (False, s3_key)
    
    print(f"Downloading {len(files_to_download):,} files with {max_workers} workers...")
    print()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(files_to_download), desc="Overall progress") as pbar:
            for success, s3_key in executor.map(download_file, files_to_download):
                if success:
                    downloaded += 1
                else:
                    failed += 1
                pbar.update(1)
    
    print()
    print("=" * 70)
    print("Download Complete!")
    print("=" * 70)
    print(f"✅ Downloaded: {downloaded:,} files")
    if failed > 0:
        print(f"❌ Failed: {failed:,} files")
    print(f"📂 Location: {output_dir}")
    print()


if __name__ == "__main__":
    Fire(download_segments)


"""
GCS utilities for Google Cloud Storage operations.

Supports batch processing with progress tracking for large datasets.
"""

import os
from pathlib import Path
from google.cloud import storage
from tqdm import tqdm
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GCSClient:
    """GCS client wrapper for dataset operations."""
    
    def __init__(
        self,
        bucket_name: Optional[str] = None,
        credentials_path: Optional[str] = None,
    ):
        """
        Initialize GCS client.
        
        Args:
            bucket_name: GCS bucket name (defaults to env var GCS_BUCKET_NAME)
            credentials_path: Path to service account JSON (defaults to env var GOOGLE_APPLICATION_CREDENTIALS)
        """
        self.credentials_path = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        self.bucket_name = bucket_name or os.environ.get("GCS_BUCKET_NAME", "audio-data-gemini")
        
        if self.credentials_path and not os.path.exists(self.credentials_path):
            raise ValueError(f"Credentials file not found: {self.credentials_path}")
        
        # Create client
        if self.credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
        
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)
        
        logger.info(f"GCS client initialized for bucket: {self.bucket_name}")
    
    def upload_file(
        self,
        local_path: str,
        gcs_path: str,
        show_progress: bool = True,
    ) -> bool:
        """
        Upload a file to GCS.
        
        Args:
            local_path: Path to local file
            gcs_path: GCS path (without gs://bucket/)
            show_progress: Show progress bar
            
        Returns:
            True if successful
        """
        try:
            blob = self.bucket.blob(gcs_path)
            
            if show_progress:
                file_size = os.path.getsize(local_path)
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Uploading {os.path.basename(local_path)}") as pbar:
                    def callback(bytes_transferred):
                        pbar.update(bytes_transferred - pbar.n)
                    
                    blob.upload_from_filename(local_path)
            else:
                blob.upload_from_filename(local_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            return False
    
    def download_file(
        self,
        gcs_path: str,
        local_path: str,
        show_progress: bool = True,
    ) -> bool:
        """
        Download a file from GCS.
        
        Args:
            gcs_path: GCS path (without gs://bucket/)
            local_path: Path to save file locally
            show_progress: Show progress bar
            
        Returns:
            True if successful
        """
        try:
            # Create parent directory if needed
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            blob = self.bucket.blob(gcs_path)
            
            if show_progress:
                file_size = blob.size
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Downloading {os.path.basename(gcs_path)}") as pbar:
                    def callback(bytes_transferred):
                        pbar.update(bytes_transferred - pbar.n)
                    
                    blob.download_to_filename(local_path)
            else:
                blob.download_to_filename(local_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {gcs_path}: {e}")
            return False
    
    def list_files(
        self,
        prefix: str = "",
        suffix: str = "",
    ) -> List[str]:
        """
        List files in GCS bucket with optional prefix and suffix filtering.
        
        Args:
            prefix: Only list files starting with this prefix
            suffix: Only list files ending with this suffix
            
        Returns:
            List of GCS paths
        """
        try:
            files = []
            blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
            
            for blob in blobs:
                if suffix and not blob.name.endswith(suffix):
                    continue
                files.append(blob.name)
            
            logger.info(f"Found {len(files)} files with prefix '{prefix}' and suffix '{suffix}'")
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    def file_exists(self, gcs_path: str) -> bool:
        """
        Check if a file exists in GCS.
        
        Args:
            gcs_path: GCS path to check
            
        Returns:
            True if file exists
        """
        try:
            blob = self.bucket.blob(gcs_path)
            return blob.exists()
        except:
            return False
    
    def delete_file(self, gcs_path: str) -> bool:
        """
        Delete a file from GCS.
        
        Args:
            gcs_path: GCS path to delete
            
        Returns:
            True if successful
        """
        try:
            blob = self.bucket.blob(gcs_path)
            blob.delete()
            return True
        except Exception as e:
            logger.error(f"Failed to delete {gcs_path}: {e}")
            return False
    
    def get_file_size(self, gcs_path: str) -> Optional[int]:
        """
        Get size of file in GCS.
        
        Args:
            gcs_path: GCS path
            
        Returns:
            File size in bytes, or None if not found
        """
        try:
            blob = self.bucket.blob(gcs_path)
            blob.reload()
            return blob.size
        except:
            return None


def test_connection(bucket_name="audio-data-gemini"):
    """Test GCS connection with current credentials."""
    try:
        client = GCSClient(bucket_name=bucket_name)
        files = client.list_files(prefix="", suffix="")
        print(f"✅ Connection successful! Found {len(files)} files in bucket.")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


if __name__ == "__main__":
    import sys
    bucket = sys.argv[1] if len(sys.argv) > 1 else "audio-data-gemini"
    test_connection(bucket)


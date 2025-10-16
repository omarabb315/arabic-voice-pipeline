"""
S3 utilities for custom S3-compatible endpoint.

Supports Al Jazeera's custom S3 endpoint with proper authentication
and progress tracking for large file transfers.
"""

import os
import boto3
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S3Client:
    """S3 client wrapper for custom endpoint operations."""
    
    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
        verify_ssl: bool = False,  # Set to False for self-signed certs
    ):
        """
        Initialize S3 client with custom endpoint.
        
        Args:
            endpoint_url: S3 endpoint URL (reads from S3_ENDPOINT_URL env var if None)
            aws_access_key_id: Access key (reads from S3_ACCESS_KEY env var if None)
            aws_secret_access_key: Secret key (reads from S3_SECRET_KEY env var if None)
            bucket_name: Bucket name (reads from S3_BUCKET_NAME env var if None)
            verify_ssl: Whether to verify SSL certificates
        """
        self.endpoint_url = endpoint_url or os.environ.get("S3_ENDPOINT_URL")
        self.access_key = aws_access_key_id or os.environ.get("S3_ACCESS_KEY")
        self.secret_key = aws_secret_access_key or os.environ.get("S3_SECRET_KEY")
        self.bucket_name = bucket_name or os.environ.get("S3_BUCKET_NAME")
        
        if not all([self.endpoint_url, self.access_key, self.secret_key, self.bucket_name]):
            raise ValueError(
                "Missing S3 credentials. Set environment variables: "
                "S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET_NAME"
            )
        
        # Create boto3 client
        self.client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            verify=verify_ssl,
        )
        
        logger.info(f"S3 client initialized for bucket: {self.bucket_name}")
        logger.info(f"Endpoint: {self.endpoint_url}")
    
    def upload_file(
        self,
        local_path: str,
        s3_key: str,
        show_progress: bool = True,
    ) -> bool:
        """
        Upload a file to S3 with progress bar.
        
        Args:
            local_path: Path to local file
            s3_key: S3 key (path in bucket)
            show_progress: Show progress bar
            
        Returns:
            True if successful
        """
        try:
            file_size = os.path.getsize(local_path)
            
            if show_progress:
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Uploading {os.path.basename(local_path)}") as pbar:
                    def callback(bytes_transferred):
                        pbar.update(bytes_transferred)
                    
                    self.client.upload_file(
                        local_path,
                        self.bucket_name,
                        s3_key,
                        Callback=callback
                    )
            else:
                self.client.upload_file(local_path, self.bucket_name, s3_key)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            return False
    
    def download_file(
        self,
        s3_key: str,
        local_path: str,
        show_progress: bool = True,
    ) -> bool:
        """
        Download a file from S3 with progress bar.
        
        Args:
            s3_key: S3 key (path in bucket)
            local_path: Path to save file locally
            show_progress: Show progress bar
            
        Returns:
            True if successful
        """
        try:
            # Create parent directory if needed
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Get file size
            response = self.client.head_object(Bucket=self.bucket_name, Key=s3_key)
            file_size = response['ContentLength']
            
            if show_progress:
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Downloading {os.path.basename(s3_key)}") as pbar:
                    def callback(bytes_transferred):
                        pbar.update(bytes_transferred)
                    
                    self.client.download_file(
                        self.bucket_name,
                        s3_key,
                        local_path,
                        Callback=callback
                    )
            else:
                self.client.download_file(self.bucket_name, s3_key, local_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            return False
    
    def list_files(
        self,
        prefix: str = "",
        suffix: str = "",
    ) -> List[str]:
        """
        List files in S3 bucket with optional prefix and suffix filtering.
        
        Args:
            prefix: Only list files starting with this prefix
            suffix: Only list files ending with this suffix
            
        Returns:
            List of S3 keys
        """
        try:
            files = []
            paginator = self.client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if suffix and not key.endswith(suffix):
                            continue
                        files.append(key)
            
            logger.info(f"Found {len(files)} files with prefix '{prefix}' and suffix '{suffix}'")
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    def file_exists(self, s3_key: str) -> bool:
        """
        Check if a file exists in S3.
        
        Args:
            s3_key: S3 key to check
            
        Returns:
            True if file exists
        """
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except:
            return False
    
    def delete_file(self, s3_key: str) -> bool:
        """
        Delete a file from S3.
        
        Args:
            s3_key: S3 key to delete
            
        Returns:
            True if successful
        """
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete {s3_key}: {e}")
            return False
    
    def get_file_size(self, s3_key: str) -> Optional[int]:
        """
        Get size of file in S3.
        
        Args:
            s3_key: S3 key
            
        Returns:
            File size in bytes, or None if not found
        """
        try:
            response = self.client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return response['ContentLength']
        except:
            return None


def test_connection():
    """Test S3 connection with current environment variables."""
    try:
        client = S3Client()
        files = client.list_files(prefix="", suffix="")
        print(f"✅ Connection successful! Found {len(files)} files in bucket.")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


if __name__ == "__main__":
    test_connection()


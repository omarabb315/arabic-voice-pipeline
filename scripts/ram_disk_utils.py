"""
RAM Disk Utilities for H200 Processing

Detects and manages temporary storage locations optimized for high RAM systems.
Supports /dev/shm, custom tmpfs mounts, and regular /tmp with OS caching.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_available_ram_gb() -> float:
    """
    Get available RAM in GB.
    
    Returns:
        Available RAM in GB
    """
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    # Extract KB value and convert to GB
                    kb = int(line.split()[1])
                    gb = kb / (1024 ** 2)
                    return gb
    except Exception as e:
        logger.warning(f"Could not read /proc/meminfo: {e}")
        return 0.0


def get_total_ram_gb() -> float:
    """
    Get total RAM in GB.
    
    Returns:
        Total RAM in GB
    """
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    kb = int(line.split()[1])
                    gb = kb / (1024 ** 2)
                    return gb
    except Exception as e:
        logger.warning(f"Could not read /proc/meminfo: {e}")
        return 0.0


def get_directory_space_gb(path: str) -> Tuple[float, float]:
    """
    Get total and available space in GB for a directory.
    
    Args:
        path: Directory path to check
        
    Returns:
        Tuple of (total_gb, available_gb)
    """
    try:
        stat = shutil.disk_usage(path)
        total_gb = stat.total / (1024 ** 3)
        available_gb = stat.free / (1024 ** 3)
        return total_gb, available_gb
    except Exception as e:
        logger.warning(f"Could not get disk usage for {path}: {e}")
        return 0.0, 0.0


def check_tmpfs_mount(path: str) -> bool:
    """
    Check if a path is mounted as tmpfs (RAM-based filesystem).
    
    Args:
        path: Path to check
        
    Returns:
        True if mounted as tmpfs, False otherwise
    """
    try:
        result = subprocess.run(
            ['df', '-T', path],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                # Check if filesystem type is tmpfs or shm
                fs_type = lines[1].split()[1]
                return fs_type in ['tmpfs', 'shm']
    except Exception as e:
        logger.warning(f"Could not check tmpfs mount for {path}: {e}")
    
    return False


def find_best_temp_location(
    required_space_gb: float = 300.0,
    preferred_paths: list = None
) -> Tuple[str, str]:
    """
    Find the best temporary storage location for processing.
    
    Priority:
    1. Custom tmpfs mounts (e.g., ~/ram_disk)
    2. /dev/shm if large enough (>= required_space_gb)
    3. /tmp with OS caching (will use available RAM automatically)
    
    Args:
        required_space_gb: Minimum required space in GB (default: 300GB)
        preferred_paths: List of preferred paths to check first
        
    Returns:
        Tuple of (path, storage_type) where storage_type is 'tmpfs', 'shm', or 'cached'
    """
    
    logger.info("=" * 70)
    logger.info("Detecting Best Temporary Storage Location")
    logger.info("=" * 70)
    
    # Check system RAM
    total_ram = get_total_ram_gb()
    available_ram = get_available_ram_gb()
    
    logger.info(f"System RAM: {total_ram:.1f} GB total, {available_ram:.1f} GB available")
    
    if available_ram < required_space_gb:
        logger.warning(
            f"⚠️  Available RAM ({available_ram:.1f} GB) is less than "
            f"required space ({required_space_gb:.1f} GB)"
        )
        logger.warning("Processing may be slower or fail due to insufficient memory")
    
    # Default paths to check
    if preferred_paths is None:
        preferred_paths = []
    
    # Add common tmpfs locations
    candidate_paths = preferred_paths + [
        os.path.expanduser("~/ram_disk"),
        "/dev/shm",
        "/tmp"
    ]
    
    logger.info(f"\nChecking candidate locations (need {required_space_gb:.1f} GB)...")
    logger.info("")
    
    # Check each candidate
    for path in candidate_paths:
        if not os.path.exists(path):
            logger.info(f"❌ {path}: Does not exist")
            continue
        
        is_tmpfs = check_tmpfs_mount(path)
        total_space, available_space = get_directory_space_gb(path)
        
        logger.info(f"📁 {path}:")
        logger.info(f"   Type: {'tmpfs (RAM-based)' if is_tmpfs else 'Regular filesystem'}")
        logger.info(f"   Space: {available_space:.1f} GB available / {total_space:.1f} GB total")
        
        # Custom tmpfs mount (highest priority)
        if is_tmpfs and path not in ['/dev/shm', '/tmp'] and available_space >= required_space_gb:
            logger.info(f"   ✅ Selected: Custom tmpfs mount with sufficient space")
            return path, 'tmpfs'
        
        # /dev/shm with sufficient space
        elif path == '/dev/shm' and is_tmpfs and available_space >= required_space_gb:
            logger.info(f"   ✅ Selected: /dev/shm with sufficient space")
            return path, 'shm'
        
        # /tmp as fallback
        elif path == '/tmp' and available_space >= required_space_gb:
            if is_tmpfs:
                logger.info(f"   ✅ Selected: /tmp (tmpfs)")
                return path, 'tmpfs'
            else:
                logger.info(f"   ✅ Selected: /tmp with OS RAM caching")
                logger.info(f"   Note: With {available_ram:.1f} GB available RAM, OS will cache in memory")
                return path, 'cached'
        
        else:
            if available_space < required_space_gb:
                logger.info(f"   ❌ Insufficient space (need {required_space_gb:.1f} GB)")
            else:
                logger.info(f"   ⏭️  Skipped (lower priority)")
    
    # Fallback to /tmp even if space is less than required
    logger.warning(f"\n⚠️  No location found with {required_space_gb:.1f} GB available")
    logger.warning(f"Falling back to /tmp - may need to process in smaller chunks")
    return '/tmp', 'cached'


def setup_temp_directory(
    base_path: str,
    subdir: str = "neutts_processing",
    clean_if_exists: bool = False
) -> str:
    """
    Set up temporary directory for processing.
    
    Args:
        base_path: Base path (e.g., /tmp or /dev/shm)
        subdir: Subdirectory name (default: neutts_processing)
        clean_if_exists: If True, clean directory if it already exists
        
    Returns:
        Full path to temporary directory
    """
    temp_dir = os.path.join(base_path, subdir)
    
    if os.path.exists(temp_dir):
        if clean_if_exists:
            logger.info(f"Cleaning existing directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            os.makedirs(temp_dir, exist_ok=True)
        else:
            logger.info(f"Using existing directory: {temp_dir}")
    else:
        logger.info(f"Creating directory: {temp_dir}")
        os.makedirs(temp_dir, exist_ok=True)
    
    return temp_dir


def validate_ram_for_processing(
    data_size_gb: float,
    overhead_factor: float = 1.5
) -> bool:
    """
    Validate that there's sufficient RAM for processing.
    
    Args:
        data_size_gb: Expected data size in GB
        overhead_factor: Multiplier for overhead (default: 1.5 = 50% overhead)
        
    Returns:
        True if sufficient RAM, False otherwise
    """
    required_ram = data_size_gb * overhead_factor
    available_ram = get_available_ram_gb()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("RAM Validation")
    logger.info("=" * 70)
    logger.info(f"Data size: {data_size_gb:.1f} GB")
    logger.info(f"Estimated requirement: {required_ram:.1f} GB (with {overhead_factor}x overhead)")
    logger.info(f"Available RAM: {available_ram:.1f} GB")
    
    if available_ram >= required_ram:
        logger.info(f"✅ Sufficient RAM available ({available_ram:.1f} >= {required_ram:.1f} GB)")
        return True
    else:
        logger.warning(f"⚠️  Insufficient RAM ({available_ram:.1f} < {required_ram:.1f} GB)")
        logger.warning("Processing may be slower or fail")
        return False


def get_optimal_temp_config(
    required_space_gb: float = 300.0,
    data_size_gb: float = 221.0
) -> dict:
    """
    Get optimal temporary storage configuration.
    
    Args:
        required_space_gb: Required storage space in GB
        data_size_gb: Expected data size in GB
        
    Returns:
        Dictionary with configuration:
        {
            'temp_dir': str,
            'storage_type': str,
            'available_space_gb': float,
            'available_ram_gb': float,
            'ram_sufficient': bool
        }
    """
    # Find best location
    base_path, storage_type = find_best_temp_location(required_space_gb)
    
    # Validate RAM
    ram_sufficient = validate_ram_for_processing(data_size_gb)
    
    # Get final stats
    _, available_space = get_directory_space_gb(base_path)
    available_ram = get_available_ram_gb()
    
    temp_dir = os.path.join(base_path, "neutts_processing")
    
    config = {
        'temp_dir': temp_dir,
        'base_path': base_path,
        'storage_type': storage_type,
        'available_space_gb': available_space,
        'available_ram_gb': available_ram,
        'ram_sufficient': ram_sufficient,
        'is_tmpfs': storage_type in ['tmpfs', 'shm']
    }
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("Final Configuration")
    logger.info("=" * 70)
    logger.info(f"Temp directory: {config['temp_dir']}")
    logger.info(f"Storage type: {config['storage_type']}")
    logger.info(f"Available space: {config['available_space_gb']:.1f} GB")
    logger.info(f"Available RAM: {config['available_ram_gb']:.1f} GB")
    logger.info(f"RAM sufficient: {'✅ Yes' if config['ram_sufficient'] else '⚠️  No'}")
    logger.info("=" * 70)
    logger.info("")
    
    return config


if __name__ == "__main__":
    # Test the utilities
    print("Testing RAM Disk Utilities")
    print()
    
    config = get_optimal_temp_config(
        required_space_gb=300.0,
        data_size_gb=221.0
    )
    
    print("\nConfiguration Summary:")
    for key, value in config.items():
        print(f"  {key}: {value}")


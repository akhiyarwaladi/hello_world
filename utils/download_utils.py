#!/usr/bin/env python3
"""
Download utility functions for malaria detection datasets
Author: Malaria Detection Team
Date: 2024
"""

import os
import requests
import hashlib
import zipfile
import tarfile
import gdown
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm
import time


def download_with_progress(url: str, 
                         destination: Path, 
                         chunk_size: int = 8192,
                         timeout: int = 30,
                         max_retries: int = 3) -> bool:
    """
    Download file from URL with progress bar and retry logic
    
    Args:
        url: URL to download from
        destination: Path to save file
        chunk_size: Size of chunks to download
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        
    Returns:
        bool: True if download successful, False otherwise
    """
    
    for attempt in range(max_retries):
        try:
            print(f"Downloading {url} (attempt {attempt + 1}/{max_retries})")
            
            # Create session with headers
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            response = session.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            with open(destination, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, 
                         desc=destination.name) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"✓ Downloaded {destination.name}")
            return True
            
        except (requests.RequestException, IOError) as e:
            print(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {url} after {max_retries} attempts")
                return False
    
    return False


def download_google_drive(file_id: str, 
                         destination: Path,
                         verify_size: bool = True) -> bool:
    """
    Download file from Google Drive
    
    Args:
        file_id: Google Drive file ID
        destination: Path to save file
        verify_size: Whether to verify file size after download
        
    Returns:
        bool: True if download successful, False otherwise
    """
    
    try:
        print(f"Downloading from Google Drive: {file_id}")
        
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Use gdown for Google Drive downloads
        gdown.download(id=file_id, output=str(destination), quiet=False)
        
        if destination.exists():
            print(f"✓ Downloaded {destination.name}")
            return True
        else:
            print(f"✗ Failed to download {destination.name}")
            return False
            
    except Exception as e:
        print(f"Google Drive download failed: {e}")
        return False


def verify_file_integrity(file_path: Path, 
                         expected_md5: Optional[str] = None,
                         expected_size: Optional[int] = None) -> bool:
    """
    Verify downloaded file integrity
    
    Args:
        file_path: Path to file to verify
        expected_md5: Expected MD5 hash
        expected_size: Expected file size in bytes
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False
    
    # Check file size
    actual_size = file_path.stat().st_size
    if expected_size and actual_size != expected_size:
        print(f"Size mismatch: expected {expected_size}, got {actual_size}")
        return False
    
    # Check MD5 hash
    if expected_md5:
        print("Verifying file integrity...")
        
        md5_hash = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        
        actual_md5 = md5_hash.hexdigest()
        if actual_md5 != expected_md5:
            print(f"MD5 mismatch: expected {expected_md5}, got {actual_md5}")
            return False
        
        print("✓ File integrity verified")
    
    return True


def extract_archive(archive_path: Path, 
                   extract_to: Path,
                   remove_after_extract: bool = False) -> bool:
    """
    Extract various archive formats
    
    Args:
        archive_path: Path to archive file
        extract_to: Directory to extract to
        remove_after_extract: Whether to delete archive after extraction
        
    Returns:
        bool: True if extraction successful, False otherwise
    """
    
    try:
        print(f"Extracting {archive_path.name}...")
        
        extract_to.mkdir(parents=True, exist_ok=True)
        
        if archive_path.suffix.lower() == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
                
        elif archive_path.suffix.lower() in ['.tar', '.gz', '.tgz', '.bz2']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
                
        else:
            print(f"Unsupported archive format: {archive_path.suffix}")
            return False
        
        print(f"✓ Extracted to {extract_to}")
        
        if remove_after_extract and archive_path.exists():
            archive_path.unlink()
            print(f"✓ Removed archive {archive_path.name}")
        
        return True
        
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False


def get_file_info(file_path: Path) -> Dict[str, Any]:
    """
    Get information about a file
    
    Args:
        file_path: Path to file
        
    Returns:
        Dict containing file information
    """
    
    if not file_path.exists():
        return {}
    
    stat = file_path.stat()
    
    # Calculate MD5
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    
    return {
        'name': file_path.name,
        'size': stat.st_size,
        'size_mb': round(stat.st_size / (1024 * 1024), 2),
        'md5': md5_hash.hexdigest(),
        'modified': stat.st_mtime,
        'exists': True
    }


def create_download_manifest(download_dir: Path, 
                           manifest_path: Optional[Path] = None) -> Path:
    """
    Create manifest file of downloaded files
    
    Args:
        download_dir: Directory containing downloaded files
        manifest_path: Path to save manifest (optional)
        
    Returns:
        Path to created manifest file
    """
    
    if manifest_path is None:
        manifest_path = download_dir / "download_manifest.json"
    
    import json
    
    manifest = {
        'created': time.time(),
        'directory': str(download_dir),
        'files': {}
    }
    
    # Scan directory for files
    for file_path in download_dir.rglob('*'):
        if file_path.is_file():
            rel_path = file_path.relative_to(download_dir)
            manifest['files'][str(rel_path)] = get_file_info(file_path)
    
    # Save manifest
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Download manifest saved to {manifest_path}")
    return manifest_path


def check_disk_space(path: Path, required_bytes: int) -> bool:
    """
    Check if there's enough disk space for download
    
    Args:
        path: Path to check
        required_bytes: Required space in bytes
        
    Returns:
        bool: True if enough space available
    """
    
    import shutil
    
    try:
        free_bytes = shutil.disk_usage(path).free
        required_gb = required_bytes / (1024**3)
        free_gb = free_bytes / (1024**3)
        
        print(f"Required space: {required_gb:.2f} GB")
        print(f"Available space: {free_gb:.2f} GB")
        
        if free_bytes < required_bytes:
            print("[WARNING] Insufficient disk space!")
            return False
        
        return True
        
    except Exception as e:
        print(f"Could not check disk space: {e}")
        return True  # Assume OK if can't check


def cleanup_failed_downloads(download_dir: Path, 
                           min_size_bytes: int = 1024) -> int:
    """
    Clean up failed/partial downloads
    
    Args:
        download_dir: Directory to clean
        min_size_bytes: Minimum file size to keep
        
    Returns:
        int: Number of files removed
    """
    
    removed_count = 0
    
    for file_path in download_dir.rglob('*'):
        if file_path.is_file():
            if file_path.stat().st_size < min_size_bytes:
                print(f"Removing small/empty file: {file_path.name}")
                file_path.unlink()
                removed_count += 1
    
    print(f"✓ Cleaned up {removed_count} failed downloads")
    return removed_count
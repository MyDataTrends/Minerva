"""
Model Downloader - Download open source LLM models.

Provides functionality to download GGUF models from HuggingFace
and other sources with progress tracking.
"""
import os
import logging
from pathlib import Path
from typing import Optional, Callable
import urllib.request
import hashlib

from llm_manager.providers.base import DOWNLOADABLE_MODELS, ModelInfo, ProviderType

logger = logging.getLogger(__name__)

# Default download directory
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class DownloadProgress:
    """Track download progress."""
    
    def __init__(self, callback: Optional[Callable[[float, str], None]] = None):
        self.callback = callback
        self.total_size = 0
        self.downloaded = 0
    
    def __call__(self, block_num: int, block_size: int, total_size: int):
        if total_size > 0:
            self.total_size = total_size
            self.downloaded = min(block_num * block_size, total_size)
            percent = (self.downloaded / total_size) * 100
            
            if self.callback:
                size_mb = self.downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                self.callback(percent, f"{size_mb:.1f} / {total_mb:.1f} MB")


def get_available_downloads() -> dict:
    """Get list of models available for download."""
    return DOWNLOADABLE_MODELS


def download_model(
    model_id: str,
    destination_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Optional[Path]:
    """
    Download a model from the available downloads list.
    
    Args:
        model_id: ID from DOWNLOADABLE_MODELS
        destination_dir: Directory to save the model
        progress_callback: Function called with (percent, status_string)
        
    Returns:
        Path to downloaded file, or None if failed
    """
    if model_id not in DOWNLOADABLE_MODELS:
        logger.error(f"Unknown model ID: {model_id}")
        return None
    
    model_info = DOWNLOADABLE_MODELS[model_id]
    url = model_info["url"]
    filename = model_info["filename"]
    
    # Setup destination
    dest_dir = Path(destination_dir or DEFAULT_MODELS_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    
    # Check if already downloaded
    if dest_path.exists():
        expected_size = model_info.get("size_gb", 0) * 1024 * 1024 * 1024
        actual_size = dest_path.stat().st_size
        
        # If sizes are close (within 5%), assume it's complete
        if expected_size > 0 and abs(actual_size - expected_size) / expected_size < 0.05:
            logger.info(f"Model already downloaded: {dest_path}")
            if progress_callback:
                progress_callback(100, "Already downloaded")
            return dest_path
    
    logger.info(f"Downloading {model_info['name']} from {url}")
    
    if progress_callback:
        progress_callback(0, "Starting download...")
    
    try:
        progress = DownloadProgress(progress_callback)
        urllib.request.urlretrieve(url, dest_path, reporthook=progress)
        
        logger.info(f"Download complete: {dest_path}")
        if progress_callback:
            progress_callback(100, "Download complete!")
        
        return dest_path
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        
        # Clean up partial download
        if dest_path.exists():
            try:
                dest_path.unlink()
            except Exception:
                pass
        
        if progress_callback:
            progress_callback(0, f"Failed: {e}")
        
        return None


def create_model_info_for_download(model_id: str, file_path: Path) -> ModelInfo:
    """
    Create a ModelInfo object for a downloaded model.
    
    Args:
        model_id: ID from DOWNLOADABLE_MODELS
        file_path: Path to the downloaded file
        
    Returns:
        ModelInfo object for the downloaded model
    """
    download_info = DOWNLOADABLE_MODELS.get(model_id, {})
    
    return ModelInfo(
        id=f"local:{file_path.stem}",
        name=download_info.get("name", file_path.stem),
        provider_type=ProviderType.LOCAL,
        path_or_endpoint=str(file_path),
        description=download_info.get("description", "Downloaded model"),
        size_gb=download_info.get("size_gb"),
        context_length=4096,
        capabilities=["chat", "completion"],
    )


def get_models_directory() -> Path:
    """Get the models directory, creating if needed."""
    DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_MODELS_DIR


def list_downloaded_models() -> list:
    """List models that have been downloaded to the models directory."""
    models_dir = get_models_directory()
    models = []
    
    for path in models_dir.glob("*.gguf"):
        if path.is_file():
            size_gb = path.stat().st_size / (1024 ** 3)
            models.append({
                "filename": path.name,
                "path": str(path),
                "size_gb": round(size_gb, 1),
            })
    
    return models

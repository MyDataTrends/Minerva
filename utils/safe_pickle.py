"""
Safe Pickle Module for Assay.

Provides secure pickle operations with:
- Path validation (no traversal attacks)
- Checksum generation and verification
- Safe load/dump wrappers

Usage:
    from utils.safe_pickle import safe_dump, safe_load, verify_checksum
    
    # Save with checksum
    safe_dump(model, "model.pkl")
    
    # Load with verification
    model = safe_load("model.pkl")
"""
import hashlib
import logging
import pickle
import json
from pathlib import Path
from typing import Any, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Allowed base directories for pickle files
ALLOWED_DIRS = {
    "models",
    "local_data",
    "artifacts",
    ".assay",
}

# Maximum file size for pickle loads (100MB)
MAX_PICKLE_SIZE = 100 * 1024 * 1024

# Checksum file suffix
CHECKSUM_SUFFIX = ".sha256"


# =============================================================================
# Path Validation
# =============================================================================

def validate_pickle_path(path: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    """
    Validate that a pickle path is safe.
    
    Args:
        path: The path to validate
        base_dir: Optional base directory to resolve relative paths
        
    Returns:
        Resolved, validated Path
        
    Raises:
        ValueError: If path is unsafe (traversal, outside allowed dirs)
    """
    path = Path(path)
    
    # Resolve to absolute path
    if base_dir:
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    
    # Check for path traversal attempts
    path_str = str(path)
    if ".." in path_str:
        raise ValueError(f"Path traversal detected: {path}")
    
    # Check extension
    if path.suffix not in {".pkl", ".pickle", ".joblib"}:
        raise ValueError(f"Invalid pickle extension: {path.suffix}")
    
    # Check that path is within an allowed directory
    path_parts = set(path.parts)
    cwd_parts = set(Path.cwd().parts)
    
    # Allow if within current working directory
    if cwd_parts.issubset(path.parts):
        relative = path.relative_to(Path.cwd())
        if relative.parts and relative.parts[0] in ALLOWED_DIRS:
            return path
    
    # Allow if within home directory .assay
    home_assay = Path.home() / ".assay"
    try:
        path.relative_to(home_assay)
        return path
    except ValueError:
        pass
    
    # If running in development, be more permissive
    if any(allowed in path_parts for allowed in ALLOWED_DIRS):
        logger.debug(f"Allowing path in development mode: {path}")
        return path
    
    raise ValueError(f"Path outside allowed directories: {path}")


def validate_file_size(path: Path) -> None:
    """Check that file size is within limits."""
    if path.exists():
        size = path.stat().st_size
        if size > MAX_PICKLE_SIZE:
            raise ValueError(f"Pickle file too large: {size} bytes (max: {MAX_PICKLE_SIZE})")


# =============================================================================
# Checksum Operations
# =============================================================================

def compute_checksum(data: bytes) -> str:
    """Compute SHA256 checksum of data."""
    return hashlib.sha256(data).hexdigest()


def compute_file_checksum(path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def save_checksum(path: Path, checksum: str, metadata: Optional[dict] = None) -> None:
    """Save checksum to companion file."""
    checksum_path = path.with_suffix(path.suffix + CHECKSUM_SUFFIX)
    
    data = {
        "checksum": checksum,
        "algorithm": "sha256",
        "created_at": datetime.now().isoformat(),
        "file_size": path.stat().st_size if path.exists() else 0,
    }
    
    if metadata:
        data["metadata"] = metadata
    
    with open(checksum_path, "w") as f:
        json.dump(data, f, indent=2)
    
    logger.debug(f"Saved checksum for {path}")


def load_checksum(path: Path) -> Optional[str]:
    """Load checksum from companion file."""
    checksum_path = path.with_suffix(path.suffix + CHECKSUM_SUFFIX)
    
    if not checksum_path.exists():
        return None
    
    try:
        with open(checksum_path, "r") as f:
            data = json.load(f)
        return data.get("checksum")
    except Exception as e:
        logger.warning(f"Failed to load checksum: {e}")
        return None


def verify_checksum(path: Path) -> bool:
    """
    Verify file matches stored checksum.
    
    Returns:
        True if valid, False if mismatch or no checksum found
    """
    path = Path(path)
    
    if not path.exists():
        logger.warning(f"File does not exist: {path}")
        return False
    
    stored = load_checksum(path)
    if not stored:
        logger.warning(f"No checksum file for: {path}")
        return False
    
    computed = compute_file_checksum(path)
    
    if computed != stored:
        logger.error(f"Checksum mismatch for {path}: expected {stored[:16]}..., got {computed[:16]}...")
        return False
    
    logger.debug(f"Checksum verified for {path}")
    return True


# =============================================================================
# Safe Pickle Operations
# =============================================================================

def safe_dump(
    obj: Any,
    path: Union[str, Path],
    base_dir: Optional[Path] = None,
    add_checksum: bool = True,
    metadata: Optional[dict] = None,
) -> Path:
    """
    Safely dump an object to a pickle file with checksum.
    
    Args:
        obj: Object to pickle
        path: Destination path
        base_dir: Optional base directory
        add_checksum: Whether to create checksum file
        metadata: Optional metadata to store with checksum
        
    Returns:
        The resolved path where file was saved
    """
    path = Path(path)
    
    # Validate path
    if base_dir:
        full_path = validate_pickle_path(path, base_dir)
    else:
        # Allow saving to current directory or subdirectories
        full_path = Path.cwd() / path if not path.is_absolute() else path
        full_path = full_path.resolve()
    
    # Ensure directory exists
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Serialize to bytes first (for checksum)
    data = pickle.dumps(obj)
    
    # Check size
    if len(data) > MAX_PICKLE_SIZE:
        raise ValueError(f"Serialized object too large: {len(data)} bytes")
    
    # Write file
    with open(full_path, "wb") as f:
        f.write(data)
    
    # Add checksum
    if add_checksum:
        checksum = compute_checksum(data)
        meta = metadata or {}
        meta["object_type"] = type(obj).__name__
        save_checksum(full_path, checksum, meta)
    
    logger.info(f"Saved pickle: {full_path}")
    return full_path


def safe_load(
    path: Union[str, Path],
    base_dir: Optional[Path] = None,
    verify: bool = True,
    allow_missing_checksum: bool = True,
) -> Any:
    """
    Safely load a pickle file with validation.
    
    Args:
        path: Path to pickle file
        base_dir: Optional base directory
        verify: Whether to verify checksum
        allow_missing_checksum: Allow loading if no checksum file exists
        
    Returns:
        Unpickled object
        
    Raises:
        ValueError: If path is invalid or checksum fails
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)
    
    # Validate path
    if base_dir:
        full_path = validate_pickle_path(path, base_dir)
    else:
        full_path = validate_pickle_path(path)
    
    if not full_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {full_path}")
    
    # Validate file size
    validate_file_size(full_path)
    
    # Verify checksum
    if verify:
        stored = load_checksum(full_path)
        if stored:
            if not verify_checksum(full_path):
                raise ValueError(f"Checksum verification failed: {full_path}")
        elif not allow_missing_checksum:
            raise ValueError(f"No checksum file found: {full_path}")
        else:
            logger.warning(f"Loading pickle without checksum verification: {full_path}")
    
    # Load
    with open(full_path, "rb") as f:
        obj = pickle.load(f)
    
    logger.debug(f"Loaded pickle: {full_path}")
    return obj


# =============================================================================
# Migration Helper
# =============================================================================

def add_checksum_to_existing(path: Union[str, Path]) -> bool:
    """
    Add checksum to an existing pickle file.
    
    Use this to migrate existing pickle files to have checksums.
    
    Returns:
        True if checksum was added, False on error
    """
    try:
        path = Path(path).resolve()
        
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return False
        
        checksum = compute_file_checksum(path)
        save_checksum(path, checksum, {"migrated": True})
        
        logger.info(f"Added checksum to existing file: {path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to add checksum: {e}")
        return False

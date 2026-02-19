"""
Local Model Scanner - Discover GGUF models on the filesystem.

Scans common locations and user-specified directories for
compatible model files (.gguf, .bin).
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set
import re

from llm_manager.providers.base import ModelInfo, ProviderType

logger = logging.getLogger(__name__)


# Common locations where models might be stored
DEFAULT_SCAN_PATHS = [
    # Project-relative
    Path(__file__).resolve().parents[1] / "models",
    Path(__file__).resolve().parents[1] / "adm" / "llm_backends" / "local_model",
    
    # User home directories
    Path.home() / ".cache" / "huggingface" / "hub",
    Path.home() / ".ollama" / "models",
    Path.home() / "models",
    Path.home() / ".local" / "share" / "models",
    
    # LM Studio default locations
    Path.home() / ".cache" / "lm-studio" / "models",
    
    # GPT4All default
    Path.home() / ".cache" / "gpt4all",
    Path.home() / "AppData" / "Local" / "nomic.ai" / "GPT4All",
    
    # Windows common locations
    Path("C:/models"),
    Path("D:/models"),
]

# Patterns to identify model types from filenames
MODEL_PATTERNS = {
    r"mistral.*7b": ("Mistral 7B", "Mistral 7B Instruct model"),
    r"llama[-_]?3.*8b": ("Llama 3 8B", "Meta Llama 3 8B model"),
    r"llama[-_]?2.*7b": ("Llama 2 7B", "Meta Llama 2 7B model"),
    r"llama[-_]?2.*13b": ("Llama 2 13B", "Meta Llama 2 13B model"),
    r"deepseek.*coder": ("DeepSeek Coder", "DeepSeek Coder model"),
    r"phi[-_]?3": ("Phi-3", "Microsoft Phi-3 model"),
    r"phi[-_]?2": ("Phi-2", "Microsoft Phi-2 model"),
    r"gemma.*2b": ("Gemma 2B", "Google Gemma 2B model"),
    r"gemma.*7b": ("Gemma 7B", "Google Gemma 7B model"),
    r"qwen": ("Qwen", "Alibaba Qwen model"),
    r"codellama": ("Code Llama", "Meta Code Llama model"),
    r"starcoder": ("StarCoder", "BigCode StarCoder model"),
    r"wizardcoder": ("WizardCoder", "WizardCoder model"),
    r"openchat": ("OpenChat", "OpenChat model"),
    r"neural[-_]?chat": ("Neural Chat", "Intel Neural Chat model"),
    r"zephyr": ("Zephyr", "HuggingFace Zephyr model"),
    r"orca": ("Orca", "Microsoft Orca model"),
    r"vicuna": ("Vicuna", "LMSYS Vicuna model"),
}


def _identify_model(filename: str) -> tuple:
    """
    Identify model type from filename.
    
    Returns (name, description) or (None, None) if unknown.
    """
    filename_lower = filename.lower()
    
    for pattern, (name, desc) in MODEL_PATTERNS.items():
        if re.search(pattern, filename_lower):
            return name, desc
    
    return None, None


def _get_file_size_gb(path: Path) -> float:
    """Get file size in GB."""
    try:
        return path.stat().st_size / (1024 ** 3)
    except Exception:
        return 0.0


def scan_directory(
    directory: Path,
    extensions: Set[str] = {".gguf", ".bin"},
    recursive: bool = True,
) -> List[ModelInfo]:
    """
    Scan a directory for model files.
    
    Args:
        directory: Path to scan
        extensions: File extensions to look for
        recursive: Whether to scan subdirectories
        
    Returns:
        List of discovered ModelInfo objects
    """
    models = []
    
    if not directory.exists():
        return models
    
    pattern = "**/*" if recursive else "*"
    
    for ext in extensions:
        for filepath in directory.glob(f"{pattern}{ext}"):
            if not filepath.is_file():
                continue
            
            # Get file info
            size_gb = _get_file_size_gb(filepath)
            
            # Skip very small files (likely not models)
            if size_gb < 0.1:
                continue
            
            # Identify model type
            name, desc = _identify_model(filepath.name)
            
            if name is None:
                # Use filename as fallback
                name = filepath.stem.replace("-", " ").replace("_", " ").title()
                desc = f"Local model: {filepath.name}"
            
            # Create unique ID from path
            model_id = f"local:{filepath.stem}"
            
            models.append(ModelInfo(
                id=model_id,
                name=name,
                provider_type=ProviderType.LOCAL,
                path_or_endpoint=str(filepath),
                description=desc,
                size_gb=round(size_gb, 1),
                context_length=4096,  # Default, can be updated
                capabilities=["chat", "completion"],
            ))
    
    return models


def scan_for_local_models(
    additional_paths: Optional[List[Path]] = None,
    include_default_paths: bool = True,
) -> List[ModelInfo]:
    """
    Scan for local LLM models in common locations.
    
    Args:
        additional_paths: Extra directories to scan
        include_default_paths: Whether to scan default locations
        
    Returns:
        List of discovered ModelInfo objects
    """
    all_models = []
    scanned_paths = set()
    
    # Build list of paths to scan
    paths_to_scan = []
    
    if include_default_paths:
        paths_to_scan.extend(DEFAULT_SCAN_PATHS)
    
    if additional_paths:
        paths_to_scan.extend(additional_paths)
    
    # Scan each path
    for path in paths_to_scan:
        path = Path(path)
        
        # Skip if already scanned or doesn't exist
        if str(path) in scanned_paths or not path.exists():
            continue
        
        scanned_paths.add(str(path))
        
        logger.debug(f"Scanning for models in: {path}")
        models = scan_directory(path)
        all_models.extend(models)
    
    # Also check environment variable for custom path
    custom_path = os.getenv("ASSAY_MODELS_PATH")
    if custom_path:
        custom_models = scan_directory(Path(custom_path))
        all_models.extend(custom_models)
    
    # Remove duplicates by path
    seen_paths = set()
    unique_models = []
    for model in all_models:
        if model.path_or_endpoint not in seen_paths:
            seen_paths.add(model.path_or_endpoint)
            unique_models.append(model)
    
    logger.info(f"Found {len(unique_models)} local model(s)")
    return unique_models


def check_ollama_models() -> List[ModelInfo]:
    """
    Check for models available via Ollama if installed.
    
    Returns list of models available through Ollama.
    """
    import subprocess
    
    models = []
    
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            
            for line in lines:
                parts = line.split()
                if len(parts) >= 1:
                    name = parts[0]
                    size = parts[1] if len(parts) > 1 else ""
                    
                    models.append(ModelInfo(
                        id=f"ollama:{name}",
                        name=f"Ollama: {name}",
                        provider_type=ProviderType.OLLAMA,
                        path_or_endpoint=f"http://localhost:11434",
                        description=f"Ollama model: {name}",
                        size_gb=float(size.replace("GB", "")) if "GB" in size else None,
                        capabilities=["chat", "completion"],
                    ))
                    
    except FileNotFoundError:
        logger.debug("Ollama not installed")
    except subprocess.TimeoutExpired:
        logger.debug("Ollama command timed out")
    except Exception as e:
        logger.debug(f"Error checking Ollama: {e}")
    
    return models

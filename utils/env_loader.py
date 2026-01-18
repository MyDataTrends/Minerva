"""
Environment loader utility.

Loads environment variables from .env file at project root.
"""
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_env(env_path: Path = None) -> bool:
    """
    Load environment variables from .env file.
    
    Args:
        env_path: Path to .env file. Defaults to project root.
        
    Returns:
        True if .env was loaded, False otherwise.
    """
    if env_path is None:
        # Find project root (where .env should be)
        env_path = Path(__file__).resolve().parents[1] / ".env"
    
    if not env_path.exists():
        logger.debug(f".env file not found at {env_path}")
        return False
    
    try:
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                
                # Parse KEY=VALUE
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    # Only set if not already in environment
                    if key and key not in os.environ:
                        os.environ[key] = value
                        logger.debug(f"Loaded env var: {key}")
        
        logger.info(f"Loaded environment from {env_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load .env: {e}")
        return False


# Auto-load on import
_loaded = load_env()

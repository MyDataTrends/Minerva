"""
Secure Key Storage - Encrypted storage for API keys.

Uses Fernet (symmetric encryption) to store keys locally.
If cryptography is not available, falls back to a hidden JSON (with warning).
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Try to import cryptography
try:
    from cryptography.fernet import Fernet
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    logger.warning("Cryptography module not found. Falling back to plain text storage.")

class KeyStorage:
    def __init__(self, app_name: str = "minerva"):
        self.app_dir = Path.home() / f".{app_name}"
        self.app_dir.mkdir(exist_ok=True)
        
        self.key_file = self.app_dir / "master.key"
        self.secrets_file = self.app_dir / "secrets.enc"
        self.plain_file = self.app_dir / "secrets.json"
        
        self._fernet = None
        if HAS_CRYPTO:
            self._init_crypto()

    def _init_crypto(self):
        """Initialize encryption key."""
        if self.key_file.exists():
            key = self.key_file.read_bytes()
        else:
            key = Fernet.generate_key()
            # Set restrictive permissions if possible (os.chmod)
            self.key_file.write_bytes(key)
        
        try:
            self._fernet = Fernet(key)
        except Exception as e:
            logger.error(f"Failed to load encryption key: {e}")
            HAS_CRYPTO = False

    def save_key(self, service: str, api_key: str):
        """Save a key for a service."""
        data = self._load_data()
        data[service] = api_key
        self._save_data(data)

    def get_key(self, service: str) -> Optional[str]:
        """Get a key for a service."""
        data = self._load_data()
        return data.get(service)

    def list_services(self) -> list:
        """List services with stored keys."""
        return list(self._load_data().keys())

    def _load_data(self) -> Dict[str, str]:
        """Load and decrypt data."""
        if HAS_CRYPTO and self._fernet:
            if not self.secrets_file.exists():
                return {}
            try:
                encrypted = self.secrets_file.read_bytes()
                decrypted = self._fernet.decrypt(encrypted)
                return json.loads(decrypted)
            except Exception as e:
                logger.error(f"Corrupt secrets file: {e}")
                return {}
        else:
            # Fallback
            if not self.plain_file.exists():
                return {}
            try:
                return json.loads(self.plain_file.read_text())
            except Exception:
                return {}

    def _save_data(self, data: Dict[str, str]):
        """Encrypt and save data."""
        if HAS_CRYPTO and self._fernet:
            json_bytes = json.dumps(data).encode("utf-8")
            encrypted = self._fernet.encrypt(json_bytes)
            self.secrets_file.write_bytes(encrypted)
        else:
            self.plain_file.write_text(json.dumps(data, indent=2))

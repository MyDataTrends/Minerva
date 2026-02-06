"""
Secure Credential Manager - Encrypted API key storage.

Uses Fernet symmetric encryption (AES-128-CBC with HMAC) to store API keys securely.
Keys are derived from a user-provided master password using PBKDF2.

Usage:
    cred_mgr = CredentialManager()
    cred_mgr.store_credential("fred", "your-api-key", master_password="...")
    api_key = cred_mgr.get_credential("fred", master_password="...")
"""
import os
import json
import base64
import hashlib
import secrets
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import cryptography, fall back to basic obfuscation if not available
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography package not installed. Using basic obfuscation (less secure).")


@dataclass
class APICredential:
    """Stored API credential."""
    api_id: str
    encrypted_key: str
    salt: str
    created_at: str
    last_used: Optional[str] = None


class CredentialManager:
    """
    Secure credential storage with encryption.
    
    Credentials are stored in ~/.minerva/credentials.json with:
    - AES-128 encryption (via Fernet) when cryptography is available
    - Base64 obfuscation as fallback (not secure, but better than plaintext)
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize credential manager."""
        if storage_path is None:
            self.storage_path = Path.home() / ".minerva" / "credentials.json"
        else:
            self.storage_path = Path(storage_path)
        
        # Ensure directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing credentials
        self._credentials: Dict[str, Dict[str, Any]] = {}
        self._load_credentials()
    
    def _load_credentials(self) -> None:
        """Load credentials from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    self._credentials = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load credentials: {e}")
                self._credentials = {}
    
    def _save_credentials(self) -> None:
        """Save credentials to storage."""
        try:
            with open(self.storage_path, "w") as f:
                json.dump(self._credentials, f, indent=2)
            # Set restrictive permissions on Unix
            if os.name != 'nt':
                os.chmod(self.storage_path, 0o600)
        except IOError as e:
            logger.error(f"Failed to save credentials: {e}")
            raise
    
    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        if CRYPTO_AVAILABLE:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=480000,  # OWASP recommended minimum
            )
            return base64.urlsafe_b64encode(kdf.derive(password.encode()))
        else:
            # Fallback: simple hash (NOT SECURE, just obfuscation)
            return base64.urlsafe_b64encode(
                hashlib.sha256(password.encode() + salt).digest()
            )
    
    def _encrypt(self, plaintext: str, password: str) -> tuple[str, str]:
        """Encrypt a value with password."""
        salt = secrets.token_bytes(16)
        key = self._derive_key(password, salt)
        
        if CRYPTO_AVAILABLE:
            fernet = Fernet(key)
            encrypted = fernet.encrypt(plaintext.encode())
            return base64.urlsafe_b64encode(encrypted).decode(), base64.urlsafe_b64encode(salt).decode()
        else:
            # Fallback: XOR obfuscation (NOT SECURE)
            key_bytes = key[:len(plaintext.encode())]
            obfuscated = bytes(a ^ b for a, b in zip(plaintext.encode(), key_bytes * (len(plaintext) // len(key_bytes) + 1)))
            return base64.urlsafe_b64encode(obfuscated).decode(), base64.urlsafe_b64encode(salt).decode()
    
    def _decrypt(self, encrypted: str, salt: str, password: str) -> str:
        """Decrypt a value with password."""
        salt_bytes = base64.urlsafe_b64decode(salt.encode())
        key = self._derive_key(password, salt_bytes)
        encrypted_bytes = base64.urlsafe_b64decode(encrypted.encode())
        
        if CRYPTO_AVAILABLE:
            fernet = Fernet(key)
            return fernet.decrypt(encrypted_bytes).decode()
        else:
            # Fallback: XOR deobfuscation
            key_bytes = key[:len(encrypted_bytes)]
            deobfuscated = bytes(a ^ b for a, b in zip(encrypted_bytes, key_bytes * (len(encrypted_bytes) // len(key_bytes) + 1)))
            return deobfuscated.decode()
    
    def store_credential(
        self, 
        api_id: str, 
        api_key: str, 
        master_password: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store an API credential securely.
        
        Args:
            api_id: Identifier for the API (e.g., "fred", "alpha_vantage")
            api_key: The API key to store
            master_password: Password used to encrypt the key
            metadata: Optional metadata (e.g., email, account name)
        """
        from datetime import datetime
        
        encrypted_key, salt = self._encrypt(api_key, master_password)
        
        self._credentials[api_id] = {
            "encrypted_key": encrypted_key,
            "salt": salt,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        
        self._save_credentials()
        logger.info(f"Stored credential for {api_id}")
    
    def get_credential(
        self, 
        api_id: str, 
        master_password: str
    ) -> Optional[str]:
        """
        Retrieve an API credential.
        
        Args:
            api_id: Identifier for the API
            master_password: Password used to decrypt the key
            
        Returns:
            Decrypted API key or None if not found
        """
        if api_id not in self._credentials:
            return None
        
        cred = self._credentials[api_id]
        
        try:
            api_key = self._decrypt(
                cred["encrypted_key"],
                cred["salt"],
                master_password
            )
            
            # Update last_used timestamp
            from datetime import datetime
            cred["last_used"] = datetime.now().isoformat()
            self._save_credentials()
            
            return api_key
        except Exception as e:
            logger.error(f"Failed to decrypt credential for {api_id}: {e}")
            return None
    
    def has_credential(self, api_id: str) -> bool:
        """Check if a credential exists for an API."""
        return api_id in self._credentials
    
    def list_credentials(self) -> list[Dict[str, Any]]:
        """List all stored credentials (without the actual keys)."""
        return [
            {
                "api_id": api_id,
                "created_at": cred.get("created_at"),
                "last_used": cred.get("last_used"),
                "metadata": cred.get("metadata", {}),
            }
            for api_id, cred in self._credentials.items()
        ]
    
    def delete_credential(self, api_id: str) -> bool:
        """Delete a stored credential."""
        if api_id in self._credentials:
            del self._credentials[api_id]
            self._save_credentials()
            logger.info(f"Deleted credential for {api_id}")
            return True
        return False
    
    def check_env_fallback(self, api_id: str, env_var: str) -> Optional[str]:
        """
        Check environment variable as fallback for credential.
        
        This allows users to set API keys via environment variables
        instead of storing them encrypted.
        """
        return os.environ.get(env_var)


# =============================================================================
# Helper Functions
# =============================================================================

def get_or_prompt_credential(
    api_id: str,
    api_name: str,
    env_var: str,
    signup_url: str,
    master_password: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get a credential or return info needed to prompt user.
    
    Returns:
        Dict with either:
        - {"success": True, "api_key": "..."} if credential found
        - {"success": False, "needs_setup": True, "signup_url": "...", "env_var": "..."}
    """
    cred_mgr = CredentialManager()
    
    # 1. Check environment variable first
    env_key = os.environ.get(env_var)
    if env_key:
        return {"success": True, "api_key": env_key, "source": "environment"}
    
    # 2. Check encrypted storage
    if cred_mgr.has_credential(api_id):
        if master_password:
            api_key = cred_mgr.get_credential(api_id, master_password)
            if api_key:
                return {"success": True, "api_key": api_key, "source": "encrypted_storage"}
        
        # Has credential but needs password
        return {
            "success": False,
            "needs_password": True,
            "api_id": api_id,
            "api_name": api_name,
        }
    
    # 3. Need to set up credential
    return {
        "success": False,
        "needs_setup": True,
        "api_id": api_id,
        "api_name": api_name,
        "signup_url": signup_url,
        "env_var": env_var,
        "instructions": f"Get a free API key from {signup_url}",
    }

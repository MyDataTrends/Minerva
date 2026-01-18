"""
Model Registry - Central management for LLM models and providers.

Handles:
- Registering and discovering models
- Managing the active model
- Providing access to model instances
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from llm_manager.providers.base import (
    ModelInfo, 
    ProviderType, 
    LLMProvider,
    KNOWN_MODELS, 
    DOWNLOADABLE_MODELS,
)
from llm_manager.scanner import scan_for_local_models, check_ollama_models

logger = logging.getLogger(__name__)

# Config file location
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
CONFIG_FILE = CONFIG_DIR / "llm_config.json"


class ModelRegistry:
    """
    Central registry for managing LLM models and providers.
    
    Tracks available models (local + cloud), manages active model selection,
    and provides access to provider instances.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._models: Dict[str, ModelInfo] = {}
        self._active_model_id: Optional[str] = None
        self._active_provider: Optional[LLMProvider] = None
        self._initialized = True
        
        # Load saved config
        self._load_config()
        
        # Register known cloud models
        for model_id, model_info in KNOWN_MODELS.items():
            if model_info.provider_type != ProviderType.LOCAL:
                self._models[model_id] = model_info
    
    def _load_config(self) -> None:
        """Load saved configuration."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as f:
                    config = json.load(f)
                    self._active_model_id = config.get("active_model")
            except Exception as e:
                logger.warning(f"Failed to load LLM config: {e}")
    
    def _save_config(self) -> None:
        """Save configuration."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            config = {
                "active_model": self._active_model_id,
            }
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save LLM config: {e}")
    
    def scan_local_models(self, additional_paths: Optional[List[Path]] = None) -> int:
        """
        Scan for local models and add them to registry.
        
        Returns number of models found.
        """
        local_models = scan_for_local_models(additional_paths)
        
        for model in local_models:
            self._models[model.id] = model
        
        # Also check Ollama
        ollama_models = check_ollama_models()
        for model in ollama_models:
            self._models[model.id] = model
        
        return len(local_models) + len(ollama_models)
    
    def register_model(self, model_info: ModelInfo) -> None:
        """Register a model in the registry."""
        self._models[model_info.id] = model_info
        logger.info(f"Registered model: {model_info.id}")
    
    def unregister_model(self, model_id: str) -> bool:
        """Remove a model from the registry."""
        if model_id in self._models:
            del self._models[model_id]
            if self._active_model_id == model_id:
                self._active_model_id = None
                self._active_provider = None
            return True
        return False
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model info by ID."""
        return self._models.get(model_id)
    
    def get_all_models(self) -> List[ModelInfo]:
        """Get all registered models."""
        return list(self._models.values())
    
    def get_local_models(self) -> List[ModelInfo]:
        """Get only local models."""
        return [m for m in self._models.values() if m.provider_type == ProviderType.LOCAL]
    
    def get_cloud_models(self) -> List[ModelInfo]:
        """Get only cloud API models."""
        return [m for m in self._models.values() if m.provider_type in (
            ProviderType.OPENAI, 
            ProviderType.ANTHROPIC,
        )]
    
    def get_ollama_models(self) -> List[ModelInfo]:
        """Get Ollama models."""
        return [m for m in self._models.values() if m.provider_type == ProviderType.OLLAMA]
    
    def set_active_model(self, model_id: str) -> bool:
        """
        Set the active model.
        
        Returns True if successful.
        """
        if model_id not in self._models:
            logger.error(f"Model not found: {model_id}")
            return False
        
        # Unload current provider if different
        if self._active_provider and self._active_model_id != model_id:
            try:
                self._active_provider.unload()
            except Exception:
                pass
            self._active_provider = None
        
        self._active_model_id = model_id
        self._save_config()
        logger.info(f"Active model set to: {model_id}")
        return True
    
    def get_active_model(self) -> Optional[ModelInfo]:
        """Get the currently active model info (doesn't load the model)."""
        if self._active_model_id:
            return self._models.get(self._active_model_id)
        return None
    
    def get_active_provider(self, auto_load: bool = False) -> Optional[LLMProvider]:
        """
        Get the active provider instance.
        
        Args:
            auto_load: If True, attempt to load the model. If False (default),
                       only return the provider if already loaded.
        
        NOTE: auto_load=True can crash the app if the model has issues.
        Use load_active_model() instead for explicit loading with error handling.
        """
        if not self._active_model_id:
            return None
        
        # Return cached provider if available
        if self._active_provider and self._active_provider.is_loaded:
            return self._active_provider
        
        if not auto_load:
            return None
        
        # Auto-load requested - this can crash!
        return self.load_active_model()
    
    def load_active_model(self) -> Optional[LLMProvider]:
        """
        Explicitly load the active model.
        
        This is the ONLY place where model loading should happen.
        Call this when the user explicitly requests model loading.
        
        Returns the provider if successful, None if failed.
        """
        if not self._active_model_id:
            logger.warning("No active model set")
            return None
        
        model_info = self._models.get(self._active_model_id)
        if not model_info:
            logger.warning(f"Model not found: {self._active_model_id}")
            return None
        
        # Unload existing provider first
        if self._active_provider:
            try:
                self._active_provider.unload()
            except Exception:
                pass
            self._active_provider = None
        
        # Create new provider
        provider = self._create_provider(model_info)
        if not provider:
            logger.error(f"Failed to create provider for {model_info.name}")
            return None
        
        # Attempt to load - this is where crashes can happen
        logger.info(f"Loading model: {model_info.name}...")
        try:
            if provider.load():
                self._active_provider = provider
                logger.info(f"Model loaded successfully: {model_info.name}")
                return provider
            else:
                logger.error(f"Failed to load model: {model_info.name}")
                return None
        except Exception as e:
            logger.error(f"Exception loading model {model_info.name}: {e}")
            return None
    
    def _create_provider(self, model_info: ModelInfo) -> Optional[LLMProvider]:
        """Create a provider instance for a model."""
        try:
            if model_info.provider_type == ProviderType.LOCAL:
                from llm_manager.providers.local import LocalProvider
                return LocalProvider(model_info)
            
            elif model_info.provider_type == ProviderType.OPENAI:
                from llm_manager.providers.openai_provider import OpenAIProvider
                return OpenAIProvider(model_info)
            
            elif model_info.provider_type == ProviderType.ANTHROPIC:
                from llm_manager.providers.anthropic_provider import AnthropicProvider
                return AnthropicProvider(model_info)
            
            elif model_info.provider_type == ProviderType.OLLAMA:
                from llm_manager.providers.ollama import OllamaProvider
                return OllamaProvider(model_info)
            
            else:
                logger.warning(f"Unknown provider type: {model_info.provider_type}")
                return None
                
        except ImportError as e:
            logger.error(f"Provider not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to create provider: {e}")
            return None
    
    def get_downloadable_models(self) -> Dict[str, Dict]:
        """Get list of models available for download."""
        return DOWNLOADABLE_MODELS


# Convenience functions
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get the global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def get_available_models() -> List[ModelInfo]:
    """Get all available models."""
    return get_registry().get_all_models()


def get_active_model() -> Optional[ModelInfo]:
    """Get the currently active model."""
    return get_registry().get_active_model()


def set_active_model(model_id: str) -> bool:
    """Set the active model by ID."""
    return get_registry().set_active_model(model_id)


def scan_models(additional_paths: Optional[List[Path]] = None) -> int:
    """Scan for local models."""
    return get_registry().scan_local_models(additional_paths)

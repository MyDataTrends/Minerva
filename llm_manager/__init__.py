"""
LLM Manager - Unified interface for managing LLM providers.

Supports:
- Local models (GGUF format via llama-cpp-python)
- Cloud APIs (OpenAI, Anthropic, etc.)
- Model discovery and download
"""
from llm_manager.registry import (
    ModelRegistry,
    get_available_models,
    get_active_model,
    set_active_model,
)
from llm_manager.providers.base import LLMProvider
from llm_manager.scanner import scan_for_local_models

__all__ = [
    "ModelRegistry",
    "LLMProvider",
    "get_available_models",
    "get_active_model",
    "set_active_model",
    "scan_for_local_models",
]

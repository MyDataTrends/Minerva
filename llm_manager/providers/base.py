"""
Base LLM Provider - Abstract interface for all LLM backends.

Providers handle the actual inference calls to different LLM types:
- Local models (llama-cpp, transformers)
- Cloud APIs (OpenAI, Anthropic, etc.)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Types of LLM providers."""
    LOCAL = "local"           # Local model file (GGUF, etc.)
    OPENAI = "openai"         # OpenAI API
    ANTHROPIC = "anthropic"   # Anthropic Claude API
    OLLAMA = "ollama"         # Ollama local server
    CUSTOM_API = "custom"     # Custom API endpoint


@dataclass
class ModelInfo:
    """Information about an available model."""
    id: str                              # Unique identifier
    name: str                            # Display name
    provider_type: ProviderType          # Type of provider
    path_or_endpoint: str                # File path or API endpoint
    description: str = ""                # Human-readable description
    size_gb: Optional[float] = None      # Model size in GB
    context_length: int = 4096           # Max context window
    capabilities: List[str] = field(default_factory=list)  # e.g., ["chat", "completion"]
    requires_api_key: bool = False       # Whether API key is needed
    api_key_env_var: str = ""            # Environment variable for API key
    api_model_id: Optional[str] = None   # Specific API model ID (e.g. for versioning)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "provider_type": self.provider_type.value,
            "path_or_endpoint": self.path_or_endpoint,
            "description": self.description,
            "size_gb": self.size_gb,
            "context_length": self.context_length,
            "capabilities": self.capabilities,
            "requires_api_key": self.requires_api_key,
        }


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Each provider implementation handles a specific backend:
    - LocalProvider: llama-cpp-python for GGUF models
    - OpenAIProvider: OpenAI API
    - AnthropicProvider: Claude API
    - OllamaProvider: Ollama local server
    """
    
    provider_type: ProviderType = ProviderType.LOCAL
    
    def __init__(self, model_info: ModelInfo, api_key: Optional[str] = None):
        self.model_info = model_info
        self._api_key = api_key
        self._model = None
        self._loaded = False
    
    @property
    def api_key(self) -> Optional[str]:
        """Get API key from init or environment."""
        if self._api_key:
            return self._api_key
        if self.model_info.api_key_env_var:
            import os
            return os.getenv(self.model_info.api_key_env_var)
        return None
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    @abstractmethod
    def load(self) -> bool:
        """
        Load the model or initialize the API connection.
        
        Returns True if successful.
        """
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload the model to free memory."""
        pass
    
    @abstractmethod
    def complete(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate a completion for the given prompt.
        
        Args:
            prompt: Input text to complete
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            
        Returns:
            Generated text
        """
        pass
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate a chat response.
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Assistant's response text
        """
        # Default: convert to prompt-style completion
        prompt = "\n".join(
            f"{m['role'].upper()}: {m['content']}" 
            for m in messages
        )
        prompt += "\nASSISTANT:"
        return self.complete(prompt, max_tokens, temperature, **kwargs)
    
    def test_connection(self) -> bool:
        """Test if the provider is working."""
        try:
            if not self._loaded:
                if not self.load():
                    return False
            result = self.complete("Say 'test' if you can hear me.", max_tokens=10)
            return len(result) > 0
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Known model configurations for easy setup
KNOWN_MODELS = {
    # Local models (user must have files)
    "mistral-7b": ModelInfo(
        id="mistral-7b",
        name="Mistral 7B",
        provider_type=ProviderType.LOCAL,
        path_or_endpoint="",  # Set by scanner
        description="Mistral 7B Instruct - Fast, capable 7B model",
        size_gb=4.1,
        context_length=8192,
        capabilities=["chat", "completion"],
    ),
    "llama-3-8b": ModelInfo(
        id="llama-3-8b",
        name="Llama 3 8B",
        provider_type=ProviderType.LOCAL,
        path_or_endpoint="",
        description="Meta Llama 3 8B - Strong general-purpose model",
        size_gb=4.7,
        context_length=8192,
        capabilities=["chat", "completion"],
    ),
    "deepseek-coder": ModelInfo(
        id="deepseek-coder",
        name="DeepSeek Coder",
        provider_type=ProviderType.LOCAL,
        path_or_endpoint="",
        description="DeepSeek Coder - Specialized for code generation",
        size_gb=3.8,
        context_length=16384,
        capabilities=["chat", "completion", "code"],
    ),
    "phi-3-mini": ModelInfo(
        id="phi-3-mini",
        name="Phi-3 Mini",
        provider_type=ProviderType.LOCAL,
        path_or_endpoint="",
        description="Microsoft Phi-3 Mini - Small but capable",
        size_gb=2.2,
        context_length=4096,
        capabilities=["chat", "completion"],
    ),
    
    # Cloud APIs
    # Cloud APIs
    "gpt-5.3": ModelInfo(
        id="gpt-5.3",
        name="GPT-5.3",
        provider_type=ProviderType.OPENAI,
        path_or_endpoint="https://api.openai.com/v1",
        description="OpenAI GPT-5.3 - 'Garlic' Architecture (SOTA)",
        context_length=128000,
        capabilities=["chat", "completion", "vision", "reasoning"],
        requires_api_key=True,
        api_key_env_var="OPENAI_API_KEY",
        api_model_id="gpt-5.3-preview-2026", # Explicit version
    ),
    "gpt-5.3-codex": ModelInfo(
        id="gpt-5.3-codex",
        name="GPT-5.3 Codex",
        provider_type=ProviderType.OPENAI,
        path_or_endpoint="https://api.openai.com/v1",
        description="OpenAI GPT-5.3 Codex - Agentic Coding Specialist",
        context_length=400000,
        capabilities=["chat", "completion", "code"],
        requires_api_key=True,
        api_key_env_var="OPENAI_API_KEY",
        api_model_id="gpt-5.3-codex-2026-02",
    ),
    "claude-4-5-sonnet": ModelInfo(
        id="claude-4-5-sonnet",
        name="Claude 4.5 Sonnet",
        provider_type=ProviderType.ANTHROPIC,
        path_or_endpoint="https://api.anthropic.com/v1",
        description="Anthropic Claude 4.5 Sonnet - SOTA Intelligence",
        context_length=200000,
        capabilities=["chat", "completion"],
        requires_api_key=True,
        api_key_env_var="ANTHROPIC_API_KEY",
        api_model_id="claude-sonnet-4-5-20250929",
    ),
    "claude-4-6-opus": ModelInfo(
        id="claude-4-6-opus",
        name="Claude 4.6 Opus",
        provider_type=ProviderType.ANTHROPIC,
        path_or_endpoint="https://api.anthropic.com/v1",
        description="Anthropic Claude 4.6 Opus - Maximum Reasoning",
        context_length=500000,
        capabilities=["chat", "completion"],
        requires_api_key=True,
        api_key_env_var="ANTHROPIC_API_KEY",
        api_model_id="claude-opus-4-6", # No date suffix reported for primary alias yet
    ),
    "claude-4-5-haiku": ModelInfo(
        id="claude-4-5-haiku",
        name="Claude 4.5 Haiku",
        provider_type=ProviderType.ANTHROPIC,
        path_or_endpoint="https://api.anthropic.com/v1",
        description="Anthropic Claude 4.5 Haiku - Ultra Fast",
        context_length=200000,
        capabilities=["chat", "completion"],
        requires_api_key=True,
        api_key_env_var="ANTHROPIC_API_KEY",
        api_model_id="claude-haiku-4-5-20251001",
    ),
}


# Downloadable open source models
DOWNLOADABLE_MODELS = {
    "mistral-7b-instruct-q4": {
        "name": "Mistral 7B Instruct (Q4)",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "size_gb": 4.1,
        "description": "Mistral 7B Instruct - Great balance of speed and quality",
    },
    "llama-3-8b-instruct-q4": {
        "name": "Llama 3 8B Instruct (Q4)",
        "url": "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        "filename": "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        "size_gb": 4.7,
        "description": "Meta Llama 3 8B - Strong general-purpose model",
    },
    "phi-3-mini-q4": {
        "name": "Phi-3 Mini (Q4)",
        "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
        "filename": "Phi-3-mini-4k-instruct-q4.gguf",
        "size_gb": 2.2,
        "description": "Microsoft Phi-3 Mini - Small but capable",
    },
    "deepseek-coder-6.7b-q4": {
        "name": "DeepSeek Coder 6.7B (Q4)",
        "url": "https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
        "filename": "deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
        "size_gb": 3.8,
        "description": "DeepSeek Coder - Specialized for code",
    },
}

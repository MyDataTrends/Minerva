"""
Local LLM Provider - Uses subprocess for GGUF models.

Runs llama-cpp-python in a separate process to isolate it from
Streamlit's threading model and prevent C-level crashes.
"""
import logging
from typing import Optional, List, Dict

from llm_manager.providers.base import LLMProvider, ModelInfo, ProviderType

logger = logging.getLogger(__name__)


class LocalProvider(LLMProvider):
    """
    Provider for local GGUF models using subprocess isolation.
    
    The model runs in a separate process via llm_server.py,
    and we communicate via HTTP.
    """
    
    provider_type = ProviderType.LOCAL
    
    def __init__(self, model_info: ModelInfo, **kwargs):
        super().__init__(model_info)
        self._kwargs = kwargs
        self._subprocess = None
    
    def load(self) -> bool:
        """Load the local model via subprocess."""
        if self._loaded:
            return True
        
        try:
            from llm_manager.subprocess_manager import get_llm_subprocess
            
            self._subprocess = get_llm_subprocess()
            model_path = self.model_info.path_or_endpoint
            
            logger.info(f"Loading local model via subprocess: {model_path}")
            
            n_ctx = min(self.model_info.context_length, 2048)
            n_gpu = self._kwargs.get("n_gpu_layers", 0)
            
            if self._subprocess.load_model(model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu):
                self._loaded = True
                logger.info(f"Model loaded successfully: {self.model_info.name}")
                return True
            else:
                logger.error(f"Failed to load model via subprocess")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        if self._subprocess is None:
            return False
        return self._subprocess.is_running()
    
    def unload(self) -> None:
        """Unload the model."""
        if self._subprocess is not None:
            self._subprocess.unload_model()
            self._loaded = False
            logger.info(f"Model unloaded: {self.model_info.name}")
    
    def complete(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate completion using subprocess."""
        if not self._loaded or self._subprocess is None:
            if not self.load():
                return ""
        
        try:
            result = self._subprocess.complete(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=kwargs.get("stop", ["</s>", "\n\n"]),
            )
            return result
            
        except Exception as e:
            logger.error(f"Completion failed: {e}")
            return ""
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate chat response using subprocess."""
        if not self._loaded or self._subprocess is None:
            if not self.load():
                return ""
        
        try:
            result = self._subprocess.chat(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return result
                
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            return ""

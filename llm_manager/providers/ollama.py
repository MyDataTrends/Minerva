"""
Ollama Provider - Uses Ollama local server for models.
"""
import logging
from typing import Optional, List, Dict

import requests

from llm_manager.providers.base import LLMProvider, ModelInfo, ProviderType

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """
    Provider for Ollama local server.
    """
    
    provider_type = ProviderType.OLLAMA
    
    def __init__(self, model_info: ModelInfo, base_url: str = "http://localhost:11434"):
        super().__init__(model_info)
        self.base_url = model_info.path_or_endpoint or base_url
        self._model_name = model_info.id.replace("ollama:", "")
    
    def load(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self._loaded = True
                logger.info(f"Ollama server connected: {self.base_url}")
                return True
            return False
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            return False
    
    def unload(self) -> None:
        """Nothing to unload."""
        self._loaded = False
    
    def complete(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate completion using Ollama."""
        if not self._loaded:
            if not self.load():
                return ""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self._model_name,
                    "prompt": prompt,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                    "stream": False,
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            
            logger.error(f"Ollama API error: {response.status_code}")
            return ""
            
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            return ""
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate chat response using Ollama."""
        if not self._loaded:
            if not self.load():
                return ""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self._model_name,
                    "messages": messages,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                    "stream": False,
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json().get("message", {}).get("content", "").strip()
            
            logger.error(f"Ollama API error: {response.status_code}")
            return ""
            
        except Exception as e:
            logger.error(f"Ollama chat request failed: {e}")
            return ""

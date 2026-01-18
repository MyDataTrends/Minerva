"""
OpenAI Provider - Uses OpenAI API for GPT models.
"""
import os
import logging
from typing import Optional, List, Dict

from llm_manager.providers.base import LLMProvider, ModelInfo, ProviderType

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    Provider for OpenAI API models (GPT-4, GPT-4o, etc.).
    """
    
    provider_type = ProviderType.OPENAI
    
    def __init__(self, model_info: ModelInfo, api_key: Optional[str] = None):
        super().__init__(model_info, api_key)
        self._client = None
    
    def load(self) -> bool:
        """Initialize the OpenAI client."""
        if self._loaded and self._client is not None:
            return True
        
        if not self.api_key:
            logger.error("OpenAI API key not set. Set OPENAI_API_KEY environment variable.")
            return False
        
        try:
            from openai import OpenAI
            
            self._client = OpenAI(api_key=self.api_key)
            self._loaded = True
            logger.info(f"OpenAI client initialized for: {self.model_info.name}")
            return True
            
        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return False
    
    def unload(self) -> None:
        """Nothing to unload for API clients."""
        self._client = None
        self._loaded = False
    
    def _get_model_name(self) -> str:
        """Get the OpenAI model name."""
        model_map = {
            "gpt-4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini",
            "gpt-4": "gpt-4",
            "gpt-4-turbo": "gpt-4-turbo",
            "gpt-3.5-turbo": "gpt-3.5-turbo",
        }
        return model_map.get(self.model_info.id, "gpt-4o-mini")
    
    def complete(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate completion using OpenAI API."""
        # Use chat completion with system prompt
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate chat response using OpenAI API."""
        if not self._loaded or self._client is None:
            if not self.load():
                return ""
        
        try:
            response = self._client.chat.completions.create(
                model=self._get_model_name(),
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return ""

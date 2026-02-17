"""
Anthropic Provider - Uses Anthropic API for Claude models.
"""
import logging
from typing import Optional, List, Dict

from llm_manager.providers.base import LLMProvider, ModelInfo, ProviderType

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """
    Provider for Anthropic API models (Claude 3, Claude 3.5, etc.).
    """
    
    provider_type = ProviderType.ANTHROPIC
    
    def __init__(self, model_info: ModelInfo, api_key: Optional[str] = None):
        super().__init__(model_info, api_key)
        self._client = None
    
    def load(self) -> bool:
        """Initialize the Anthropic client."""
        if self._loaded and self._client is not None:
            return True
        
        if not self.api_key:
            logger.error("Anthropic API key not set. Set ANTHROPIC_API_KEY environment variable.")
            return False
        
        try:
            from anthropic import Anthropic
            
            self._client = Anthropic(api_key=self.api_key)
            self._loaded = True
            logger.info(f"Anthropic client initialized for: {self.model_info.name}")
            return True
            
        except ImportError:
            logger.error("anthropic package not installed. Run: pip install anthropic")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            return False
    
    def unload(self) -> None:
        """Nothing to unload for API clients."""
        self._client = None
        self._loaded = False
    
    def _get_model_name(self) -> str:
        """Get the Anthropic model name."""
        # Use explicit API model ID if available, otherwise fallback to internal ID
        return self.model_info.api_model_id or self.model_info.id
    
    def complete(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate completion using Anthropic API."""
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
        """Generate chat response using Anthropic API."""
        if not self._loaded or self._client is None:
            if not self.load():
                return ""
        
        try:
            # Extract system message if present
            system = None
            api_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system = msg["content"]
                else:
                    api_messages.append(msg)
            
            params = {
                "model": self._get_model_name(),
                "messages": api_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            
            if system:
                params["system"] = system
            
            response = self._client.messages.create(**params)
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            return ""

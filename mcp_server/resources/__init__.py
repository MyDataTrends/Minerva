"""
MCP Resource Registry.

Resources provide a way for the LLM to read data directly via a URI scheme.
Unlike tools (which are active), resources are passive data reading endpoints.

URI Scheme:
    resource://{provider}/{path}

Examples:
    resource://datasets/sales_data
    resource://api/schema/users
    resource://system/logs/recent
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Callable, Awaitable
from dataclasses import dataclass
import abc

logger = logging.getLogger(__name__)


@dataclass
class ResourceInfo:
    """Information about an available resource."""
    uri: str
    name: str
    description: str
    mime_type: str = "application/json"


class BaseResourceProvider(abc.ABC):
    """Base class for resource providers."""
    
    name: str = "base"
    
    @abc.abstractmethod
    def list_resources(self) -> List[ResourceInfo]:
        """List available resources."""
        pass
    
    @abc.abstractmethod
    async def read_resource(self, uri: str, session=None) -> Any:
        """Read content of a resource."""
        pass


# Global registry of resource providers
_resource_providers: Dict[str, BaseResourceProvider] = {}


def register_provider(provider: BaseResourceProvider) -> BaseResourceProvider:
    """Register a resource provider."""
    _resource_providers[provider.name] = provider
    return provider


def get_provider(name: str) -> Optional[BaseResourceProvider]:
    """Get a provider by name."""
    return _resource_providers.get(name)


def list_all_resources() -> List[ResourceInfo]:
    """List all resources from all providers."""
    resources = []
    for provider in _resource_providers.values():
        try:
            resources.extend(provider.list_resources())
        except Exception as e:
            logger.warning(f"Error listing resources from {provider.name}: {e}")
    return resources


async def read_resource(uri: str, session=None) -> Any:
    """
    Read a resource by URI.
    
    Format: resource://{provider}/{path}
    """
    if not uri.startswith("resource://"):
        raise ValueError(f"Invalid resource URI: {uri}")
    
    # Parse URI
    try:
        parts = uri.replace("resource://", "").split("/", 1)
        if len(parts) != 2:
            raise ValueError("Invalid URI format")
        
        provider_name, path = parts
        
        provider = get_provider(provider_name)
        if not provider:
            raise ValueError(f"Unknown resource provider: {provider_name}")
            
        return await provider.read_resource(uri, session=session)
        
    except ValueError as e:
        raise
    except Exception as e:
        logger.error(f"Error reading resource {uri}: {e}")
        raise RuntimeError(f"Failed to read resource: {e}")

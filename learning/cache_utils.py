"""
Cache utilities for memory management.

Provides TTL-based caching for expensive operations like embeddings,
with configurable limits to prevent unbounded memory growth.
"""
import time
import logging
from functools import wraps
from typing import Any, Dict, Optional, Callable, Hashable
from collections import OrderedDict
import hashlib

logger = logging.getLogger(__name__)


class TTLCache:
    """
    A simple TTL (Time-To-Live) and LRU cache with size limits.
    
    Features:
    - TTL expiration: entries expire after `ttl_seconds`
    - LRU eviction: least recently used entries removed when `max_size` exceeded
    - Memory stats: track hits, misses, evictions
    
    Usage:
        cache = TTLCache(max_size=1000, ttl_seconds=3600)
        cache.set("key", value)
        result = cache.get("key")  # Returns None if expired or missing
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize TTL cache.
        
        Args:
            max_size: Maximum number of entries before LRU eviction
            ttl_seconds: Time-to-live for each entry in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[Hashable, Dict[str, Any]] = OrderedDict()
        
        # Stats
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def _make_key(self, key: Any) -> str:
        """Convert any key to a hashable string."""
        if isinstance(key, str):
            return key
        # For complex objects, hash their repr
        return hashlib.md5(repr(key).encode()).hexdigest()
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value if exists and not expired, else None."""
        cache_key = self._make_key(key)
        
        if cache_key not in self._cache:
            self._misses += 1
            return None
        
        entry = self._cache[cache_key]
        
        # Check TTL
        if time.time() - entry["timestamp"] > self.ttl_seconds:
            del self._cache[cache_key]
            self._misses += 1
            return None
        
        # Move to end (most recently used)
        self._cache.move_to_end(cache_key)
        self._hits += 1
        return entry["value"]
    
    def set(self, key: Any, value: Any) -> None:
        """Set value with current timestamp."""
        cache_key = self._make_key(key)
        
        # If exists, remove first to update order
        if cache_key in self._cache:
            del self._cache[cache_key]
        
        # Evict LRU if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)  # Remove oldest
            self._evictions += 1
        
        self._cache[cache_key] = {
            "value": value,
            "timestamp": time.time()
        }
    
    def clear(self) -> None:
        """Clear all entries."""
        self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": round(hit_rate, 1),
        }


def ttl_cache(max_size: int = 128, ttl_seconds: int = 3600):
    """
    Decorator for caching function results with TTL.
    
    Usage:
        @ttl_cache(max_size=100, ttl_seconds=600)
        def expensive_function(arg):
            ...
    """
    cache = TTLCache(max_size=max_size, ttl_seconds=ttl_seconds)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key = (func.__name__, args, tuple(sorted(kwargs.items())))
            
            result = cache.get(key)
            if result is not None:
                return result
            
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        
        # Attach cache for inspection/clearing
        wrapper.cache = cache
        wrapper.cache_info = cache.get_stats
        wrapper.cache_clear = cache.clear
        return wrapper
    
    return decorator


# Global embedding cache instance (1000 embeddings, 1 hour TTL)
_embedding_cache: Optional[TTLCache] = None


def get_embedding_cache() -> TTLCache:
    """Get or create the global embedding cache."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = TTLCache(max_size=1000, ttl_seconds=3600)
    return _embedding_cache

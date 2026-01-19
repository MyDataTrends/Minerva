"""
Lazy Import Utilities for MCP.

Provides deferred importing of heavy libraries to speed up module load times.
Libraries like pandas, sklearn, plotly are only imported when actually needed.
"""

from typing import Any, Callable
import sys


class LazyLoader:
    """
    A lazy loader that defers importing a module until first access.
    
    Usage:
        pd = LazyLoader("pandas")
        # pandas is not imported yet
        df = pd.DataFrame()  # NOW pandas is imported
    """
    
    def __init__(self, module_name: str):
        self._module_name = module_name
        self._module = None
    
    def _load(self):
        if self._module is None:
            self._module = __import__(self._module_name, fromlist=[''])
        return self._module
    
    def __getattr__(self, name: str) -> Any:
        module = self._load()
        return getattr(module, name)
    
    def __repr__(self) -> str:
        if self._module:
            return repr(self._module)
        return f"<LazyLoader({self._module_name!r})>"


def lazy_import(module_name: str) -> LazyLoader:
    """Create a lazy import for a module."""
    return LazyLoader(module_name)


# Pre-configured lazy loaders for common heavy libraries
def get_pandas():
    """Get pandas, importing lazily on first use."""
    import pandas
    return pandas


def get_numpy():
    """Get numpy, importing lazily on first use."""
    import numpy
    return numpy


def get_sklearn():
    """Get sklearn, importing lazily on first use."""
    import sklearn
    return sklearn


def get_plotly():
    """Get plotly.express, importing lazily on first use."""
    import plotly.express
    return plotly.express


def get_plotly_go():
    """Get plotly.graph_objects, importing lazily on first use."""
    import plotly.graph_objects
    return plotly.graph_objects


# Cached instances for performance
_pandas = None
_numpy = None


def pd():
    """Get cached pandas instance."""
    global _pandas
    if _pandas is None:
        import pandas
        _pandas = pandas
    return _pandas


def np():
    """Get cached numpy instance."""
    global _numpy
    if _numpy is None:
        import numpy
        _numpy = numpy
    return _numpy

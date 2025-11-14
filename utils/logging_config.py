import warnings
from utils.logging import configure_logging as _configure_logging

_warned = False

def configure_logging(*args, **kwargs):
    """Shim for deprecated utils.logging_config module."""
    global _warned
    if not _warned:
        warnings.warn(
            "utils.logging_config is deprecated; use utils.logging instead",
            DeprecationWarning,
            stacklevel=2,
        )
        _warned = True
    return _configure_logging(*args, **kwargs)

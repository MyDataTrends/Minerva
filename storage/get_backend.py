import os

from .base import StorageBackend
from .local_backend import LocalStorage
from .s3_backend import S3Storage


def get_backend(name: str) -> StorageBackend:
    """Return a storage backend instance for ``name``.

    Parameters
    ----------
    name:
        Identifier of the backend implementation.

    Returns
    -------
    StorageBackend

    Raises
    ------
    ValueError
        If ``name`` is not one of ``"local"`` or ``"s3"``.
    """

    name = name.lower()
    if name == "s3":
        return S3Storage()
    elif name == "local":
        return LocalStorage()
    else:
        raise ValueError(
            f"Unknown backend '{name}'. Supported: 'local', 's3'."
        )


def _init_backend() -> StorageBackend:
    backend_name = os.getenv("APP_STORAGE_BACKEND", "local")
    return get_backend(backend_name)


backend: StorageBackend = _init_backend()


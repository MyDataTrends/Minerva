from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    """Abstract interface for storage backends."""

    @abstractmethod
    def get_file(self, path: str) -> Path:
        """Retrieve a file at ``path`` and return a local ``Path``."""

    @abstractmethod
    def put_file(self, src: Path, dest: str) -> None:
        """Store ``src`` at ``dest`` within the backend."""

    @abstractmethod
    def list_files(self, prefix: str = "") -> list[str]:
        """List files under ``prefix`` in the backend."""

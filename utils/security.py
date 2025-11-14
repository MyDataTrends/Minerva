from __future__ import annotations

from pathlib import Path

__all__ = ["secure_join"]


def secure_join(base: Path | str, *parts: str | Path) -> Path:
    """Safely join ``parts`` to ``base`` resolving symlinks.

    The returned path is guaranteed to reside within ``base``. Any attempt to
    traverse outside ``base`` via ``..`` or symlinks results in a ``ValueError``.
    """

    base_path = Path(base).resolve()
    current = base_path
    for part in parts:
        current = (current / part).resolve()
        if not current.is_relative_to(base_path):
            raise ValueError("Attempted path traversal outside base directory")
    return current

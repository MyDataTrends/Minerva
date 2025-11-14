from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

CATEGORY_CATALOG_PATH = Path(__file__).resolve().parents[1] / "catalog" / "category_catalog.json"
TAG_CATALOG_PATH = Path(__file__).resolve().parents[1] / "catalog" / "fine_grained_tags.json"


def _load_catalog(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def _save_catalog(catalog: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, indent=2))


def save_user_category(user_id: str, category: str, path: Path = CATEGORY_CATALOG_PATH) -> None:
    """Persist a user's selected category."""
    catalog = _load_catalog(path)
    catalog[user_id] = category
    _save_catalog(catalog, path)


def get_user_category(user_id: str, path: Path = CATEGORY_CATALOG_PATH) -> Optional[str]:
    """Retrieve a saved category for a user."""
    catalog = _load_catalog(path)
    return catalog.get(user_id)


def save_fine_grained_tags(dataset_name: str, tags: str, path: Path = TAG_CATALOG_PATH) -> None:
    """Persist fine-grained LLM tags for a dataset."""
    catalog = _load_catalog(path)
    catalog[dataset_name] = tags
    _save_catalog(catalog, path)


def get_fine_grained_tags(dataset_name: str, path: Path = TAG_CATALOG_PATH) -> Optional[str]:
    """Retrieve stored LLM tags for a dataset."""
    catalog = _load_catalog(path)
    return catalog.get(dataset_name)

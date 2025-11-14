"""Dataset tagging utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .metadata_parser import parse_metadata
from .llm_preprocessor import tag_dataset_with_llm

# Default location for the lightweight catalog of dataset tags
CATALOG_PATH = Path(__file__).resolve().parents[1] / "catalog" / "tag_catalog.json"


def generate_tags(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate tags for a DataFrame using metadata and LLM tagging."""
    metadata = parse_metadata(df)
    llm_tags = tag_dataset_with_llm(df)
    tags: Dict[str, Any] = {
        "columns": metadata.get("columns", []),
        "dtypes": metadata.get("dtypes", {}),
        "llm_tags": llm_tags,
    }
    return tags


def _load_catalog(path: Path = CATALOG_PATH) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def _save_catalog(catalog: Dict[str, Any], path: Path = CATALOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, indent=2))


def store_tags(dataset_name: str, tags: Dict[str, Any], path: Path = CATALOG_PATH) -> None:
    """Store tags for a dataset in the catalog."""
    catalog = _load_catalog(path)
    catalog[dataset_name] = tags
    _save_catalog(catalog, path)


def get_tags(dataset_name: str, path: Path = CATALOG_PATH) -> Dict[str, Any] | None:
    """Retrieve tags for a dataset from the catalog."""
    catalog = _load_catalog(path)
    return catalog.get(dataset_name)

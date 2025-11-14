from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_ROLES_PATH = Path(__file__).resolve().parents[1] / "config" / "semantic_roles.yaml"


@lru_cache()
def _load_roles() -> list[str]:
    with open(_ROLES_PATH, "r", encoding="utf-8") as f:
        data = json.loads(json.dumps(__import__('yaml').safe_load(f))) or {}
    return list(data.keys())


def _encode(texts: list[str]) -> np.ndarray:
    """Return embeddings for ``texts`` using MiniLM if available."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(texts)
    except Exception:
        vec = TfidfVectorizer().fit(texts)
        return vec.transform(texts).toarray()


def map_description_to_role(description: str) -> str:
    """Map a free-text ``description`` to the closest known role."""
    roles = _load_roles()
    if not roles:
        return "unknown"
    texts = [description] + roles
    emb = _encode(texts)
    sims = cosine_similarity([emb[0]], emb[1:])[0]
    best = int(np.argmax(sims))
    return roles[best]


def map_descriptions(descriptions: Iterable[str]) -> dict[str, str]:
    """Map each description in ``descriptions`` to a role."""
    return {d: map_description_to_role(d) for d in descriptions}

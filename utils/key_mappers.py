import hashlib
from typing import Dict, List

import pandas as pd

# Minimal placeholder mappings for demonstration purposes
_ZIP_TO_FIPS: Dict[str, str] = {
    "02108": "25025",
    "30301": "13121",
}
_FIPS_TO_ZIP: Dict[str, str] = {v: k for k, v in _ZIP_TO_FIPS.items()}


def zip_to_fips(zip_code: str) -> str:
    """Return FIPS code for a ZIP code if known."""
    return _ZIP_TO_FIPS.get(str(zip_code), str(zip_code))


def fips_to_zip(fips: str) -> str:
    """Return ZIP code for a FIPS code if known."""
    return _FIPS_TO_ZIP.get(str(fips), str(fips))


def city_state_to_hash(city: str, state: str) -> str:
    """Hash city and state into a stable key."""
    text = f"{city.lower()}_{state.lower()}"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def hash_columns(df: pd.DataFrame, columns: List[str], new_col: str) -> pd.DataFrame:
    """Add a SHA-1 hash column based on ``columns``."""
    df[new_col] = (
        df[columns]
        .astype(str)
        .agg("_".join, axis=1)
        .apply(lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest())
    )
    return df

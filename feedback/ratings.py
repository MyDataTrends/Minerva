import json
import os
from datetime import datetime
from pathlib import Path
from utils.security import secure_join


def store_rating(rating: int, file_path: str = "feedback/ratings.json") -> None:
    """Append a rating to the JSON file with a timestamp.

    Parameters
    ----------
    rating : int
        Score provided by the user (1-5).
    file_path : str, optional
        Location of the JSON file used to persist ratings.
    """
    dest_dir = Path(file_path).parent
    safe_path = secure_join(dest_dir, Path(file_path).name)
    os.makedirs(dest_dir, exist_ok=True)
    try:
        with open(safe_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []

    data.append({"timestamp": datetime.utcnow().isoformat(), "rating": int(rating)})
    with open(safe_path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def get_average_rating(file_path: str = "feedback/ratings.json") -> float:
    """Return the average rating stored in ``file_path``.

    Parameters
    ----------
    file_path : str, optional
        Location of the JSON file used to persist ratings.

    Returns
    -------
    float
        Average rating across all entries. ``0.0`` if no ratings exist.
    """

    dest_dir = Path(file_path).parent
    safe_path = secure_join(dest_dir, Path(file_path).name)
    try:
        with open(safe_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return 0.0

    if not data:
        return 0.0

    total = sum(int(item.get("rating", 0)) for item in data)
    return total / len(data)

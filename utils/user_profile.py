import json
import os
from pathlib import Path
from utils.security import secure_join

__all__ = ["get_user_tier"]


def _profile_path() -> Path:
    env_path = os.getenv("USER_PROFILE_PATH", "user_profile.json")
    p = Path(env_path)
    if p.is_absolute():
        return p
    return secure_join(Path.cwd(), env_path)


def get_user_tier(user_id: str) -> str:
    path = _profile_path()
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return "free"
    info = data.get(user_id, {})
    return info.get("tier", "free")

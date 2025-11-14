from pathlib import Path

from tagging.category_manager import (
    save_user_category,
    get_user_category,
    save_fine_grained_tags,
    get_fine_grained_tags,
)


def test_save_and_get_user_category(tmp_path: Path):
    catalog = tmp_path / "cat.json"
    save_user_category("u1", "finance", path=catalog)
    assert get_user_category("u1", path=catalog) == "finance"


def test_save_and_get_fine_grained_tags(tmp_path: Path):
    catalog = tmp_path / "tags.json"
    save_fine_grained_tags("data.csv", "cols: A,B", path=catalog)
    assert get_fine_grained_tags("data.csv", path=catalog) == "cols: A,B"

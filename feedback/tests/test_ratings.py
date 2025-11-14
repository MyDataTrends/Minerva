import json
from feedback.ratings import store_rating, get_average_rating

def test_store_rating(tmp_path):
    file_path = tmp_path / "ratings.json"
    store_rating(4, file_path=str(file_path))
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["rating"] == 4


def test_get_average_rating(tmp_path):
    file_path = tmp_path / "ratings.json"
    store_rating(3, file_path=str(file_path))
    store_rating(5, file_path=str(file_path))
    avg = get_average_rating(str(file_path))
    assert avg == 4.0

from pathlib import Path
import pytest

from storage.local_backend import LocalStorage
from modeling.model_training import load_model


def test_put_file_traversal(tmp_path: Path):
    backend = LocalStorage(base_dir=tmp_path)
    src = tmp_path / "src.txt"
    src.write_text("hi")
    with pytest.raises(ValueError):
        backend.put_file(src, "../evil.txt")


def test_model_load_traversal(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        load_model("../../etc/passwd")

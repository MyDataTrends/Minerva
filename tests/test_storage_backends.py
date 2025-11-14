from pathlib import Path

from storage.local_backend import LocalStorage
from storage.s3_backend import S3Storage
from storage.get_backend import get_backend
import pytest


def test_local_storage_roundtrip(local_storage: LocalStorage, tmp_path: Path):
    src = tmp_path / "src.txt"
    src.write_text("hello")
    local_storage.put_file(src, "sub/data.txt")
    result = local_storage.get_file("sub/data.txt")
    assert result.read_text() == "hello"
    files = local_storage.list_files("sub")
    assert "sub/data.txt" in files and len(files) == 1


def test_s3_storage_roundtrip(s3_storage: S3Storage, tmp_path: Path):
    src = tmp_path / "src.txt"
    src.write_text("hello")
    s3_storage.put_file(src, "folder/data.txt")
    keys = s3_storage.list_files("folder")
    assert "folder/data.txt" in keys
    out = s3_storage.get_file("folder/data.txt")
    assert out.read_text() == "hello"


def test_get_backend_invalid():
    with pytest.raises(ValueError):
        get_backend("missing")



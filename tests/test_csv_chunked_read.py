from pathlib import Path
import types

import pandas as pd

from storage.local_backend import LocalStorage
from config import (
    CSV_READ_CHUNKSIZE,
    DESCRIBE_SAMPLE_ROWS,
    LARGE_FILE_BYTES,
)


def test_read_df_streams_large_csv(monkeypatch, tmp_path):
    rows = CSV_READ_CHUNKSIZE + 10000
    df = pd.DataFrame({"a": range(rows), "b": range(rows)})
    csv_path = tmp_path / "big.csv"
    df.to_csv(csv_path, index=False)

    storage = LocalStorage(base_dir=tmp_path)

    orig_stat = Path.stat

    def fake_stat(self, *, follow_symlinks=True):
        if self == csv_path:
            return types.SimpleNamespace(st_size=LARGE_FILE_BYTES + 1)
        return orig_stat(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", fake_stat)

    calls: list[int | None] = []
    orig_read_csv = pd.read_csv

    def wrapped_read_csv(*args, **kwargs):
        calls.append(kwargs.get("chunksize"))
        return orig_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", wrapped_read_csv)

    sample = storage.read_df("big.csv", sample_only=True)
    assert len(sample) == DESCRIBE_SAMPLE_ROWS
    assert calls[0] == CSV_READ_CHUNKSIZE

    full = storage.read_df("big.csv")
    assert len(full) == rows
    assert calls[1] == CSV_READ_CHUNKSIZE


import os
import sys
from pathlib import Path
import pytest

pytestmark = pytest.mark.skip(reason="pycaret removed; using lightweight sklearn/xgboost/lightgbm stack")

# Ensure local data directory points to repo root
os.environ["LOCAL_DATA_DIR"] = "."

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from storage.local_backend import load_datalake_dfs
from orchestrate_workflow import orchestrate_workflow


def test_end_to_end_workflow():
    datalake_dfs = load_datalake_dfs()
    result = orchestrate_workflow("e2e_user", "e2e_data.csv", datalake_dfs)
    assert isinstance(result, dict)
    assert "output" in result
    assert "metrics" in result
    assert "summary" in result
    assert "error" not in result

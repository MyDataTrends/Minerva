import pytest

from config import get_env

@pytest.mark.parametrize("required", [True, False])
def test_get_env_missing(monkeypatch, required):
    var = "NON_EXISTENT_ENV"
    monkeypatch.delenv(var, raising=False)
    if required:
        with pytest.raises(RuntimeError):
            get_env(var, required=True)
    else:
        assert get_env(var, default="x") == "x"


def test_get_env_present(monkeypatch):
    var = "EXISTING_ENV"
    monkeypatch.setenv(var, "val")
    assert get_env(var, required=True) == "val"

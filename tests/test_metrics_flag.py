import socket
import subprocess
import sys
import time
import os
import pytest


def _can_bind_port(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("localhost", port))
        except OSError:
            return False
    return True


def test_import_does_not_bind_port(monkeypatch):
    port = 8000
    if not _can_bind_port(port):
        pytest.skip("port is already in use")

    env = os.environ.copy()
    env["ENABLE_PROMETHEUS"] = "1"
    proc = subprocess.Popen([
        sys.executable,
        "-c",
        "import main; import time; time.sleep(1)",
    ], env=env)

    try:
        # Give the subprocess a moment to potentially start the server
        time.sleep(0.2)
        assert _can_bind_port(port)
    finally:
        proc.terminate()
        proc.wait()

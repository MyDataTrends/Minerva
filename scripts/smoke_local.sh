#!/usr/bin/env bash
set -euo pipefail

tmp_csv=$(mktemp)
cleanup() {
  rm -f "$tmp_csv"
  if [[ -n "${server_pid:-}" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Create a temporary 50k-row CSV
python - <<'PY' "$tmp_csv"
import csv, sys
path = sys.argv[1]
with open(path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['value'])
    for i in range(50000):
        w.writerow([i])
PY

# Run metadata parse CLI
python - <<'PY' "$tmp_csv"
import pandas as pd, sys
from preprocessing.metadata_parser import parse_metadata
parse_metadata(pd.read_csv(sys.argv[1]))
PY

# Start the app without LLM
python main.py --no-llm >/tmp/smoke_server.log 2>&1 &
server_pid=$!

# Wait for server to be ready
for _ in {1..10}; do
  if curl -sSf http://localhost:8000/healthz >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

# Verify health and readiness endpoints
curl -sSf http://localhost:8000/healthz >/dev/null
curl -s http://localhost:8000/readyz >/dev/null

echo "Smoke test passed"

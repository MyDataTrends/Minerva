# Minerva – Modular Data Analysis Platform

Minerva ingests user data, enriches it with public datasets, selects a suitable analysis path, trains a model when appropriate, and renders results in a Streamlit dashboard and a small FastAPI service.

This refactor clarifies module boundaries, improves performance, and adds backward-compatible shims to ease migration.

## What’s new (refactor highlights)

- **Structure**

  - `orchestration/analysis_selector.py` now owns analyzer selection. A shim at project root re-exports `select_analyzer`.

  - `integration/semantic_merge.py` contains semantic merge logic. `Integration/semantic_integration.py` is now a documented compatibility shim with a deprecation warning.

  - Analyzers live under `modeling/analyzers/` with shims in old locations.

- **Reliability**

  - Models persist per run at `models/<run_id>/best_model` and are loaded reliably on reruns.

- **Correctness**

  - Categorical encoding only applies to object dtype columns; numeric columns are no longer encoded as categories.

  - The model wrapper now passes through DataFrame inputs to `predict`.

- **Performance**

  - Quick model search uses parallel CV (`ML_N_JOBS`), histogram tree method for XGBoost, and reasonable estimator counts.

## Quick start

1) Install dependencies

   - pip

     ```bash
     python -m venv venv && . venv/bin/activate
     pip install -r requirements.txt
     ```

   - conda

     ```bash
     conda env create -f environment.yml
     conda activate my_env
     ```

2) Optional: ingest sample data and build semantic index

   ```bash
   python -m Data_Intake.datalake_ingestion s3 --bucket my-bucket --prefix retail/ --dest datasets
   # or
   python -m Data_Intake.datalake_ingestion api https://example.com/data.csv --dest datasets
   ```

3) Run tests and launch the dashboard

   ```bash
   pytest -q
   streamlit run ui/dashboard.py
   ```

4) View session history

   ```bash
   streamlit run ui/session_history.py
   ```

## Directory overview

- `orchestration/` – preprocessing, enrichment, analyzer selection, output generation, agent triggers

- `integration/` – semantic dataset merge logic (`semantic_merge.py`)

- `modeling/` – analyzers, model selection/training, metrics

- `preprocessing/` – data cleaning, metadata parsing, LLM helpers

- `storage/` – local and cloud storage helpers, session DB

- `ui/` – Streamlit dashboard and session history

- `Data_Intake/` – ingestion CLI for S3/API and semantic index build

- `config/` – env parsing and feature flags

- `tests/` – unit/integration tests

- `docs/` – architecture, onboarding, and use cases documentation

- `infra/environments` – Conda env specs

- `infra/requirements` – pip requirement pins

- `examples/` – runnable demos (e.g., `test_workflow_demo.py`, `imputation_confidence_demo.py`)

- `tools/` – CLI utilities (e.g., `alignment_drift_monitor.py`)

## Configuration

Set in `.env` or environment variables. Key options:

- General

  - `LOCAL_DATA_DIR` (default: `local_data`)

  - `LOG_LEVEL` (default: `INFO`), `LOG_FILE` (default: `app.log`)

  - `LOG_DIR` (default: `logs`) – centralized log folder; logs go to `LOG_DIR/LOG_FILE`

- Feature flags (config/feature_flags.py)

  - `ALLOW_FULL_COMPARE_MODELS` (default: `False`) – restricts heavy model comparison runs (now using lightweight sklearn/xgboost/lightgbm stack)

  - `MAX_ROWS_FIRST_PASS` (default: `25000`), `MAX_FEATURES_FIRST_PASS` (default: `100`)

  - `MAX_ROWS_FULL` (default: `5000`, via env in analysis_router)

  - `MODEL_TIME_BUDGET_SECONDS` (default: `60`)

  - `ENABLE_HEAVY_EXPLANATIONS` (default: `False`)

- Performance

  - `ML_N_JOBS` (default: `-1`) – parallelism for quick model search and RandomForest

- Free tier limits

  - `MAX_REQUESTS_FREE` (default: `20`), `MAX_GB_FREE` (default: `1`)

## How it works

1) Ingest: `Data_Intake.datalake_ingestion` pulls data into `datasets/` and builds a semantic index.

2) Orchestrate: `orchestrate_workflow.py` loads the file, runs light validation/cleaning, and optional diagnostics.

3) Enrich: `integration.semantic_merge` ranks public tables and merges the best fit.

4) Analyze: `orchestration.analysis_selector.select_analyzer` picks one Analyzer from the registry and runs it.

5) Output: predictions, summaries, and artifacts are generated and stored; agents may trigger follow-ups.

## Modeling policy and routing

- `modeling/suitability_check.assess_modelability` does a cheap feasibility pass.

- `orchestration.analysis_router.route_analysis` decides `no_model | baseline | full` based on stats, flags, row caps, and optional hints.

- `modeling/model_selector.select_best_model` uses:

  - Heavy compare only if `ALLOW_FULL_COMPARE_MODELS=True` and constrained by `MODEL_ALLOWLIST` (using lightweight sklearn/xgboost/lightgbm stack).

  - Otherwise, a fast CV sweep over a small curated candidate set.

## API (FastAPI)

- `GET /healthz` – service health

- `GET /sessions` – list recent sessions

- `GET /sessions/{run_id}` – fetch one

- `POST /sessions/{run_id}/rerun` – rerun a session

Run locally:

```bash
uvicorn main:app --reload
```

## Fresh setup (Windows + VS Code)

- Create venv and activate

  - `python -m venv .venv`

  - `\.venv\Scripts\Activate.ps1`

- Install dependencies

  - `pip install -r infra\requirements\requirements.txt`

- Configure env

  - `copy config/.env.example config/.env` and edit (set `LOG_DIR`, `LOCAL_DATA_DIR`, etc.)

- Run tests

  - `pytest -q`

- Demos

  - `python -m examples.test_workflow_demo`

  - `python -m examples.imputation_confidence_demo`

- Services

  - API: `uvicorn main:app --reload`

  - Dashboard: `streamlit run ui\dashboard.py`

See `docs/Onboarding_Windows.md` for a detailed walkthrough and `docs/Use_Cases.md` for common scenarios.

## CLI flags (mirrored to env vars)

- `--no-llm` → `ENABLE_LOCAL_LLM=0`

- `--enable-prometheus` → `ENABLE_PROMETHEUS=1`

- `--safe-logs` → `REDACTION_ENABLED=1`

- `--dev-lenient` → `LOCAL_DEV_LENIENT=1`

## Reruns and persistence

- Best model is saved at `models/<run_id>/best_model`.

- Reruns load the persisted model and re-evaluate metrics without retraining when possible.

## Deprecations and migration notes

- `Integration.semantic_integration` is deprecated in favor of `integration.semantic_merge`.

  - A compatibility shim remains and emits a `DeprecationWarning`.

- Analyzer files at `modeling/*_analyzer.py` re-export classes from `modeling/analyzers/`.

- The top-level `analysis_selector.py` re-exports from `orchestration.analysis_selector`.

## Development

- Tests

  ```bash
  pytest -q
  pytest tests/test_end_to_end_workflow.py
  ```

- Dashboard

  ```bash
  streamlit run ui/dashboard.py
  ```

## Example: end-to-end (Python)

```python
import pandas as pd
from orchestrate_workflow import orchestrate_workflow

datalake = {
    "merge_df1.csv": pd.read_csv("datasets/merge_df1.csv"),
    "merge_df2.csv": pd.read_csv("datasets/merge_df2.csv"),
}
result = orchestrate_workflow(
    user_id="demo",
    file_name="sample_dataset.csv",
    datalake_dfs=datalake,
)
print(result.get("summary"))
```

from __future__ import annotations
from pathlib import Path
from storage.get_backend import backend
from modeling.model_selector import evaluate_model
from modeling.model_training import train_model, save_model, load_model
from preprocessing.llm_summarizer import generate_summary
from storage.local_backend import load_run_metadata
from orchestration.data_preprocessor import DataPreprocessor
from preprocessing.metadata_parser import parse_metadata, pre_scan_metadata
from config import (
    MAX_ROWS_FIRST_PASS,
    MAX_FEATURES_FIRST_PASS,
    PROFILE_SAMPLE_ROWS,
    MAX_REQUESTS_FREE,
    MAX_GB_FREE,
)
from utils.usage_tracker import check_quota

import logging
from utils.logging import configure_logging
import pandas as pd

from workflow_manager import WorkflowManager
import json
from orchestrator import orchestrate_dashboard

configure_logging()


def orchestrate_workflow(
    user_id: str | None = None,
    file_name: str | None = None,
    datalake_dfs=None,
    target_column=None,
    category=None,
    user_labels: dict[str, str] | None = None,
    run_id: str | None = None,
    diagnostics_config: dict | None = None,
):
    """Entry point for running the workflow via ``WorkflowManager``."""
    df = None
    if datalake_dfs:
        df = next(iter(datalake_dfs.values()), None)
    elif user_id and file_name:
        try:
            dp = DataPreprocessor()
            df, _ = dp.load_dataset(user_id, file_name, run_id)
        except Exception as exc:  # pragma: no cover - safety
            logging.warning("Pre-scan failed: %s", exc)

    if df is not None:
        scan = pre_scan_metadata(df)
        if scan["rows"] > MAX_ROWS_FIRST_PASS or scan["columns"] > MAX_FEATURES_FIRST_PASS:
            logging.warning(
                "Dataset over caps - rows: %d cols: %d mem_mb: %.2f",
                scan["rows"],
                scan["columns"],
                scan["memory_bytes"] / (1024 ** 2),
            )
            sample = df.head(PROFILE_SAMPLE_ROWS)
            summary = parse_metadata(sample)
            return {"error": "dataset_over_cap", "summary": summary}

    manager = WorkflowManager(
        user_id or "",
        file_name or "",
        datalake_dfs,
        target_column=target_column,
        category=category,
        user_labels=user_labels,
        run_id=run_id,
    )
    result = manager.preprocess_and_cache(diagnostics_config)
    if result:
        return result
    manager.enrich_with_public_data()

    warning = None
    if user_id:
        allowed, msg, _ = check_quota(user_id)
        if not allowed:
            return msg or {"error": "quota_exceeded"}
        warning = msg.get("warning") if msg else None

    result = manager.select_and_run_analysis()
    if result:
        if warning:
            result["warning"] = warning
        return result

    if user_id:
        allowed, msg, _ = check_quota(user_id)
        if not allowed:
            return msg or {"error": "quota_exceeded"}
        if warning is None and msg:
            warning = msg.get("warning")

    manager.generate_outputs()
    if manager.result and warning:
        manager.result["warning"] = warning
    manager.trigger_agent_actions()
    return manager.result or {}

def run_workflow(data: pd.DataFrame, target: str = None):
    """
    UI-friendly shim around the core `orchestrate_workflow` engine.

    Writes the DataFrame to the expected local path, then calls `orchestrate_workflow`.

    Args:
        data (pd.DataFrame): The input dataset for processing.
        target (str, optional): The target column name. Defaults to None.

    Returns:
        dict: The result of the workflow execution.

    Side Effects:
        Ensures the expected directory structure exists and writes the input DataFrame to a CSV file.
    """
    user_id = "local_user"
    file_name = "local_upload.csv"
    user_dir = Path("local_data") / "User_Data" / user_id
    user_dir.mkdir(parents=True, exist_ok=True)

    # Persist CSV for the orchestrator
    csv_path = user_dir / file_name
    data.to_csv(csv_path, index=False)

    # Delegate to the core engine
    return orchestrate_workflow(
        user_id=user_id,
        file_name=file_name,
        datalake_dfs={file_name: data},
        target_column=target,
    )


def setup(*_args, **_kwargs):
    """Legacy no-op setup hook preserved for backward-compatible tests."""
    return None
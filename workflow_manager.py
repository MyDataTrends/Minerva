from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import logging

import pandas as pd

from orchestration.data_preprocessor import DataPreprocessor
from orchestration.semantic_enricher import SemanticEnricher
from orchestration.analyzer_selector_helper import AnalyzerSelector
from orchestration.output_generator import OutputGenerator
from orchestration.agent_trigger import AgentTrigger
from orchestration.metadata_cache import MetadataCache


class WorkflowManager:
    """Coordinate workflow steps using helper classes."""

    def __init__(
        self,
        user_id: str,
        file_name: str,
        datalake_dfs: Optional[dict] = None,
        target_column: Optional[str] = None,
        category: Optional[str] = None,
        user_labels: Optional[dict[str, str]] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self.user_id = user_id
        self.file_name = file_name
        self.datalake_dfs = datalake_dfs or {}
        self.target_column = target_column
        self.category = category
        self.user_labels = user_labels
        self.run_id = run_id

        self.data: Optional[pd.DataFrame] = None
        self.result: Optional[dict] = None
        self.best_model = None
        self.best_predictions = None
        self.best_df = None
        self.best_score = 0.0
        self.roles_dict: Dict[str, str] = {}
        self.needs_role_review = False
        self.metrics_history: Dict[str, Any] = {}
        self.model_info: Dict[str, Any] = {}
        self.cache = MetadataCache()
        self.diagnostics: Dict[str, Any] = {}

    # ----- Step 1 -----
    def preprocess_and_cache(
        self, diagnostics_config: Optional[Dict[str, Any]] = None
    ) -> Optional[dict]:
        """Load, validate and optionally run diagnostics on the dataset."""

        diagnostics_config = diagnostics_config or {}
        check_misalignment = diagnostics_config.get("check_misalignment", False)
        check_context_missing = diagnostics_config.get("check_context_missing", False)
        score_imputations = diagnostics_config.get("score_imputations", False)
        monitor_drift = diagnostics_config.get("monitor_drift", False)
        expected_schema = diagnostics_config.get("expected_schema")

        activated = [
            name
            for name, flag in [
                ("misalignment", check_misalignment),
                ("context_missing", check_context_missing),
                ("score_imputations", score_imputations),
                ("monitor_drift", monitor_drift),
                ("schema_validation", bool(expected_schema)),
            ]
            if flag
        ]
        if activated:
            logging.info("Diagnostics enabled: %s", ", ".join(activated))

        pre = DataPreprocessor()
        data, target, err, path = pre.run(
            self.user_id,
            self.file_name,
            self.target_column,
            self.run_id,
        )
        if err is not None:
            self.result = err
            return self.result
        if data is None:
            return None

        if expected_schema is not None:
            try:
                pre.validate(data, expected_schema=expected_schema)
            except ValueError as exc:
                self.result = {"error": str(exc)}
                return self.result

        misalignment_schema = None
        if check_misalignment:
            misalignment_schema = {
                c: (int if str(data[c].dtype).startswith("int") else float if str(data[c].dtype).startswith("float") else bool if str(data[c].dtype) == "bool" else str)
                for c in data.columns
            }

        baseline_stats = None
        if monitor_drift:
            from alignment_drift_monitor import generate_historical_stats

            baseline_stats = generate_historical_stats(data.fillna(0))

        return_diag = any(
            [check_misalignment, check_context_missing, score_imputations, monitor_drift]
        )
        cleaned = pre.clean(
            data,
            check_misalignment=check_misalignment,
            misalignment_schema=misalignment_schema,
            check_context_missing=check_context_missing,
            score_imputations_flag=score_imputations,
            monitor_drift=monitor_drift,
            baseline_stats=baseline_stats,
            return_diagnostics=return_diag,
        )

        if return_diag:
            cleaned_data, diagnostics = cleaned  # type: ignore[assignment]
            self.diagnostics = diagnostics
            self.data = cleaned_data
        else:
            self.data = cleaned  # type: ignore[assignment]
            self.diagnostics = {}

        self.target_column = target
        self.file_path = path
        return None

    # ----- Step 2 -----
    def enrich_with_public_data(self) -> None:
        enricher = SemanticEnricher()
        if self.data is not None:
            self.data = enricher.enrich(
                self.data,
                self.datalake_dfs,
                self.category,
                self.file_name,
                self.user_id,
            )

    # ----- Step 3 -----
    def select_and_run_analysis(self, intent: Optional[dict] = None) -> Optional[dict]:
        if self.data is None or self.target_column is None:
            self.result = {"error": "No data"}
            return self.result
        modeling_required = True
        reasoning = ""
        if intent and intent.get("modeling_decision"):
            md = intent["modeling_decision"]
            modeling_required = md.get("modeling_required", True)
            reasoning = md.get("reasoning", "")

        if not modeling_required:
            from analysis_selector import select_analyzer
            analyzer = select_analyzer(self.data, preferred="DescriptiveAnalyzer")
            desc_res = analyzer.run(self.data)
            self.result = {
                "analysis_type": "descriptive",
                "stats": desc_res.get("artifacts"),
                "modeling_decision": intent.get("modeling_decision") if intent else {},
            }
            return None

        selector = AnalyzerSelector()
        res = selector.analyze(
            self.data,
            self.target_column,
            self.datalake_dfs,
            self.user_labels,
            self.run_id,
        )
        if res.get("error") or res.get("modeling_failed") or "_model" not in res:
            self.result = res
            return res
        self.result = res
        self.best_model = res.pop("_model")
        self.best_predictions = res.pop("_preds")
        self.best_df = res.pop("_best_df")
        self.roles_dict = res.pop("_roles_dict")
        self.metadata_file = res.pop("_metadata_file")
        self.best_score = res.pop("_best_score")
        self.metrics_history = res.get("metrics", {})
        self.model_info = res.get("model_info", {})
        self.needs_role_review = res.get("needs_role_review", False)
        self.run_id = res.get("run_id", self.run_id)
        return None

    # ----- Step 4 -----
    def generate_outputs(self) -> None:
        if self.best_model is None or self.best_predictions is None:
            return
        generator = OutputGenerator()
        self.result = generator.generate(
            self.best_model,
            self.best_predictions,
            self.metrics_history,
            self.model_info,
            self.run_id,
            self.data,
            self.target_column,
            self.needs_role_review,
            self.file_name,
        )
        if self.diagnostics:
            self.result["diagnostics"] = self.diagnostics

    # ----- Step 5 -----
    def trigger_agent_actions(self) -> None:
        if self.result is None or self.best_predictions is None:
            return
        trigger = AgentTrigger()
        trigger.trigger(
            self.result,
            self.best_predictions,
            self.best_score,
            self.file_name,
            self.best_model,
            self.best_df,
            self.roles_dict,
        )

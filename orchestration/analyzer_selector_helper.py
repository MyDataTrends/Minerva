from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
from uuid import uuid4

import pandas as pd
from sklearn.metrics import mean_absolute_error

from preprocessing.metadata_parser import infer_column_meta, merge_user_labels
from preprocessing.save_meta import save_column_roles, _hash_df
from Integration.semantic_integration import rank_and_merge
from orchestration.analysis_selector import select_analyzer
from preprocessing.llm_preprocessor import recommend_models_with_llm
from preprocessing.llm_analyzer import ask_follow_up_question
from preprocessing.llm_summarizer import generate_summary
from output.output_formatter import format_output, format_analysis
from config import MIN_R2, MAX_MAPE, MIN_ROWS, MODEL_SAVE_PATH
from modeling.suitability_check import assess_modelability
from modeling.baseline_runner import run_baseline


class AnalyzerSelector:
    """Select and run the best analyzer for the dataset."""

    def _run_pipeline(self, df: pd.DataFrame, meta, target_column, datalake_dfs):
        from orchestrate_workflow import train_model, evaluate_model
        df_run, report = rank_and_merge(df, meta)
        if target_column not in df_run.columns:
            logging.warning("Target column %s not in dataset after merge", target_column)
            return None
        model = train_model(df_run.drop(columns=[target_column]), df_run[target_column], datalake_dfs)
        preds = model.predict(df_run.drop(columns=[target_column]))
        mae_val = mean_absolute_error(df_run[target_column], preds)
        em = evaluate_model(model, df_run.drop(columns=[target_column]), df_run[target_column])
        em["mae"] = mae_val
        return df_run, report, model, preds, em

    def analyze(
        self,
        df: pd.DataFrame,
        target_column: str,
        datalake_dfs: Optional[dict] = None,
        user_labels: Optional[dict[str, str]] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        import orchestrate_workflow as ow
        datalake_dfs = datalake_dfs or {}

        # ---------- Cheap suitability check before heavy modeling ----------
        suitability = assess_modelability(df, target_column)
        if not suitability["is_modelable"]:
            desc_analyzer = select_analyzer(df, preferred="DescriptiveAnalyzer")
            desc_res = desc_analyzer.run(df)
            return {
                "analysis_type": "descriptive",
                "stats": desc_res.get("artifacts"),
                "modeling_skipped": True,
                "failure_reason": suitability["reason"],
                "suitability": suitability,
            }
        if suitability["reason"].startswith("borderline"):
            desc_analyzer = select_analyzer(df, preferred="DescriptiveAnalyzer")
            desc_res = desc_analyzer.run(df)
            baseline_res = run_baseline(df, target_column, suitability["task"])
            return {
                "analysis_type": "baseline",
                "stats": desc_res.get("artifacts"),
                "baseline": baseline_res,
                "suitability": suitability,
            }

        run_id = run_id or str(uuid4())
        model_dir = Path(os.getenv("LOCAL_DATA_DIR", "local_data")) / "models" / run_id
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "best_model"

        column_meta = infer_column_meta(df)
        result = self._run_pipeline(df, column_meta, target_column, datalake_dfs)
        if result is None:
            return {"error": "Target column missing after merge"}
        df_run, merge_report, model, predictions, eval_metrics = result
        metrics_history: Dict[str, Any] = {"semantic_merge": eval_metrics}
        model_recommendations = {"semantic_merge": recommend_models_with_llm(df_run)}
        needs_role_review = False
        score_ok = eval_metrics["r2"] >= MIN_R2 and eval_metrics["mape"] <= MAX_MAPE
        if len(df_run) >= MIN_ROWS and not score_ok:
            needs_role_review = True
        if not score_ok:
            logging.warning("Model metrics below thresholds - running descriptive fallback")
            desc_analyzer = select_analyzer(df, preferred="DescriptiveAnalyzer")
            desc_res = desc_analyzer.run(df)
            fail_reason = (
                f"Model did not meet quality thresholds (r2={eval_metrics['r2']:.2f}, "
                f"mape={eval_metrics['mape']:.2f})."
            )
            summary_output = generate_summary(
                data_stats=metrics_history,
                model_results={},
                prompt=(
                    "Modeling failed due to low quality metrics. "
                    "Explain possible reasons and suggest next steps."
                ),
            )
            res = {
                "analysis_type": "descriptive",
                "stats": desc_res.get("artifacts"),
                "metrics": metrics_history,
                "modeling_failed": True,
                "failure_reason": fail_reason,
                **summary_output,
            }
            return res

        best_model = model
        best_predictions = predictions
        best_report = merge_report
        best_score = eval_metrics["mae"]
        best_df = df_run
        roles_dict = {m.name: m.role for m in column_meta}
        metadata_file = None
        if user_labels:
            tuned_meta = merge_user_labels(column_meta, user_labels)
            result2 = self._run_pipeline(df, tuned_meta, target_column, datalake_dfs)
            if result2 is not None:
                df2, report2, model2, preds2, metrics2 = result2
                metrics_history["user_merge"] = metrics2
                if metrics2["mae"] < best_score:
                    best_model = model2
                    best_score = metrics2["mae"]
                    best_predictions = preds2
                    best_report = report2
                    score_ok = metrics2["r2"] >= MIN_R2 and metrics2["mape"] <= MAX_MAPE
                    needs_role_review = False
                    roles_path = save_column_roles(df2, tuned_meta, identifier=run_id)
                    metadata_file = roles_path
                    best_df = df2
                    roles_dict = {m.name: m.role for m in tuned_meta}
                    column_meta = tuned_meta
        formatted_output = format_output(best_predictions)
        context = f"Model: {best_model}"
        best_answer = ask_follow_up_question("What is the accuracy of the model?", context)
        formatted_analysis = format_analysis(best_answer)
        res = {
            "output": formatted_output,
            "analysis": formatted_analysis,
            "metrics": metrics_history,
            "recommended_models": model_recommendations,
            "actions": [],
            "needs_role_review": needs_role_review,
            "run_id": run_id,
            "model_info": {
                "model_type": type(best_model).__name__,
                "merge_report": best_report,
            },
            "_model": best_model,
            "_preds": best_predictions,
            "_best_df": best_df,
            "_roles_dict": roles_dict,
            "_metadata_file": metadata_file,
            "_best_score": best_score,
        }
        return res

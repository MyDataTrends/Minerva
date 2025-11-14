from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import pandas as pd
import sys

from storage.local_backend import log_run_metadata
from output.output_formatter import format_output, format_analysis
from preprocessing.llm_summarizer import generate_summary
from preprocessing.llm_analyzer import ask_follow_up_question


class OutputGenerator:
    """Compile results and summaries."""

    def generate(
        self,
        model,
        predictions,
        metrics: Dict[str, Any],
        model_info: Dict[str, Any],
        run_id: str,
        data: pd.DataFrame,
        target_column: str,
        needs_role_review: bool,
        file_name: str,
    ) -> Dict[str, Any]:
        sys.modules.pop("streamlit", None)
        from orchestrate_workflow import save_model

        save_model(model, "best_model", run_id)
        output_dir = Path("output_files")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{run_id}_predictions.csv"
        pd.DataFrame({"prediction": predictions}).to_csv(output_path, index=False)
        formatted_output = format_output(predictions)
        context = f"Model: {model}"
        try:
            best_answer = ask_follow_up_question("What is the accuracy of the model?", context)
        except Exception:
            best_answer = "analysis"
        formatted_analysis = format_analysis(best_answer)
        log_run_metadata(
            run_id,
            True,
            needs_role_review,
            file_name=file_name,
            model_type=type(model).__name__,
            model_path=str(Path("models") / run_id / "best_model"),
            metadata_path=str(output_path),
            output_path=str(output_path),
        )
        result = {
            "output": formatted_output,
            "analysis": formatted_analysis,
            "metrics": metrics,
            "recommended_models": {},
            "actions": [],
            "needs_role_review": needs_role_review,
            "run_id": run_id,
            "model_info": model_info,
        }
        summary_prompt = (
            "Given these data statistics and model outputs, write a one-page business summary, "
            "list the top 3 next steps, and include a clear call to action."
        )
        summary_output = generate_summary(
            data_stats=metrics,
            model_results=model_info,
            prompt=summary_prompt,
        )
        result.update(summary_output)
        return result

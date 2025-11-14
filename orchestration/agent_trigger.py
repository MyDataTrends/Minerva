from __future__ import annotations

from typing import Dict, Any

from agents.action_agent import execute_actions
from preprocessing.save_meta import _hash_df


class AgentTrigger:
    """Trigger downstream agents based on results."""

    def trigger(
        self,
        result: Dict[str, Any],
        predictions,
        best_score: float,
        file_name: str,
        model,
        best_df,
        roles_dict,
    ) -> None:
        result["actions"] = execute_actions(
            {
                "predictions": predictions.tolist(),
                "mae": best_score,
                "file_name": file_name,
                "model_type": type(model).__name__,
                "metadata_file": "",
                "output_path": "models",
                "column_roles": roles_dict,
                "data_hash": _hash_df(best_df),
            }
        )

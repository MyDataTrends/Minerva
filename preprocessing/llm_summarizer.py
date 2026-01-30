import json
from llm_manager.llm_interface import get_llm_completion
from preprocessing.sanitize import redact
from config import (
    MISTRAL_MODEL_PATH,
    LLM_SUMMARY_TOKEN_BUDGET,
    LLM_TEMPERATURE,
)


def generate_summary(
    data_stats: dict,
    model_results: dict,
    prompt: str,
    dataset_meta: dict | None = None,
    column_meta: list | None = None,
):
    """Run the embedded Mistral shard to produce a plain-English summary.

    Parameters
    ----------
    data_stats : dict
        Dictionary of pipeline metrics (e.g. R2, MAPE, column types).
    model_results : dict
        Dictionary of model info (e.g. model name, feature importances).
    prompt : str
        Instructions for summary and recommendations.
    dataset_meta : dict, optional
        Rich dataset metadata from `parse_metadata` (columns, dtypes, summary).
    column_meta : list, optional
        List of ColumnMeta objects from `infer_column_meta` (roles, confidence).
    """

    # Format column metadata for LLM context
    column_context = []
    if column_meta:
        for cm in column_meta:
            entry = {"name": cm.name, "role": cm.role, "confidence": cm.confidence}
            if cm.description:
                entry["description"] = cm.description
            column_context.append(entry)

    # 1. Package inputs for the model
    payload = {
        "data_stats": data_stats,
        "model_results": model_results,
        "instructions": redact(prompt),
    }
    if dataset_meta:
        payload["dataset_metadata"] = dataset_meta
    if column_context:
        payload["column_roles"] = column_context

    # 2. Execute inference via unified LLM client
    response = get_llm_completion(
        json.dumps(payload),
        max_tokens=LLM_SUMMARY_TOKEN_BUDGET,
        temperature=LLM_TEMPERATURE,
    )

    insights = response.strip()
    details = {
        "data_stats": data_stats,
        "model_results": model_results,
    }

    # 4. Return the generated text and artifacts
    return {
        "summary": insights,
        "artifacts": details,
    }

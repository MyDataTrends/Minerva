import json
from preprocessing.llm_preprocessor import LLMClient
from preprocessing.sanitize import redact
from config import (
    MISTRAL_MODEL_PATH,
    LLM_SUMMARY_TOKEN_BUDGET,
    LLM_TEMPERATURE,
)


def generate_summary(data_stats: dict, model_results: dict, prompt: str):
    """Run the embedded Mistral shard to produce a plain-English summary.

    Parameters
    ----------
    data_stats : dict
        Dictionary of pipeline metrics (e.g. R2, MAPE, column types).
    model_results : dict
        Dictionary of model info (e.g. model name, feature importances).
    prompt : str
        Instructions for summary and recommendations.
    """

    # 1. Package inputs for the model
    payload = {
        "data_stats": data_stats,
        "model_results": model_results,
        "instructions": redact(prompt),
    }

    # 2. Execute inference via unified LLM client
    response = LLMClient().complete(
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

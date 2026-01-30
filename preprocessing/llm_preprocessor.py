"""Utilities for preprocessing data with a local LLM backend.

Now unified with the central llm_manager.
"""

import logging
import pandas as pd
from typing import Optional, Dict

from preprocessing.prompt_templates import generate_prompt
from preprocessing.metadata_parser import parse_metadata
from preprocessing.sanitize import redact
from config import (
    LLM_TOKEN_BUDGET,
    LLM_TEMPERATURE,
    LLM_MAX_INPUT_CHARS,
    LLM_NON_PRINTABLE_THRESHOLD,
)

# Import unified interface
from llm_manager.llm_interface import get_llm_completion

_logger = logging.getLogger(__name__)

def _guard_input(text: str) -> str:
    red = redact(text)
    if not red:
        return red
    total = len(red)
    non_printables = sum(
        1 for c in red if not c.isprintable() and c not in "\n\r\t"
    )
    if total and (non_printables / total) > LLM_NON_PRINTABLE_THRESHOLD:
        _logger.warning(
            "LLM input rejected due to non-printable ratio %.2f",
            non_printables / total,
        )
        raise ValueError("Input appears to be binary or high-entropy data")
    if total > LLM_MAX_INPUT_CHARS:
        truncated = red[:LLM_MAX_INPUT_CHARS] + "[TRUNCATED]"
        _logger.warning("LLM input truncated: %s", truncated)
        return truncated
    return red


def llm_completion(prompt: str, max_tokens: int = LLM_TOKEN_BUDGET) -> str:
    """
    Get completion from the central LLM manager.
    Wrapper for backward compatibility.
    """
    try:
        prompt = _guard_input(prompt)
    except ValueError:
        return "LLM input rejected"

    try:
        return get_llm_completion(prompt, max_tokens=max_tokens)
    except Exception as exc:
        return f"LLM error: {exc}"


def preprocess_data_with_llm(data, task: str = "text-classification"):
    """Deprecated: returning input unchanged."""
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(data)


def analyze_dataset_with_llm(df: pd.DataFrame) -> str:
    metadata = parse_metadata(df)
    prompt = generate_prompt(metadata)
    return llm_completion(prompt, max_tokens=LLM_TOKEN_BUDGET)


def preprocess_multiple_tasks(data, tasks=None):
    if tasks is None:
        tasks = ["text-classification", "sentiment-analysis"]
    results = {}
    for task in tasks:
        results[task] = llm_completion(f"Perform {task} on: {data}", max_tokens=LLM_TOKEN_BUDGET)
    return results


def handle_missing_values(data: pd.DataFrame, strategy: str = "mean"):
    if strategy == "mean":
        return data.fillna(data.mean())
    if strategy == "median":
        return data.fillna(data.median())
    if strategy == "mode":
        return data.fillna(data.mode().iloc[0])
    raise ValueError("Unsupported strategy")


def handle_outliers(data: pd.DataFrame, method: str = "IQR"):
    if method == "IQR":
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        return data[~((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr)))]
    if method == "Z-score":
        from scipy import stats
        return data[(abs(stats.zscore(data)) < 3)]
    raise ValueError("Unsupported method")


def encode_categorical_data(data: pd.Series) -> pd.Series:
    return data.astype("category").cat.codes


def preprocess_data_with_agents(data: pd.DataFrame) -> pd.DataFrame:
    data = handle_missing_values(data)
    data = handle_outliers(data)
    data = encode_categorical_data(data)
    return data


def score_similarity(uploaded_df: pd.DataFrame, datalake_dfs: dict) -> list:
    """Return datalake files sorted by similarity to the uploaded DataFrame."""
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        raise RuntimeError("scikit-learn is required for similarity scoring")
        
    uploaded_metadata = parse_metadata(uploaded_df)
    uploaded_vector = pd.DataFrame(uploaded_metadata["summary"]).values.flatten()
    uploaded_tags = set(str(col).lower() for col in uploaded_metadata["columns"])

    similarity_scores = {}
    for file_name, datalake_df in datalake_dfs.items():
        datalake_metadata = parse_metadata(datalake_df)
        datalake_vector = pd.DataFrame(datalake_metadata["summary"]).values.flatten()
        datalake_tags = set(str(col).lower() for col in datalake_metadata["columns"])

        # Vector similarity based on dataset summaries
        vec_score = cosine_similarity([uploaded_vector], [datalake_vector])[0][0]

        # Tag similarity based on overlapping column names
        union = uploaded_tags | datalake_tags
        tag_score = len(uploaded_tags & datalake_tags) / len(union) if union else 0.0

        # Combine the scores with equal weight
        similarity_scores[file_name] = (vec_score + tag_score) / 2

    sorted_files = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_files


def tag_dataset_with_llm(df: pd.DataFrame) -> str:
    metadata = parse_metadata(df)
    prompt = (
        "Tag the dataset columns and infer structure.\n"
        f"Columns: {metadata['columns']}\n"
        f"Types: {metadata['dtypes']}\n"
    )
    return llm_completion(prompt, max_tokens=LLM_TOKEN_BUDGET)


def recommend_models_with_llm(df: pd.DataFrame) -> str:
    metadata = parse_metadata(df)
    prompt = (
        "Recommend modeling approaches for the dataset.\n"
        f"Summary: {metadata['summary']}\n"
    )
    return llm_completion(prompt, max_tokens=LLM_TOKEN_BUDGET)

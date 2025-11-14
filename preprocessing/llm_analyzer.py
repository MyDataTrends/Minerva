"""Helpers that rely on the local LLM for dataset analysis."""
from preprocessing.llm_preprocessor import (
    analyze_dataset_with_llm,
    score_similarity,
    llm_completion,
)


def analyze_dataset(df):
    """Analyze ``df`` with the local LLM and return summary and artifacts."""

    insights = analyze_dataset_with_llm(df)
    details = {}
    return {
        "summary": insights,
        "artifacts": details,
    }


def score_dataset_similarity(uploaded_df, datalake_dfs):
    return score_similarity(uploaded_df, datalake_dfs)


def ask_follow_up_question(question: str, context: str) -> str:
    prompt = f"{context}\nQ: {question}\nA:"
    return llm_completion(prompt, max_tokens=32)


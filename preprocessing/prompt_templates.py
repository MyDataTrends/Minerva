"""Prompt templates used by the local LLM."""


def generate_prompt(metadata: dict) -> str:
    return (
        "Analyze the following dataset metadata and suggest preprocessing steps:\n"
        f"Columns: {metadata['columns']}\n"
        f"Data Types: {metadata['dtypes']}\n"
        f"Summary Statistics: {metadata['summary']}\n"
        "Based on the dataset structure, infer potential use cases and recommend preprocessing actions."
    )


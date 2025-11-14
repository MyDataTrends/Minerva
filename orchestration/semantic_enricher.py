from __future__ import annotations

import logging
from typing import Optional
import pandas as pd

from preprocessing.data_categorization import generate_tags, store_tags
from tagging.category_manager import save_user_category, save_fine_grained_tags
from preprocessing.llm_analyzer import score_dataset_similarity
from preprocessing.metadata_parser import parse_metadata
from preprocessing.data_cleaning import (
    clean_missing_values,
    normalize_text_columns,
    remove_duplicates,
    convert_to_datetime,
    remove_outliers,
    encode_categorical_columns,
)


class SemanticEnricher:
    """Join user data with public datasets and generate tags."""

    def enrich(
        self,
        df: pd.DataFrame,
        datalake_dfs: Optional[dict] = None,
        category: Optional[str] = None,
        file_name: str | None = None,
        user_id: str | None = None,
    ) -> pd.DataFrame:
        if df is None:
            return df
        datalake_dfs = datalake_dfs or {}
        metadata = parse_metadata(df)
        for column in metadata["columns"]:
            if metadata["dtypes"][column] == "object":
                df = normalize_text_columns(df, column)
                df = encode_categorical_columns(df, column)
            elif "date" in column.lower():
                df = convert_to_datetime(df, column)
            elif metadata["dtypes"][column] in ["int64", "float64"]:
                df = remove_outliers(df, column)
        df = clean_missing_values(df, strategy="fill", fill_value=0)
        df = remove_duplicates(df)
        score_dataset_similarity(df, datalake_dfs)
        tags = generate_tags(df)
        if file_name:
            store_tags(file_name, tags)
            if tags.get("llm_tags"):
                save_fine_grained_tags(file_name, tags["llm_tags"])
        if category and user_id:
            save_user_category(user_id, category)
        logging.info("Data pre-processing completed")
        return df

from __future__ import annotations

import logging
from typing import Any
from typing_extensions import override

import pandas as pd

from ..classification import run_classification
from ..interfaces import Analyzer
from utils.logging import log_metrics

_TARGET_HINTS = {"target", "label", "class", "region"}


class ClassificationAnalyzer(Analyzer):
    """Analyzer for classification tasks with categorical target."""

    def suitability_score(self, df: pd.DataFrame) -> float:
        cat_cols = df.select_dtypes(exclude="number").columns
        for col in cat_cols:
            if col.lower() in _TARGET_HINTS and df[col].nunique() <= 20:
                return 1.0
        for col in cat_cols:
            if df[col].nunique() <= 20:
                return 0.9
        # numeric columns with few unique values
        for col in df.select_dtypes(include="number").columns:
            if df[col].nunique() <= 5:
                return 0.6
        return 0.0

    def _guess_target(self, df: pd.DataFrame) -> str | None:
        for col in df.columns:
            if col.lower() in _TARGET_HINTS:
                return col
        cat_cols = df.select_dtypes(exclude="number").columns
        if cat_cols.any():
            return cat_cols[0]
        return None

    @override
    def run(self, df: pd.DataFrame, **kwargs: Any) -> dict:
        logger = logging.getLogger(__name__)
        logger.info("Running %s", self.__class__.__name__, extra={"rows": len(df)})
        target = self._guess_target(df)
        if target is None:
            raise ValueError("No categorical target column found")
        X = df.drop(columns=[target])
        y = df[target]
        artifacts = run_classification(X, y)
        summary = {"classes": list(set(y))}
        from .. import REGISTRY

        alternatives = [
            {"name": name, "score": cls().suitability_score(df)}
            for name, cls in REGISTRY.items()
            if name != self.__class__.__name__
        ]
        result = {
            "summary": summary if summary is not None else [],
            "artifacts": artifacts if artifacts is not None else {},
            "alternatives": alternatives,
        }

        metrics_dict = {
            "accuracy": artifacts.get("report", {}).get("accuracy")
        }
        log_metrics(self.__class__.__name__, metrics_dict)
        return result

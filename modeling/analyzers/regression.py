from __future__ import annotations

import logging
from typing import Any
from typing_extensions import override

import pandas as pd

from ..model_selector import run_regression
from ..interfaces import Analyzer
from utils.logging import log_metrics

_TARGET_HINTS = {"target", "y", "label", "sales"}


class RegressionAnalyzer(Analyzer):
    """Analyzer for numeric target regression problems."""

    def suitability_score(self, df: pd.DataFrame) -> float:
        numeric_cols = df.select_dtypes(include="number").columns
        if not numeric_cols.any():
            return 0.0
        # Heuristic: if a column name hints at being a target -> high
        for col in numeric_cols:
            if col.lower() in _TARGET_HINTS and df[col].nunique() > 5:
                return 1.0
        if len(numeric_cols) >= 2:
            if df[numeric_cols].nunique().max() > 10:
                return 0.6
        return 0.0

    def _guess_target(self, df: pd.DataFrame) -> str | None:
        for col in df.columns:
            if col.lower() in _TARGET_HINTS and pd.api.types.is_numeric_dtype(df[col]):
                return col
        numeric_cols = df.select_dtypes(include="number").columns
        if not numeric_cols.any():
            return None
        return df[numeric_cols].nunique().idxmax()

    @override
    def run(self, df: pd.DataFrame, **kwargs: Any) -> dict:
        logger = logging.getLogger(__name__)
        logger.info("Running %s", self.__class__.__name__, extra={"rows": len(df)})
        target = self._guess_target(df)
        if target is None:
            raise ValueError("No numeric target column found")
        X = df.drop(columns=[target])
        y = df[target]
        artifacts = run_regression(X, y)
        summary = {
            "mae": artifacts["results"]["metrics"]["mae"],
            "r2": artifacts["results"]["metrics"]["r2"],
        }
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

        metrics_dict = artifacts.get("results", {}).get("metrics", {})
        log_metrics(self.__class__.__name__, metrics_dict)
        return result

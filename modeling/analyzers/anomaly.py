from __future__ import annotations

import logging
from typing import Any
from typing_extensions import override

import pandas as pd

from ..anomaly import detect_anomalies
from ..interfaces import Analyzer
from utils.logging import log_metrics


class AnomalyAnalyzer(Analyzer):
    """Analyzer focusing solely on anomaly detection."""

    def suitability_score(self, df: pd.DataFrame) -> float:
        numeric_cols = df.select_dtypes(include="number").columns
        return 0.5 if len(numeric_cols) >= 1 else 0.0

    @override
    def run(self, df: pd.DataFrame, **kwargs: Any) -> dict:
        logger = logging.getLogger(__name__)
        logger.info("Running %s", self.__class__.__name__, extra={"rows": len(df)})
        artifacts = detect_anomalies(df.select_dtypes(include="number"))
        summary = {"n_outliers": artifacts["n_outliers"]}
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

        metrics_dict = {"n_outliers": summary.get("n_outliers", 0)}
        log_metrics(self.__class__.__name__, metrics_dict)
        return result

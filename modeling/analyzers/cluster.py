from __future__ import annotations

import logging
from typing import Any
from typing_extensions import override

import pandas as pd

from ..clustering import run_clustering
from ..interfaces import Analyzer
from utils.logging import log_metrics


class ClusterAnalyzer(Analyzer):
    """Analyzer for clustering when no clear target exists."""

    def suitability_score(self, df: pd.DataFrame) -> float:
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) > 3 and not df.select_dtypes(exclude="number").columns.any():
            return 0.8
        return 0.2 if len(numeric_cols) >= 2 else 0.0

    @override
    def run(self, df: pd.DataFrame, **kwargs: Any) -> dict:
        logger = logging.getLogger(__name__)
        logger.info("Running %s", self.__class__.__name__, extra={"rows": len(df)})
        k = min(4, max(2, len(df.select_dtypes(include="number").columns)))
        artifacts = run_clustering(df, k=k)
        summary = {"n_clusters": k}
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

        metrics_dict = {"n_clusters": k}
        log_metrics(self.__class__.__name__, metrics_dict)
        return result

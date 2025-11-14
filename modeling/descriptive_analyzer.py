from __future__ import annotations

import logging
from typing import Any
from typing_extensions import override

import pandas as pd

from descriptive.stats import generate_descriptives
from .interfaces import Analyzer
from utils.logging import log_metrics

class DescriptiveAnalyzer(Analyzer):
    """Analyzer providing basic descriptive statistics."""

    def suitability_score(self, df: pd.DataFrame) -> float:
        numeric_cols = df.select_dtypes(include="number").columns
        return 0.9 if len(numeric_cols) <= 3 else 0.3

    @override
    def run(self, df: pd.DataFrame, **kwargs: Any) -> dict:
        logger = logging.getLogger(__name__)
        logger.info("Running %s", self.__class__.__name__, extra={"rows": len(df)})
        artifacts = generate_descriptives(df)
        summary = {"columns": list(artifacts.index)} if not artifacts.empty else {}
        from . import REGISTRY

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
            "num_rows": len(df),
            "num_columns": len(df.columns),
        }
        log_metrics(self.__class__.__name__, metrics_dict)
        return result

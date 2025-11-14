from __future__ import annotations

import pandas as pd

from modeling import REGISTRY
from modeling.interfaces import Analyzer
from utils.metrics import suitability_score
from modeling.suitability_check import assess_modelability
from config.feature_flags import ALLOW_FULL_COMPARE_MODELS
from orchestration.analysis_router import route_analysis, LLMHint, MAX_ROWS_FULL


def select_analyzer(
    df: pd.DataFrame,
    policy: str = "auto",
    preferred: str | None = None,
) -> Analyzer:
    """Choose the best Analyzer based on ``suitability_score``.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    policy : str, default "auto"
        Currently only "auto" is supported.
    preferred : str, optional
        Explicit analyzer name to override automatic choice.
    """
    target = df.columns[-1]
    stats = assess_modelability(df, target)
    llm_hint = None
    if len(df) <= MAX_ROWS_FULL and ALLOW_FULL_COMPARE_MODELS:
        llm_hint = LLMHint()
    route = route_analysis(df, target, stats, llm_hint)

    # Always honor explicit preference when provided
    if preferred:
        analyzer = REGISTRY[preferred]()
        analyzer.analysis_route = route
        return analyzer

    if route == "no_model":
        analyzer = REGISTRY["DescriptiveAnalyzer"]()
        analyzer.analysis_route = route
        return analyzer

    dataset_score = suitability_score(df)
    best_name = None
    best_score = -1.0
    for name, cls in REGISTRY.items():
        score = cls().suitability_score(df) * dataset_score
        if score > best_score:
            best_name = name
            best_score = score
    if best_name is None:
        raise ValueError("No suitable analyzer found")
    analyzer = REGISTRY[best_name]()
    analyzer.analysis_route = route
    return analyzer

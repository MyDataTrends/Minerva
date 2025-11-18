"""Analyzer selection logic for choosing appropriate ML models."""

import pandas as pd
from typing import Optional, Any
from modeling.regression_analyzer import RegressionAnalyzer
from modeling.classification_analyzer import ClassificationAnalyzer
from modeling.cluster_analyzer import ClusterAnalyzer
from modeling.anomaly_analyzer import AnomalyAnalyzer
from modeling.descriptive_analyzer import DescriptiveAnalyzer
from modeling import REGISTRY
from utils.metrics import suitability_score as dataset_score


def select_analyzer(df: pd.DataFrame, preferred: Optional[str] = None) -> Any:
    """
    Select the best analyzer for the given dataset.
    
    Args:
        df: Input DataFrame
        preferred: Optional preferred analyzer class name
        
    Returns:
        Analyzer instance
    """
    if preferred:
        for analyzer_class in REGISTRY:
            if analyzer_class.__name__ == preferred:
                return analyzer_class()
    
    # Auto-select based on dataset characteristics
    best_analyzer = None
    best_score = -1
    
    for analyzer_class in REGISTRY:
        analyzer = analyzer_class()
        score = dataset_score(df) * analyzer.suitability_score(df)
        if score > best_score:
            best_score = score
            best_analyzer = analyzer
    
    return best_analyzer

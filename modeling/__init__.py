"""Modeling utilities and analyzer registry."""

from .analyzers.regression import RegressionAnalyzer
from .analyzers.classification import ClassificationAnalyzer
from .analyzers.cluster import ClusterAnalyzer
from .analyzers.anomaly import AnomalyAnalyzer
from .analyzers.descriptive import DescriptiveAnalyzer

REGISTRY = {
    "RegressionAnalyzer": RegressionAnalyzer,
    "ClassificationAnalyzer": ClassificationAnalyzer,
    "ClusterAnalyzer": ClusterAnalyzer,
    "AnomalyAnalyzer": AnomalyAnalyzer,
    "DescriptiveAnalyzer": DescriptiveAnalyzer,
}

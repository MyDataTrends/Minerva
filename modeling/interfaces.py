from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd

class Analyzer(ABC):
    """Common contract for every modelling / analysis strategy."""

    @abstractmethod
    def suitability_score(self, df: pd.DataFrame) -> float:
        """Return 0-1 score indicating how appropriate this analyzer is for *df*."""
        ...

    @abstractmethod
    def run(self, df: pd.DataFrame, **kwargs) -> dict:
        """Execute analysis and return summary and artifacts.

        Must return ``{'summary': [...], 'artifacts': {...}}``.
        """
        pass

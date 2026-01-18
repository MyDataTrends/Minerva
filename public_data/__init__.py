"""
Public Data Module - Connectors for external data sources.

Provides a unified interface for fetching and enriching data from:
- FRED (Federal Reserve Economic Data)
- Census Bureau
- World Bank
- Stock/financial APIs
- Housing market data
- Consumer spending data
"""
from public_data.connectors.base import DataConnector, DataSeries, EnrichmentSuggestion
from public_data.registry import get_all_connectors, get_connector, register_connector

__all__ = [
    "DataConnector",
    "DataSeries", 
    "EnrichmentSuggestion",
    "get_all_connectors",
    "get_connector",
    "register_connector",
]

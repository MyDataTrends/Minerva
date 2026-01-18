"""
Connector Registry - Manages and discovers data connectors.

Provides functions to register, discover, and retrieve connectors.
"""
from typing import Dict, List, Optional, Type
import logging

from public_data.connectors.base import DataConnector

logger = logging.getLogger(__name__)

# Global registry of connectors
_CONNECTORS: Dict[str, DataConnector] = {}


def register_connector(connector: DataConnector) -> None:
    """
    Register a connector instance.
    
    Args:
        connector: DataConnector instance to register
    """
    _CONNECTORS[connector.id] = connector
    logger.info(f"Registered connector: {connector.id} ({connector.name})")


def get_connector(connector_id: str) -> Optional[DataConnector]:
    """
    Get a connector by ID.
    
    Args:
        connector_id: ID of the connector
        
    Returns:
        DataConnector instance or None if not found
    """
    return _CONNECTORS.get(connector_id)


def get_all_connectors() -> List[DataConnector]:
    """
    Get all registered connectors.
    
    Returns:
        List of all registered DataConnector instances
    """
    return list(_CONNECTORS.values())


def discover_connectors() -> List[str]:
    """
    Discover and auto-register available connectors.
    
    Imports all connector modules and registers their instances.
    
    Returns:
        List of registered connector IDs
    """
    registered = []
    
    # Try to import and register FRED connector
    try:
        from public_data.connectors.fred import FREDConnector
        if "fred" not in _CONNECTORS:
            register_connector(FREDConnector())
            registered.append("fred")
    except ImportError as e:
        logger.debug(f"FRED connector not available: {e}")
    
    # Try to import and register Census connector
    try:
        from public_data.connectors.census import CensusConnector
        if "census" not in _CONNECTORS:
            register_connector(CensusConnector())
            registered.append("census")
    except ImportError as e:
        logger.debug(f"Census connector not available: {e}")
    
    # Try to import and register World Bank connector
    try:
        from public_data.connectors.world_bank import WorldBankConnector
        if "world_bank" not in _CONNECTORS:
            register_connector(WorldBankConnector())
            registered.append("world_bank")
    except ImportError as e:
        logger.debug(f"World Bank connector not available: {e}")
    
    # Try to import and register Alpha Vantage connector
    try:
        from public_data.connectors.alpha_vantage import AlphaVantageConnector
        if "alpha_vantage" not in _CONNECTORS:
            register_connector(AlphaVantageConnector())
            registered.append("alpha_vantage")
    except ImportError as e:
        logger.debug(f"Alpha Vantage connector not available: {e}")
    
    # Try to import and register local datasets connector
    try:
        from public_data.connectors.local_datasets import LocalDatasetsConnector
        if "local" not in _CONNECTORS:
            register_connector(LocalDatasetsConnector())
            registered.append("local")
    except ImportError as e:
        logger.debug(f"Local datasets connector not available: {e}")
    
    return registered



def get_suggestions_for_data(column_meta: list, df=None) -> list:
    """
    Get enrichment suggestions from all connectors for user's data.
    
    Args:
        column_meta: List of ColumnMeta objects
        df: Optional DataFrame for additional context
        
    Returns:
        List of EnrichmentSuggestion objects from all connectors
    """
    # Auto-discover if no connectors registered
    if not _CONNECTORS:
        discover_connectors()
    
    all_suggestions = []
    
    for connector in _CONNECTORS.values():
        try:
            suggestions = connector.suggest_enrichments(column_meta, df)
            all_suggestions.extend(suggestions)
        except Exception as e:
            logger.warning(f"Failed to get suggestions from {connector.id}: {e}")
    
    return all_suggestions


# Auto-discover connectors on module import
discover_connectors()

"""
Base classes for data connectors.

All connectors inherit from DataConnector and implement a common interface
for discovering, fetching, and caching external data.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataSeries:
    """Represents a single data series available from a connector."""
    id: str                          # Unique identifier (e.g., "GDP", "UNRATE")
    name: str                        # Human-readable name
    description: str                 # What the data represents
    frequency: str                   # "daily", "weekly", "monthly", "quarterly", "annual"
    start_date: Optional[str] = None # Earliest available date
    end_date: Optional[str] = None   # Latest available date
    columns: List[str] = field(default_factory=list)  # Columns this adds
    category: str = "other"          # Category for grouping
    match_roles: Set[str] = field(default_factory=set)  # Column roles this can match


@dataclass
class EnrichmentSuggestion:
    """A suggestion for enriching user data with external data."""
    connector_id: str                # ID of the connector
    series_id: str                   # ID of the data series
    name: str                        # Display name
    description: str                 # What this enrichment provides
    source: str                      # Data source name
    match_column: str                # User's column to match on
    match_type: str                  # Type of match (date, zip_code, state, etc.)
    columns_added: List[str]         # Columns that will be added
    category: str                    # Category (economic, demographics, etc.)
    confidence: float = 1.0          # How confident we are in this match (0-1)
    preview_data: Optional[pd.DataFrame] = None  # Sample data for preview


class DataConnector(ABC):
    """
    Abstract base class for all data connectors.
    
    Connectors provide a unified interface for:
    1. Discovering available data series
    2. Suggesting relevant enrichments based on user data
    3. Fetching data with caching
    4. Transforming data for merge compatibility
    """
    
    # Class-level attributes to override in subclasses
    id: str = "base"                          # Unique connector ID
    name: str = "Base Connector"              # Human-readable name
    description: str = ""                     # What data this connector provides
    supported_roles: Set[str] = set()         # Column roles this can enrich
    rate_limit: int = 60                      # Requests per minute
    requires_api_key: bool = False            # Whether API key is needed
    api_key_env_var: str = ""                 # Environment variable for API key
    base_url: str = ""                        # API base URL
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize connector with optional API key."""
        self._api_key = api_key
        self._cache: Dict[str, Any] = {}
        self._last_request_time: Optional[datetime] = None
    
    @property
    def api_key(self) -> Optional[str]:
        """Get API key from init or environment."""
        if self._api_key:
            return self._api_key
        if self.api_key_env_var:
            import os
            return os.getenv(self.api_key_env_var)
        return None
    
    @abstractmethod
    def get_available_series(self) -> List[DataSeries]:
        """
        Return list of available data series.
        
        This should return commonly used series, not necessarily all available data.
        """
        pass
    
    @abstractmethod
    def fetch_data(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **params
    ) -> pd.DataFrame:
        """
        Fetch data for a specific series.
        
        Args:
            series_id: ID of the series to fetch
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            **params: Additional connector-specific parameters
            
        Returns:
            DataFrame with the fetched data
        """
        pass
    
    def suggest_enrichments(
        self,
        column_meta: List[Any],
        df: Optional[pd.DataFrame] = None
    ) -> List[EnrichmentSuggestion]:
        """
        Suggest relevant data series based on user's columns.
        
        Args:
            column_meta: List of ColumnMeta objects describing user's data
            df: Optional DataFrame for more context
            
        Returns:
            List of EnrichmentSuggestion objects
        """
        suggestions = []
        user_roles = {m.role: m.name for m in column_meta if m.role != "unknown"}
        
        # Check which of our supported roles the user has
        matching_roles = self.supported_roles & set(user_roles.keys())
        
        if not matching_roles:
            return suggestions
        
        # Get available series and suggest matches
        for series in self.get_available_series():
            # Check if this series can match user's roles
            series_match_roles = series.match_roles or self.supported_roles
            shared_roles = series_match_roles & set(user_roles.keys())
            
            if shared_roles:
                # Pick the best matching role
                match_role = list(shared_roles)[0]
                match_column = user_roles[match_role]
                
                suggestions.append(EnrichmentSuggestion(
                    connector_id=self.id,
                    series_id=series.id,
                    name=f"{self.name}: {series.name}",
                    description=series.description,
                    source=self.name,
                    match_column=match_column,
                    match_type=match_role,
                    columns_added=series.columns,
                    category=series.category,
                ))
        
        return suggestions
    
    def _cache_key(self, series_id: str, **params) -> str:
        """Generate cache key for a request."""
        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{self.id}:{series_id}:{param_str}"
    
    def _get_cached(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached data if available and not expired."""
        if key in self._cache:
            data, timestamp = self._cache[key]
            # Cache for 1 hour
            if (datetime.now() - timestamp).seconds < 3600:
                logger.debug(f"Cache hit for {key}")
                return data
        return None
    
    def _set_cached(self, key: str, data: pd.DataFrame) -> None:
        """Cache data with timestamp."""
        self._cache[key] = (data, datetime.now())
    
    def fetch_with_cache(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **params
    ) -> pd.DataFrame:
        """
        Fetch data with caching.
        
        Checks cache first, fetches from API if not cached.
        """
        cache_key = self._cache_key(
            series_id, 
            start_date=start_date or "",
            end_date=end_date or "",
            **params
        )
        
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        data = self.fetch_data(series_id, start_date, end_date, **params)
        self._set_cached(cache_key, data)
        return data
    
    def test_connection(self) -> bool:
        """
        Test if the connector can reach its data source.
        
        Returns True if connection successful, False otherwise.
        """
        try:
            series = self.get_available_series()
            return len(series) > 0
        except Exception as e:
            logger.error(f"Connection test failed for {self.id}: {e}")
            return False

"""
Local Datasets Connector - Wraps existing datasets in the datasets/ folder.

Provides access to pre-downloaded datasets for enrichment without
needing external API calls.
"""
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from public_data.connectors.base import DataConnector, DataSeries

logger = logging.getLogger(__name__)


class LocalDatasetsConnector(DataConnector):
    """
    Connector for local pre-downloaded datasets.
    
    Wraps the existing datasets/ folder to provide consistent
    access through the connector interface.
    """
    
    id = "local"
    name = "Local Datasets"
    description = "Pre-downloaded datasets for offline enrichment"
    supported_roles = {"zip_code", "state", "date", "fips_code", "city"}
    rate_limit = 1000  # No real limit for local files
    requires_api_key = False
    
    # Default datasets directory
    _datasets_dir: Path = Path(__file__).resolve().parents[2] / "datasets"
    
    # Known local datasets and their metadata
    KNOWN_DATASETS = {
        "census_zip_income.csv": {
            "name": "Census ZIP Income",
            "description": "Median income by ZIP code",
            "match_roles": {"zip_code"},
            "category": "demographics",
            "columns": ["median_income"],
        },
        "zip_to_fips.csv": {
            "name": "ZIP to FIPS Mapping",
            "description": "Maps ZIP codes to FIPS county codes",
            "match_roles": {"zip_code"},
            "category": "geography",
            "columns": ["fips_code", "county"],
        },
        "us_states.csv": {
            "name": "US States",
            "description": "US state codes and names",
            "match_roles": {"state"},
            "category": "geography",
            "columns": ["state_name", "region"],
        },
        "holidays_events.csv": {
            "name": "Holidays and Events",
            "description": "US holidays and events calendar",
            "match_roles": {"date"},
            "category": "calendar",
            "columns": ["holiday_type", "is_holiday"],
        },
        "oil.csv": {
            "name": "Oil Prices",
            "description": "Historical oil prices",
            "match_roles": {"date"},
            "category": "economic",
            "columns": ["oil_price"],
        },
        "stores.csv": {
            "name": "Store Master",
            "description": "Retail store information",
            "match_roles": {"city", "state"},
            "category": "retail",
            "columns": ["store_type", "cluster"],
        },
    }
    
    def __init__(self, datasets_dir: Optional[Path] = None):
        super().__init__()
        if datasets_dir:
            self._datasets_dir = Path(datasets_dir)
    
    def get_available_series(self) -> List[DataSeries]:
        """Return list of available local datasets."""
        series_list = []
        
        # Check which known datasets actually exist
        for filename, info in self.KNOWN_DATASETS.items():
            filepath = self._datasets_dir / filename
            if filepath.exists():
                series_list.append(DataSeries(
                    id=filename,
                    name=info["name"],
                    description=info["description"],
                    frequency="static",
                    category=info.get("category", "other"),
                    columns=info.get("columns", []),
                    match_roles=set(info.get("match_roles", [])),
                ))
        
        # Also scan for any additional CSV/JSON files
        for filepath in self._datasets_dir.glob("*.csv"):
            if filepath.name not in self.KNOWN_DATASETS:
                series_list.append(DataSeries(
                    id=filepath.name,
                    name=filepath.stem.replace("_", " ").title(),
                    description=f"Local dataset: {filepath.name}",
                    frequency="static",
                    category="other",
                    columns=[],
                    match_roles=set(),
                ))
        
        return series_list
    
    def fetch_data(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **params
    ) -> pd.DataFrame:
        """
        Load a local dataset.
        
        Args:
            series_id: Filename of the dataset
            start_date: Not used for local files
            end_date: Not used for local files
            
        Returns:
            DataFrame with the dataset contents
        """
        filepath = self._datasets_dir / series_id
        
        if not filepath.exists():
            logger.error(f"Local dataset not found: {filepath}")
            return pd.DataFrame()
        
        try:
            if series_id.endswith(".csv"):
                return pd.read_csv(filepath)
            elif series_id.endswith(".json"):
                return pd.read_json(filepath)
            elif series_id.endswith((".xls", ".xlsx")):
                return pd.read_excel(filepath)
            else:
                logger.warning(f"Unknown file format: {series_id}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to load local dataset {series_id}: {e}")
            return pd.DataFrame()
    
    def list_files(self) -> List[str]:
        """List all available dataset files."""
        files = []
        for ext in ["*.csv", "*.json", "*.xlsx", "*.xls"]:
            files.extend([f.name for f in self._datasets_dir.glob(ext)])
        return sorted(files)

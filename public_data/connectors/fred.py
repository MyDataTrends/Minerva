"""
FRED Connector - Federal Reserve Economic Data.

Provides access to economic indicators like:
- GDP and growth rates
- Unemployment rates
- Inflation/CPI
- Interest rates
- Consumer sentiment

API Documentation: https://fred.stlouisfed.org/docs/api/
Free tier: Unlimited requests (120/minute rate limit)
"""
import os
import logging
from typing import List, Optional, Set
from datetime import datetime

import pandas as pd
import requests

from public_data.connectors.base import DataConnector, DataSeries

logger = logging.getLogger(__name__)


class FREDConnector(DataConnector):
    """
    Connector for Federal Reserve Economic Data (FRED).
    
    FRED provides free access to hundreds of thousands of economic data series
    from various government and institutional sources.
    """
    
    id = "fred"
    name = "FRED Economic Data"
    description = "Federal Reserve Economic Data - GDP, employment, inflation, and more"
    supported_roles = {"date", "datetime"}
    rate_limit = 120  # Requests per minute
    requires_api_key = True
    api_key_env_var = "FRED_API_KEY"
    base_url = "https://api.stlouisfed.org/fred"
    
    # Popular FRED series for easy access
    POPULAR_SERIES = {
        # GDP & Growth
        "GDP": {
            "name": "Gross Domestic Product",
            "description": "Quarterly GDP in billions of dollars",
            "frequency": "quarterly",
            "category": "economic",
        },
        "GDPC1": {
            "name": "Real GDP",
            "description": "Real GDP in chained 2017 dollars",
            "frequency": "quarterly",
            "category": "economic",
        },
        "A191RL1Q225SBEA": {
            "name": "Real GDP Growth Rate",
            "description": "Percent change from preceding period",
            "frequency": "quarterly",
            "category": "economic",
        },
        
        # Employment
        "UNRATE": {
            "name": "Unemployment Rate",
            "description": "Civilian unemployment rate, seasonally adjusted",
            "frequency": "monthly",
            "category": "economic",
        },
        "PAYEMS": {
            "name": "Total Nonfarm Payrolls",
            "description": "All employees, thousands, seasonally adjusted",
            "frequency": "monthly",
            "category": "economic",
        },
        "CIVPART": {
            "name": "Labor Force Participation Rate",
            "description": "Percent of population in labor force",
            "frequency": "monthly",
            "category": "economic",
        },
        
        # Inflation & Prices
        "CPIAUCSL": {
            "name": "Consumer Price Index",
            "description": "CPI for all urban consumers, all items",
            "frequency": "monthly",
            "category": "economic",
        },
        "CPILFESL": {
            "name": "Core CPI",
            "description": "CPI excluding food and energy",
            "frequency": "monthly",
            "category": "economic",
        },
        "PCEPI": {
            "name": "PCE Price Index",
            "description": "Personal consumption expenditures price index",
            "frequency": "monthly",
            "category": "economic",
        },
        
        # Interest Rates
        "DFF": {
            "name": "Federal Funds Rate",
            "description": "Effective federal funds rate",
            "frequency": "daily",
            "category": "financial",
        },
        "DGS10": {
            "name": "10-Year Treasury Rate",
            "description": "10-year Treasury constant maturity rate",
            "frequency": "daily",
            "category": "financial",
        },
        "MORTGAGE30US": {
            "name": "30-Year Mortgage Rate",
            "description": "30-year fixed rate mortgage average",
            "frequency": "weekly",
            "category": "housing",
        },
        
        # Housing
        "MSPUS": {
            "name": "Median Home Sale Price",
            "description": "Median sales price of houses sold in US",
            "frequency": "quarterly",
            "category": "housing",
        },
        "CSUSHPINSA": {
            "name": "Case-Shiller Home Price Index",
            "description": "S&P/Case-Shiller US National Home Price Index",
            "frequency": "monthly",
            "category": "housing",
        },
        "HOUST": {
            "name": "Housing Starts",
            "description": "New privately-owned housing units started",
            "frequency": "monthly",
            "category": "housing",
        },
        
        # Consumer
        "UMCSENT": {
            "name": "Consumer Sentiment",
            "description": "University of Michigan Consumer Sentiment Index",
            "frequency": "monthly",
            "category": "consumer",
        },
        "RSXFS": {
            "name": "Retail Sales",
            "description": "Advance retail sales, excluding food services",
            "frequency": "monthly",
            "category": "consumer",
        },
        "PCE": {
            "name": "Personal Consumption Expenditures",
            "description": "Total personal consumption expenditures",
            "frequency": "monthly",
            "category": "consumer",
        },
        "DSPIC96": {
            "name": "Real Disposable Personal Income",
            "description": "Real disposable personal income",
            "frequency": "monthly",
            "category": "consumer",
        },
    }
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        if not self.api_key:
            logger.warning(
                "FRED API key not set. Set FRED_API_KEY environment variable. "
                "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
            )
    
    def get_available_series(self) -> List[DataSeries]:
        """Return list of commonly used FRED series."""
        series_list = []
        
        for series_id, info in self.POPULAR_SERIES.items():
            series_list.append(DataSeries(
                id=series_id,
                name=info["name"],
                description=info["description"],
                frequency=info["frequency"],
                category=info.get("category", "economic"),
                columns=[series_id.lower()],
                match_roles={"date", "datetime"},
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
        Fetch data for a FRED series.
        
        Args:
            series_id: FRED series ID (e.g., "GDP", "UNRATE")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with 'date' and series value columns
        """
        if not self.api_key:
            logger.error("FRED API key required. Set FRED_API_KEY environment variable.")
            return pd.DataFrame()
        
        url = f"{self.base_url}/series/observations"
        
        request_params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        
        if start_date:
            request_params["observation_start"] = start_date
        if end_date:
            request_params["observation_end"] = end_date
        
        try:
            response = requests.get(url, params=request_params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "observations" not in data:
                logger.warning(f"No observations in FRED response for {series_id}")
                return pd.DataFrame()
            
            observations = data["observations"]
            
            df = pd.DataFrame(observations)
            
            if df.empty:
                return df
            
            # Clean up the data
            df = df[["date", "value"]].copy()
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            
            # Rename value column to series ID
            df = df.rename(columns={"value": series_id.lower()})
            
            return df
            
        except requests.RequestException as e:
            logger.error(f"FRED API request failed: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error processing FRED data: {e}")
            return pd.DataFrame()
    
    def fetch_multiple(
        self,
        series_ids: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch multiple series and merge them on date.
        
        Args:
            series_ids: List of FRED series IDs
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with date column and all requested series
        """
        dfs = []
        
        for series_id in series_ids:
            df = self.fetch_with_cache(series_id, start_date, end_date)
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        # Merge all on date
        result = dfs[0]
        for df in dfs[1:]:
            result = result.merge(df, on="date", how="outer")
        
        return result.sort_values("date")
    
    def search_series(self, search_text: str, limit: int = 10) -> List[dict]:
        """
        Search for FRED series by keyword.
        
        Args:
            search_text: Search query
            limit: Maximum results to return
            
        Returns:
            List of matching series info dicts
        """
        if not self.api_key:
            return []
        
        url = f"{self.base_url}/series/search"
        params = {
            "search_text": search_text,
            "api_key": self.api_key,
            "file_type": "json",
            "limit": limit,
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            return data.get("seriess", [])
            
        except Exception as e:
            logger.error(f"FRED search failed: {e}")
            return []

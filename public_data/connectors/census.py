"""
Census API Connector - US Census Bureau data.

Provides access to:
- Demographics by ZIP/FIPS
- Income and poverty statistics
- Housing characteristics
- Population counts

API Documentation: https://www.census.gov/data/developers.html
Free tier: 500 requests/day (or unlimited with API key)
"""
import os
import logging
from typing import List, Optional

import pandas as pd
import requests

from public_data.connectors.base import DataConnector, DataSeries

logger = logging.getLogger(__name__)


class CensusConnector(DataConnector):
    """
    Connector for US Census Bureau data.
    
    Provides access to American Community Survey (ACS) and other Census data.
    """
    
    id = "census"
    name = "US Census Bureau"
    description = "Demographics, income, housing, and population data by geography"
    supported_roles = {"zip_code", "fips_code", "state", "county"}
    rate_limit = 500  # Per day without key
    requires_api_key = False  # Optional but recommended
    api_key_env_var = "CENSUS_API_KEY"
    base_url = "https://api.census.gov/data"
    
    # Common ACS 5-year variables
    # See: https://api.census.gov/data/2022/acs/acs5/variables.html
    ACS_VARIABLES = {
        # Income
        "B19013_001E": {
            "name": "Median Household Income",
            "description": "Median household income in the past 12 months",
            "column": "median_household_income",
        },
        "B19301_001E": {
            "name": "Per Capita Income",
            "description": "Per capita income in the past 12 months",
            "column": "per_capita_income",
        },
        
        # Poverty
        "B17001_002E": {
            "name": "Population Below Poverty",
            "description": "Total population below poverty level",
            "column": "population_below_poverty",
        },
        
        # Population
        "B01003_001E": {
            "name": "Total Population",
            "description": "Total population count",
            "column": "total_population",
        },
        "B01002_001E": {
            "name": "Median Age",
            "description": "Median age of population",
            "column": "median_age",
        },
        
        # Housing
        "B25077_001E": {
            "name": "Median Home Value",
            "description": "Median value of owner-occupied housing units",
            "column": "median_home_value",
        },
        "B25064_001E": {
            "name": "Median Gross Rent",
            "description": "Median gross rent for renter-occupied units",
            "column": "median_rent",
        },
        "B25003_002E": {
            "name": "Owner Occupied Units",
            "description": "Number of owner-occupied housing units",
            "column": "owner_occupied_units",
        },
        "B25003_003E": {
            "name": "Renter Occupied Units",
            "description": "Number of renter-occupied housing units",
            "column": "renter_occupied_units",
        },
        
        # Education
        "B15003_022E": {
            "name": "Bachelor's Degree Holders",
            "description": "Population 25+ with bachelor's degree",
            "column": "bachelors_degree",
        },
        "B15003_023E": {
            "name": "Master's Degree Holders",
            "description": "Population 25+ with master's degree",
            "column": "masters_degree",
        },
        
        # Employment
        "B23025_004E": {
            "name": "Employed Population",
            "description": "Civilian employed population 16+",
            "column": "employed_population",
        },
        "B23025_005E": {
            "name": "Unemployed Population",
            "description": "Civilian unemployed population 16+",
            "column": "unemployed_population",
        },
    }
    
    # Predefined series groupings
    SERIES = {
        "demographics": {
            "name": "Demographics",
            "description": "Population, age, and household characteristics",
            "variables": ["B01003_001E", "B01002_001E"],
            "category": "demographics",
        },
        "income": {
            "name": "Income Statistics",
            "description": "Household and per capita income",
            "variables": ["B19013_001E", "B19301_001E", "B17001_002E"],
            "category": "demographics",
        },
        "housing": {
            "name": "Housing Characteristics",
            "description": "Home values, rent, and occupancy",
            "variables": ["B25077_001E", "B25064_001E", "B25003_002E", "B25003_003E"],
            "category": "housing",
        },
        "education": {
            "name": "Education Attainment",
            "description": "College degree holders",
            "variables": ["B15003_022E", "B15003_023E"],
            "category": "demographics",
        },
        "employment": {
            "name": "Employment Status",
            "description": "Employed and unemployed population",
            "variables": ["B23025_004E", "B23025_005E"],
            "category": "economic",
        },
    }
    
    def __init__(self, api_key: Optional[str] = None, year: int = 2022):
        super().__init__(api_key)
        self.year = year
    
    def get_available_series(self) -> List[DataSeries]:
        """Return list of available Census data series."""
        series_list = []
        
        for series_id, info in self.SERIES.items():
            columns = [
                self.ACS_VARIABLES[v]["column"] 
                for v in info["variables"] 
                if v in self.ACS_VARIABLES
            ]
            
            series_list.append(DataSeries(
                id=series_id,
                name=info["name"],
                description=info["description"],
                frequency="annual",
                category=info.get("category", "demographics"),
                columns=columns,
                match_roles={"zip_code", "fips_code", "state"},
            ))
        
        return series_list
    
    def fetch_data(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        geography: str = "state",
        state_fips: str = "*",
        **params
    ) -> pd.DataFrame:
        """
        Fetch Census data for a series.
        
        Args:
            series_id: ID from SERIES dict (e.g., "income", "housing")
            geography: "state", "county", or "zip code tabulation area"
            state_fips: FIPS code for state filter, or "*" for all
            
        Returns:
            DataFrame with geographic identifiers and data columns
        """
        if series_id not in self.SERIES:
            logger.error(f"Unknown series: {series_id}")
            return pd.DataFrame()
        
        variables = self.SERIES[series_id]["variables"]
        
        # Build API URL
        url = f"{self.base_url}/{self.year}/acs/acs5"
        
        # Build variable list
        var_list = ",".join(["NAME"] + variables)
        
        # Build geography clause
        # Note: Census API requires 'for' and 'in' as separate query params.
        # Embedding &in= inside the 'for' value causes requests to URL-encode
        # the '&', breaking the API call.
        if geography == "state":
            geo = f"state:{state_fips}"
        elif geography == "county":
            geo = "county:*"
        else:  # ZIP code tabulation area
            geo = "zip code tabulation area:*"
        
        request_params = {
            "get": var_list,
            "for": geo,
        }
        
        # Add 'in' clause for county-level queries
        if geography == "county":
            request_params["in"] = f"state:{state_fips}"
        
        if self.api_key:
            request_params["key"] = self.api_key
        
        try:
            response = requests.get(url, params=request_params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data or len(data) < 2:
                return pd.DataFrame()
            
            # First row is headers
            headers = data[0]
            rows = data[1:]
            
            df = pd.DataFrame(rows, columns=headers)
            
            # Rename variable columns to friendly names
            rename_map = {}
            for var in variables:
                if var in self.ACS_VARIABLES:
                    rename_map[var] = self.ACS_VARIABLES[var]["column"]
            
            df = df.rename(columns=rename_map)
            
            # Convert numeric columns
            for col in rename_map.values():
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            return df
            
        except requests.RequestException as e:
            logger.error(f"Census API request failed: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error processing Census data: {e}")
            return pd.DataFrame()
    
    def fetch_by_zip(self, variables: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch data for all ZIP Code Tabulation Areas.
        
        Args:
            variables: List of variable codes, or None for common ones
            
        Returns:
            DataFrame with ZIP codes and requested variables
        """
        if variables is None:
            variables = ["B19013_001E", "B01003_001E", "B25077_001E"]
        
        url = f"{self.base_url}/{self.year}/acs/acs5"
        var_list = ",".join(["NAME"] + variables)
        
        request_params = {
            "get": var_list,
            "for": "zip code tabulation area:*",
        }
        
        if self.api_key:
            request_params["key"] = self.api_key
        
        try:
            response = requests.get(url, params=request_params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            if not data or len(data) < 2:
                return pd.DataFrame()
            
            df = pd.DataFrame(data[1:], columns=data[0])
            
            # Rename
            rename_map = {"zip code tabulation area": "zip_code"}
            for var in variables:
                if var in self.ACS_VARIABLES:
                    rename_map[var] = self.ACS_VARIABLES[var]["column"]
            
            df = df.rename(columns=rename_map)
            
            # Convert numeric
            for var in variables:
                col = self.ACS_VARIABLES.get(var, {}).get("column", var)
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch ZIP data: {e}")
            return pd.DataFrame()

"""
World Bank Connector - Global development indicators.

Provides access to:
- GDP per capita by country
- Population and demographics
- Health indicators (life expectancy, mortality)
- Education metrics
- Infrastructure data

API Documentation: https://datahelpdesk.worldbank.org/knowledgebase/topics/125589
Free tier: Unlimited
"""
import logging
from typing import List, Optional

import pandas as pd
import requests

from public_data.connectors.base import DataConnector, DataSeries

logger = logging.getLogger(__name__)


class WorldBankConnector(DataConnector):
    """
    Connector for World Bank Open Data.
    
    Provides free access to global development indicators
    for all countries and regions.
    """
    
    id = "world_bank"
    name = "World Bank"
    description = "Global development indicators - GDP, population, health, education"
    supported_roles = {"country", "country_code", "date", "datetime"}
    rate_limit = 60
    requires_api_key = False
    base_url = "https://api.worldbank.org/v2"
    
    # Popular World Bank indicators
    POPULAR_INDICATORS = {
        # Economic
        "NY.GDP.PCAP.CD": {
            "name": "GDP per Capita",
            "description": "GDP per capita in current US dollars",
            "category": "economic",
        },
        "NY.GDP.MKTP.CD": {
            "name": "GDP Total",
            "description": "GDP in current US dollars",
            "category": "economic",
        },
        "NY.GDP.MKTP.KD.ZG": {
            "name": "GDP Growth Rate",
            "description": "Annual GDP growth rate percent",
            "category": "economic",
        },
        "FP.CPI.TOTL.ZG": {
            "name": "Inflation Rate",
            "description": "Consumer price inflation annual %",
            "category": "economic",
        },
        "SL.UEM.TOTL.ZS": {
            "name": "Unemployment Rate",
            "description": "Total unemployment % of labor force",
            "category": "economic",
        },
        
        # Population & Demographics
        "SP.POP.TOTL": {
            "name": "Total Population",
            "description": "Total population count",
            "category": "demographics",
        },
        "SP.URB.TOTL.IN.ZS": {
            "name": "Urban Population %",
            "description": "Urban population as % of total",
            "category": "demographics",
        },
        "SP.DYN.LE00.IN": {
            "name": "Life Expectancy",
            "description": "Life expectancy at birth, total years",
            "category": "demographics",
        },
        
        # Health
        "SH.XPD.CHEX.PC.CD": {
            "name": "Health Expenditure per Capita",
            "description": "Current health expenditure per capita (USD)",
            "category": "health",
        },
        "SH.DYN.MORT": {
            "name": "Infant Mortality Rate",
            "description": "Mortality rate, under-5 per 1,000 live births",
            "category": "health",
        },
        
        # Education
        "SE.ADT.LITR.ZS": {
            "name": "Literacy Rate",
            "description": "Adult literacy rate % ages 15+",
            "category": "education",
        },
        "SE.XPD.TOTL.GD.ZS": {
            "name": "Education Expenditure",
            "description": "Government expenditure on education % of GDP",
            "category": "education",
        },
        
        # Trade & Business
        "NE.TRD.GNFS.ZS": {
            "name": "Trade % of GDP",
            "description": "Trade (exports + imports) as % of GDP",
            "category": "economic",
        },
        "IC.BUS.EASE.XQ": {
            "name": "Ease of Doing Business",
            "description": "Ease of doing business ranking",
            "category": "economic",
        },
    }
    
    def get_available_series(self) -> List[DataSeries]:
        """Return list of commonly used World Bank indicators."""
        series_list = []
        
        for indicator_id, info in self.POPULAR_INDICATORS.items():
            series_list.append(DataSeries(
                id=indicator_id,
                name=info["name"],
                description=info["description"],
                frequency="annual",
                category=info.get("category", "global"),
                columns=[indicator_id.lower().replace(".", "_")],
                match_roles={"country", "country_code"},
            ))
        
        return series_list
    
    def fetch_data(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        countries: str = "all",
        **params
    ) -> pd.DataFrame:
        """
        Fetch data for a World Bank indicator.
        
        Args:
            series_id: World Bank indicator ID
            start_date: Start year (YYYY)
            end_date: End year (YYYY)
            countries: Country codes, comma-separated or "all"
            
        Returns:
            DataFrame with country, year, and indicator value
        """
        # Extract years from dates if provided
        start_year = start_date[:4] if start_date else "2000"
        end_year = end_date[:4] if end_date else "2023"
        
        url = f"{self.base_url}/country/{countries}/indicator/{series_id}"
        
        request_params = {
            "format": "json",
            "date": f"{start_year}:{end_year}",
            "per_page": 1000,
        }
        
        try:
            response = requests.get(url, params=request_params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Use recursive table finder to locate the actual data
            from preprocessing.data_cleaning import find_table_data
            
            observations = find_table_data(data)
            
            if not observations:
                logger.warning(f"No table data found in World Bank response for {series_id}")
                return pd.DataFrame()
            
            # Extract relevant fields
            records = []
            for obs in observations:
                if obs.get("value") is not None:
                    records.append({
                        "country_code": obs["countryiso3code"],
                        "country": obs["country"]["value"],
                        "year": int(obs["date"]),
                        series_id.lower().replace(".", "_"): obs["value"],
                    })
            
            df = pd.DataFrame(records)
            return df
            
        except requests.RequestException as e:
            logger.error(f"World Bank API request failed: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error processing World Bank data: {e}")
            return pd.DataFrame()
    
    def get_countries(self) -> pd.DataFrame:
        """Get list of all countries with their codes."""
        url = f"{self.base_url}/country"
        params = {"format": "json", "per_page": 300}
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data or len(data) < 2:
                return pd.DataFrame()
            
            countries = data[1]
            records = [
                {
                    "country_code": c["id"],
                    "country_name": c["name"],
                    "region": c.get("region", {}).get("value", ""),
                    "income_level": c.get("incomeLevel", {}).get("value", ""),
                }
                for c in countries
            ]
            
            return pd.DataFrame(records)
            
        except Exception as e:
            logger.error(f"Failed to fetch countries: {e}")
            return pd.DataFrame()

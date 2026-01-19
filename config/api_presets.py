"""
API Presets - Configuration for common data sources.

This file acts as a registry for known APIs, defining their base URLs,
authentication methods, and common endpoints/indicators.
"""

API_PRESETS = {
    "world_bank": {
        "name": "World Bank Open Data",
        "description": "Global development indicators (Poverty, Education, Health, Economy)",
        "base_url": "https://api.worldbank.org/v2",
        "type": "rest",
        "auth": {"type": "none"},  # Open API
        "documentation": "https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation",
        "endpoints": {
            "poverty_headcount": {
                "path": "country/all/indicator/SI.POV.DDAY",
                "params": {"format": "json", "per_page": 1000, "date": "2010:2023"},
                "description": "Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)"
            },
            "gdp_per_capita": {
                "path": "country/all/indicator/NY.GDP.PCAP.CD",
                "params": {"format": "json", "per_page": 1000, "date": "2010:2023"},
                "description": "GDP per capita (current US$)"
            },
            "gini_index": {
                "path": "country/all/indicator/SI.POV.GINI",
                "params": {"format": "json", "per_page": 1000, "date": "2010:2023"},
                "description": "Gini index (World Bank estimate) - Measure of inequality"
            },
            "co2_emissions": {
                "path": "country/all/indicator/EN.ATM.CO2E.PC",
                "params": {"format": "json", "per_page": 1000, "date": "2010:2023"},
                "description": "CO2 emissions (metric tons per capita)"
            },
            "literacy_rate": {
                "path": "country/all/indicator/SE.ADT.LITR.ZS",
                "params": {"format": "json", "per_page": 1000, "date": "2010:2023"},
                "description": "Literacy rate, adult total (% of people ages 15 and above)"
            }
        }
    },
    
    "fred": {
        "name": "FRED (St. Louis Fed)",
        "description": "Economic data, financial markets, and social indicators",
        "base_url": "https://api.stlouisfed.org/fred",
        "type": "rest",
        "auth": {
            "type": "api_key",
            "param_name": "api_key", # FRED passes key in query params
            "env_var": "FRED_API_KEY",
            "signup_url": "https://fred.stlouisfed.org/docs/api/api_key.html"
        },
        "documentation": "https://fred.stlouisfed.org/docs/api/fred/",
        "endpoints": {
            "gdp": {
                "path": "series/observations",
                "params": {"series_id": "GDP", "file_type": "json"},
                "description": "Gross Domestic Product"
            },
            "cpi": {
                "path": "series/observations",
                "params": {"series_id": "CPIAUCSL", "file_type": "json"},
                "description": "Consumer Price Index for All Urban Consumers"
            },
            "unemployment": {
                "path": "series/observations",
                "params": {"series_id": "UNRATE", "file_type": "json"},
                "description": "Unemployment Rate"
            },
            "income_inequality": {
                "path": "series/observations",
                "params": {"series_id": "GINIALLRH", "file_type": "json"},
                "description": "Income Inequality (GINI Ratio for US Households)"
            },
            "poverty_rate": {
                "path": "series/observations",
                "params": {"series_id": "PPAAUS00000A156NCEN", "file_type": "json"},
                "description": "Poverty Rate: All Ages"
            },
            "snap_benefits": {
                "path": "series/observations",
                "params": {"series_id": "B09010001E", "file_type": "json"}, 
                # Note: Specific series IDs might change, this is a placeholder generic logic
                "description": "SNAP Benefits Recipients (Example)"
            }
        }
    },
    
    "yfinance": {
        "name": "Yahoo Finance (via yfinance)",
        "description": "Stock market, crypto, and currency data",
        "base_url": "n/a", # Uses python library
        "type": "library", # Special type for yfinance wrapper
        "auth": {"type": "none"},
        "documentation": "https://pypi.org/project/yfinance/",
        "indicators": [
            "SPY", "BTC-USD", "ETH-USD", "VTI", "AGG"
        ]
    },
    
    "alphavantage": {
        "name": "AlphaVantage",
        "description": "Stocks, Crypto, Forex",
        "base_url": "https://www.alphavantage.co/query",
        "type": "rest",
        "auth": {
            "type": "api_key",
            "param_name": "apikey",
            "env_var": "ALPHAVANTAGE_API_KEY",
            "signup_url": "https://www.alphavantage.co/support/#api-key"
        },
        "documentation": "https://www.alphavantage.co/documentation/",
        "endpoints": {
             "time_series_daily": {
                 "path": "",
                 "params": {"function": "TIME_SERIES_DAILY", "datatype": "json"},
                 "description": "Daily stock time series"
             }
        }
    }
}

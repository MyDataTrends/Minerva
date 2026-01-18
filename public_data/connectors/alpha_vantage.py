"""
Alpha Vantage Connector - Stock market and financial data.

Provides access to:
- Real-time and historical stock prices
- Company fundamentals
- Technical indicators
- Forex and crypto data

API Documentation: https://www.alphavantage.co/documentation/
Free tier: 25 requests/day, 5 requests/minute
"""
import os
import logging
from typing import List, Optional

import pandas as pd
import requests

from public_data.connectors.base import DataConnector, DataSeries

logger = logging.getLogger(__name__)


class AlphaVantageConnector(DataConnector):
    """
    Connector for Alpha Vantage financial data.
    
    Provides access to stock prices, fundamentals, and market data.
    """
    
    id = "alpha_vantage"
    name = "Alpha Vantage"
    description = "Stock prices, company financials, forex, and crypto data"
    supported_roles = {"ticker", "symbol", "stock_symbol", "date", "datetime"}
    rate_limit = 5  # Per minute on free tier
    requires_api_key = True
    api_key_env_var = "ALPHA_VANTAGE_API_KEY"
    base_url = "https://www.alphavantage.co/query"
    
    # Available data types
    SERIES = {
        "daily_prices": {
            "name": "Daily Stock Prices",
            "description": "Open, high, low, close, volume for daily trading",
            "function": "TIME_SERIES_DAILY",
            "category": "financial",
            "columns": ["open", "high", "low", "close", "volume"],
        },
        "weekly_prices": {
            "name": "Weekly Stock Prices",
            "description": "Weekly OHLCV data",
            "function": "TIME_SERIES_WEEKLY",
            "category": "financial",
            "columns": ["open", "high", "low", "close", "volume"],
        },
        "monthly_prices": {
            "name": "Monthly Stock Prices",
            "description": "Monthly OHLCV data",
            "function": "TIME_SERIES_MONTHLY",
            "category": "financial",
            "columns": ["open", "high", "low", "close", "volume"],
        },
        "company_overview": {
            "name": "Company Overview",
            "description": "Company fundamentals - sector, market cap, P/E, etc.",
            "function": "OVERVIEW",
            "category": "financial",
            "columns": ["sector", "market_cap", "pe_ratio", "dividend_yield"],
        },
        "income_statement": {
            "name": "Income Statement",
            "description": "Annual and quarterly income statements",
            "function": "INCOME_STATEMENT",
            "category": "financial",
            "columns": ["revenue", "gross_profit", "net_income", "ebitda"],
        },
        "earnings": {
            "name": "Earnings",
            "description": "Annual and quarterly earnings data",
            "function": "EARNINGS",
            "category": "financial",
            "columns": ["reported_eps", "estimated_eps", "surprise_percentage"],
        },
    }
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        if not self.api_key:
            logger.warning(
                "Alpha Vantage API key not set. Set ALPHA_VANTAGE_API_KEY environment variable. "
                "Get a free key at: https://www.alphavantage.co/support/#api-key"
            )
    
    def get_available_series(self) -> List[DataSeries]:
        """Return list of available financial data series."""
        series_list = []
        
        for series_id, info in self.SERIES.items():
            series_list.append(DataSeries(
                id=series_id,
                name=info["name"],
                description=info["description"],
                frequency="daily" if "daily" in series_id else "varies",
                category=info.get("category", "financial"),
                columns=info.get("columns", []),
                match_roles={"ticker", "symbol", "stock_symbol"},
            ))
        
        return series_list
    
    def fetch_data(
        self,
        series_id: str,
        symbol: str = "AAPL",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **params
    ) -> pd.DataFrame:
        """
        Fetch financial data for a stock symbol.
        
        Args:
            series_id: ID from SERIES dict
            symbol: Stock ticker symbol (e.g., "AAPL", "MSFT")
            start_date: Filter results from this date
            end_date: Filter results to this date
            
        Returns:
            DataFrame with date and financial data columns
        """
        if not self.api_key:
            logger.error("Alpha Vantage API key required")
            return pd.DataFrame()
        
        if series_id not in self.SERIES:
            logger.error(f"Unknown series: {series_id}")
            return pd.DataFrame()
        
        series_info = self.SERIES[series_id]
        function = series_info["function"]
        
        request_params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
        }
        
        # Add outputsize for time series
        if "TIME_SERIES" in function:
            request_params["outputsize"] = "full"
        
        try:
            response = requests.get(self.base_url, params=request_params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return pd.DataFrame()
            
            if "Note" in data:  # Rate limit warning
                logger.warning(f"Alpha Vantage: {data['Note']}")
            
            # Parse based on function type
            if "TIME_SERIES" in function:
                return self._parse_time_series(data, start_date, end_date)
            elif function == "OVERVIEW":
                return self._parse_overview(data)
            elif function == "INCOME_STATEMENT":
                return self._parse_financials(data, "annualReports")
            elif function == "EARNINGS":
                return self._parse_earnings(data)
            else:
                logger.warning(f"Unsupported function: {function}")
                return pd.DataFrame()
                
        except requests.RequestException as e:
            logger.error(f"Alpha Vantage API request failed: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error processing Alpha Vantage data: {e}")
            return pd.DataFrame()
    
    def _parse_time_series(
        self, 
        data: dict, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Parse time series response."""
        # Find the time series key
        ts_key = None
        for key in data:
            if "Time Series" in key:
                ts_key = key
                break
        
        if not ts_key:
            return pd.DataFrame()
        
        records = []
        for date_str, values in data[ts_key].items():
            records.append({
                "date": date_str,
                "open": float(values.get("1. open", 0)),
                "high": float(values.get("2. high", 0)),
                "low": float(values.get("3. low", 0)),
                "close": float(values.get("4. close", 0)),
                "volume": int(values.get("5. volume", 0)),
            })
        
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        
        # Filter by date range
        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]
        
        return df
    
    def _parse_overview(self, data: dict) -> pd.DataFrame:
        """Parse company overview response."""
        if not data or "Symbol" not in data:
            return pd.DataFrame()
        
        record = {
            "symbol": data.get("Symbol"),
            "name": data.get("Name"),
            "sector": data.get("Sector"),
            "industry": data.get("Industry"),
            "market_cap": self._safe_float(data.get("MarketCapitalization")),
            "pe_ratio": self._safe_float(data.get("PERatio")),
            "peg_ratio": self._safe_float(data.get("PEGRatio")),
            "dividend_yield": self._safe_float(data.get("DividendYield")),
            "eps": self._safe_float(data.get("EPS")),
            "revenue_ttm": self._safe_float(data.get("RevenueTTM")),
            "profit_margin": self._safe_float(data.get("ProfitMargin")),
            "beta": self._safe_float(data.get("Beta")),
            "52_week_high": self._safe_float(data.get("52WeekHigh")),
            "52_week_low": self._safe_float(data.get("52WeekLow")),
        }
        
        return pd.DataFrame([record])
    
    def _parse_financials(self, data: dict, report_type: str) -> pd.DataFrame:
        """Parse financial statement response."""
        if report_type not in data:
            return pd.DataFrame()
        
        records = []
        for report in data[report_type][:4]:  # Last 4 periods
            records.append({
                "fiscal_date": report.get("fiscalDateEnding"),
                "revenue": self._safe_float(report.get("totalRevenue")),
                "gross_profit": self._safe_float(report.get("grossProfit")),
                "operating_income": self._safe_float(report.get("operatingIncome")),
                "net_income": self._safe_float(report.get("netIncome")),
                "ebitda": self._safe_float(report.get("ebitda")),
            })
        
        df = pd.DataFrame(records)
        df["fiscal_date"] = pd.to_datetime(df["fiscal_date"])
        return df.sort_values("fiscal_date")
    
    def _parse_earnings(self, data: dict) -> pd.DataFrame:
        """Parse earnings response."""
        if "annualEarnings" not in data:
            return pd.DataFrame()
        
        records = []
        for earning in data["annualEarnings"][:8]:
            records.append({
                "fiscal_date": earning.get("fiscalDateEnding"),
                "reported_eps": self._safe_float(earning.get("reportedEPS")),
            })
        
        df = pd.DataFrame(records)
        df["fiscal_date"] = pd.to_datetime(df["fiscal_date"])
        return df.sort_values("fiscal_date")
    
    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safely convert value to float."""
        if value is None or value == "None" or value == "-":
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def get_quote(self, symbol: str) -> dict:
        """Get real-time quote for a symbol."""
        if not self.api_key:
            return {}
        
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.api_key,
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            data = response.json()
            
            if "Global Quote" in data:
                quote = data["Global Quote"]
                return {
                    "symbol": quote.get("01. symbol"),
                    "price": self._safe_float(quote.get("05. price")),
                    "change": self._safe_float(quote.get("09. change")),
                    "change_percent": quote.get("10. change percent", "").replace("%", ""),
                    "volume": self._safe_float(quote.get("06. volume")),
                }
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get quote: {e}")
            return {}

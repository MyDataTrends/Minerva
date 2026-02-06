"""
API Registry - Curated database of public data APIs.

Provides metadata for:
- Authentication requirements and signup URLs
- Available endpoints and data types
- Intent matching keywords for automatic API selection

This registry enables the autonomous data agent to:
1. Match user queries to appropriate APIs
2. Guide users through API key setup
3. Route requests to the correct endpoints
"""
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class APIEndpoint:
    """Definition of an API endpoint."""
    path: str
    method: str = "GET"
    description: str = ""
    params: List[str] = field(default_factory=list)
    required_params: List[str] = field(default_factory=list)
    example: Optional[str] = None


@dataclass  
class APIDefinition:
    """Complete definition of an API."""
    id: str
    name: str
    description: str
    base_url: str
    
    # Authentication
    auth_type: str  # "none", "api_key", "oauth2", "bearer"
    auth_config: Dict[str, Any] = field(default_factory=dict)
    signup_url: Optional[str] = None
    free_tier: bool = True
    
    # Data types for intent matching
    data_types: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Endpoints
    endpoints: List[APIEndpoint] = field(default_factory=list)
    
    # Rate limiting
    rate_limit: Optional[int] = None  # Requests per minute
    
    # Documentation
    docs_url: Optional[str] = None
    
    # Status
    enabled: bool = True


# =============================================================================
# API Registry
# =============================================================================

API_REGISTRY: Dict[str, APIDefinition] = {}


def _register(api: APIDefinition) -> None:
    """Register an API definition."""
    API_REGISTRY[api.id] = api


# -----------------------------------------------------------------------------
# Economic & Financial Data
# -----------------------------------------------------------------------------

_register(APIDefinition(
    id="fred",
    name="Federal Reserve Economic Data (FRED)",
    description="US economic data: GDP, unemployment, inflation, interest rates, and more",
    base_url="https://api.stlouisfed.org/fred",
    auth_type="api_key",
    auth_config={
        "param_name": "api_key",
        "env_var": "FRED_API_KEY",
        "location": "query",  # query param vs header
    },
    signup_url="https://fred.stlouisfed.org/docs/api/api_key.html",
    free_tier=True,
    data_types=[
        "gdp", "unemployment", "inflation", "cpi", "interest_rates",
        "employment", "wages", "housing", "retail_sales", "industrial_production",
        "consumer_sentiment", "money_supply", "federal_funds_rate"
    ],
    keywords=[
        "economic", "economy", "fed", "federal reserve", "macro", "macroeconomic",
        "recession", "growth", "jobs", "jobless", "labor", "labour",
        "prices", "cost of living", "treasury", "bonds", "yields"
    ],
    endpoints=[
        APIEndpoint(
            path="/series/observations",
            description="Get observations for a specific series",
            params=["series_id", "observation_start", "observation_end", "units", "frequency"],
            required_params=["series_id"],
            example="/series/observations?series_id=GDP&observation_start=2020-01-01"
        ),
        APIEndpoint(
            path="/series/search",
            description="Search for series by keywords",
            params=["search_text", "search_type", "limit"],
            required_params=["search_text"],
        ),
    ],
    rate_limit=120,
    docs_url="https://fred.stlouisfed.org/docs/api/fred/",
))


_register(APIDefinition(
    id="world_bank",
    name="World Bank Open Data",
    description="Global development indicators: GDP, population, health, education across 200+ countries",
    base_url="https://api.worldbank.org/v2",
    auth_type="none",
    signup_url=None,
    free_tier=True,
    data_types=[
        "gdp", "gdp_per_capita", "population", "demographics", "health",
        "education", "poverty", "inequality", "infrastructure", "trade",
        "environment", "emissions", "life_expectancy", "mortality",
        "literacy", "unemployment", "inflation"
    ],
    keywords=[
        "world", "global", "international", "countries", "nations",
        "development", "developing", "emerging markets", "africa", "asia",
        "latin america", "europe", "comparison", "country comparison"
    ],
    endpoints=[
        APIEndpoint(
            path="/country/{countries}/indicator/{indicator}",
            description="Get indicator data for countries",
            params=["countries", "indicator", "date", "per_page", "format"],
            required_params=["countries", "indicator"],
            example="/country/USA;CHN/indicator/NY.GDP.MKTP.CD?format=json"
        ),
        APIEndpoint(
            path="/country",
            description="List all countries",
            params=["per_page", "format"],
        ),
    ],
    rate_limit=60,
    docs_url="https://datahelpdesk.worldbank.org/knowledgebase/topics/125589",
))


_register(APIDefinition(
    id="alpha_vantage",
    name="Alpha Vantage",
    description="Stock market data, forex, and cryptocurrency prices",
    base_url="https://www.alphavantage.co/query",
    auth_type="api_key",
    auth_config={
        "param_name": "apikey",
        "env_var": "ALPHA_VANTAGE_API_KEY",
        "location": "query",
    },
    signup_url="https://www.alphavantage.co/support/#api-key",
    free_tier=True,
    data_types=[
        "stocks", "stock_prices", "equities", "forex", "currency",
        "cryptocurrency", "crypto", "bitcoin", "ethereum", "commodities",
        "gold", "silver", "oil", "technical_indicators", "earnings"
    ],
    keywords=[
        "stock", "share", "market", "trading", "investment", "portfolio",
        "S&P", "dow", "nasdaq", "ticker", "symbol", "price", "quote",
        "forex", "fx", "exchange rate", "crypto", "bitcoin", "btc", "eth"
    ],
    endpoints=[
        APIEndpoint(
            path="",
            description="Query with function parameter",
            params=["function", "symbol", "interval", "outputsize"],
            required_params=["function"],
            example="?function=TIME_SERIES_DAILY&symbol=MSFT"
        ),
    ],
    rate_limit=5,  # Free tier is very limited
    docs_url="https://www.alphavantage.co/documentation/",
))


# -----------------------------------------------------------------------------
# Government & Census Data
# -----------------------------------------------------------------------------

_register(APIDefinition(
    id="census",
    name="US Census Bureau",
    description="US population, demographics, housing, and business statistics",
    base_url="https://api.census.gov/data",
    auth_type="api_key",
    auth_config={
        "param_name": "key",
        "env_var": "CENSUS_API_KEY",
        "location": "query",
    },
    signup_url="https://api.census.gov/data/key_signup.html",
    free_tier=True,
    data_types=[
        "population", "demographics", "housing", "income", "poverty",
        "employment", "business", "health_insurance", "education",
        "race", "ethnicity", "age", "gender", "household"
    ],
    keywords=[
        "census", "us population", "american", "united states",
        "demographics", "zip code", "county", "state", "city",
        "household", "families", "commute", "housing"
    ],
    endpoints=[
        APIEndpoint(
            path="/2020/acs/acs5",
            description="American Community Survey 5-year estimates",
            params=["get", "for", "in"],
            required_params=["get"],
            example="/2020/acs/acs5?get=NAME,B01001_001E&for=state:*"
        ),
    ],
    rate_limit=500,
    docs_url="https://www.census.gov/data/developers/guidance.html",
))


_register(APIDefinition(
    id="bls",
    name="Bureau of Labor Statistics",
    description="US employment, wages, prices, and productivity data",
    base_url="https://api.bls.gov/publicAPI/v2",
    auth_type="api_key",
    auth_config={
        "param_name": "registrationkey",
        "env_var": "BLS_API_KEY",
        "location": "query",
    },
    signup_url="https://data.bls.gov/registrationEngine/",
    free_tier=True,
    data_types=[
        "employment", "unemployment", "wages", "salaries", "compensation",
        "cpi", "consumer_prices", "producer_prices", "productivity",
        "occupations", "industries", "injuries", "work_stoppages"
    ],
    keywords=[
        "labor", "labour", "jobs", "employment", "wages", "salary",
        "occupation", "industry", "bls", "work", "worker"
    ],
    endpoints=[
        APIEndpoint(
            path="/timeseries/data",
            description="Get time series data",
            params=["seriesid", "startyear", "endyear"],
            required_params=["seriesid"],
        ),
    ],
    rate_limit=50,
    docs_url="https://www.bls.gov/developers/",
))


# -----------------------------------------------------------------------------
# Weather & Environment
# -----------------------------------------------------------------------------

_register(APIDefinition(
    id="openweathermap",
    name="OpenWeatherMap",
    description="Current weather, forecasts, and historical weather data",
    base_url="https://api.openweathermap.org/data/2.5",
    auth_type="api_key",
    auth_config={
        "param_name": "appid",
        "env_var": "OPENWEATHERMAP_API_KEY",
        "location": "query",
    },
    signup_url="https://openweathermap.org/api",
    free_tier=True,
    data_types=[
        "weather", "temperature", "humidity", "wind", "precipitation",
        "forecast", "historical_weather", "air_quality", "uv_index"
    ],
    keywords=[
        "weather", "temperature", "rain", "snow", "wind", "humidity",
        "forecast", "climate", "hot", "cold", "sunny", "cloudy"
    ],
    endpoints=[
        APIEndpoint(
            path="/weather",
            description="Current weather",
            params=["q", "lat", "lon", "units"],
            example="/weather?q=London&units=metric"
        ),
        APIEndpoint(
            path="/forecast",
            description="5-day forecast",
            params=["q", "lat", "lon", "units"],
        ),
    ],
    rate_limit=60,
    docs_url="https://openweathermap.org/api",
))


_register(APIDefinition(
    id="noaa",
    name="NOAA Climate Data",
    description="Historical climate and weather observations from NOAA",
    base_url="https://www.ncdc.noaa.gov/cdo-web/api/v2",
    auth_type="api_key",
    auth_config={
        "header_name": "token",
        "env_var": "NOAA_API_KEY",
        "location": "header",
    },
    signup_url="https://www.ncdc.noaa.gov/cdo-web/token",
    free_tier=True,
    data_types=[
        "climate", "weather_history", "temperature", "precipitation",
        "storms", "hurricanes", "drought", "snowfall"
    ],
    keywords=[
        "noaa", "climate", "historical weather", "storm", "hurricane",
        "drought", "long-term", "climate change", "trends"
    ],
    endpoints=[
        APIEndpoint(
            path="/data",
            description="Get climate observations",
            params=["datasetid", "locationid", "startdate", "enddate"],
            required_params=["datasetid"],
        ),
    ],
    rate_limit=5,
    docs_url="https://www.ncdc.noaa.gov/cdo-web/webservices/v2",
))


# -----------------------------------------------------------------------------
# News & Content
# -----------------------------------------------------------------------------

_register(APIDefinition(
    id="newsapi",
    name="NewsAPI",
    description="Headlines and articles from news sources worldwide",
    base_url="https://newsapi.org/v2",
    auth_type="api_key",
    auth_config={
        "header_name": "X-Api-Key",
        "env_var": "NEWSAPI_KEY",
        "location": "header",
    },
    signup_url="https://newsapi.org/register",
    free_tier=True,
    data_types=[
        "news", "headlines", "articles", "journalism", "media"
    ],
    keywords=[
        "news", "headlines", "articles", "breaking", "current events",
        "journalism", "media", "press", "reporting"
    ],
    endpoints=[
        APIEndpoint(
            path="/top-headlines",
            description="Top headlines",
            params=["country", "category", "q", "sources"],
        ),
        APIEndpoint(
            path="/everything",
            description="Search all articles",
            params=["q", "from", "to", "sortBy"],
            required_params=["q"],
        ),
    ],
    rate_limit=100,
    docs_url="https://newsapi.org/docs",
))


# -----------------------------------------------------------------------------
# Health Data
# -----------------------------------------------------------------------------

_register(APIDefinition(
    id="who",
    name="World Health Organization GHO",
    description="Global health statistics and indicators",
    base_url="https://ghoapi.azureedge.net/api",
    auth_type="none",
    signup_url=None,
    free_tier=True,
    data_types=[
        "health", "disease", "mortality", "life_expectancy", "vaccination",
        "hospital", "healthcare", "mental_health", "nutrition", "obesity"
    ],
    keywords=[
        "health", "disease", "who", "world health", "medical",
        "mortality", "death rate", "vaccination", "immunization",
        "epidemic", "pandemic", "covid", "malaria", "tuberculosis"
    ],
    endpoints=[
        APIEndpoint(
            path="/Indicator",
            description="Get indicator data",
            params=["$filter"],
        ),
        APIEndpoint(
            path="/{indicator}",
            description="Get specific indicator values",
            params=["indicator"],
            required_params=["indicator"],
        ),
    ],
    rate_limit=100,
    docs_url="https://www.who.int/data/gho/info/gho-odata-api",
))


# -----------------------------------------------------------------------------
# Geographic & Location Data
# -----------------------------------------------------------------------------

_register(APIDefinition(
    id="openstreetmap_nominatim",
    name="OpenStreetMap Nominatim",
    description="Geocoding and reverse geocoding (addresses to coordinates)",
    base_url="https://nominatim.openstreetmap.org",
    auth_type="none",
    signup_url=None,
    free_tier=True,
    data_types=[
        "geocoding", "addresses", "coordinates", "locations", "places"
    ],
    keywords=[
        "geocode", "address", "location", "coordinates", "lat", "lon",
        "latitude", "longitude", "map", "place", "city", "street"
    ],
    endpoints=[
        APIEndpoint(
            path="/search",
            description="Search for locations",
            params=["q", "format", "addressdetails", "limit"],
            required_params=["q"],
            example="/search?q=London&format=json"
        ),
        APIEndpoint(
            path="/reverse",
            description="Reverse geocode coordinates to address",
            params=["lat", "lon", "format"],
            required_params=["lat", "lon"],
        ),
    ],
    rate_limit=1,  # Very strict rate limit
    docs_url="https://nominatim.org/release-docs/develop/api/Overview/",
))


# =============================================================================
# Registry Functions
# =============================================================================

def get_api(api_id: str) -> Optional[APIDefinition]:
    """Get an API definition by ID."""
    return API_REGISTRY.get(api_id)


def get_all_apis() -> List[APIDefinition]:
    """Get all registered APIs."""
    return list(API_REGISTRY.values())


def search_apis_by_query(query: str) -> List[Dict[str, Any]]:
    """
    Search for APIs matching a natural language query.
    
    Returns list of matches with relevance scores.
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    results = []
    
    for api_id, api in API_REGISTRY.items():
        if not api.enabled:
            continue
        
        score = 0
        matched_keywords = []
        
        # Check data types
        for dt in api.data_types:
            if dt in query_lower:
                score += 10
                matched_keywords.append(dt)
        
        # Check keywords
        for kw in api.keywords:
            if kw in query_lower:
                score += 5
                matched_keywords.append(kw)
            # Partial word match
            elif any(word in kw or kw in word for word in query_words):
                score += 2
        
        # Check name and description
        if any(word in api.name.lower() for word in query_words):
            score += 3
        if any(word in api.description.lower() for word in query_words):
            score += 1
        
        if score > 0:
            results.append({
                "api_id": api_id,
                "name": api.name,
                "description": api.description,
                "score": score,
                "matched_keywords": matched_keywords[:5],
                "auth_required": api.auth_type != "none",
                "signup_url": api.signup_url,
                "free_tier": api.free_tier,
            })
    
    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    
    return results


def get_apis_requiring_auth() -> List[APIDefinition]:
    """Get all APIs that require authentication."""
    return [api for api in API_REGISTRY.values() if api.auth_type != "none"]


def get_auth_instructions(api_id: str) -> Optional[Dict[str, Any]]:
    """Get authentication instructions for an API."""
    api = get_api(api_id)
    if not api:
        return None
    
    if api.auth_type == "none":
        return {"auth_required": False}
    
    return {
        "auth_required": True,
        "auth_type": api.auth_type,
        "env_var": api.auth_config.get("env_var"),
        "signup_url": api.signup_url,
        "docs_url": api.docs_url,
        "instructions": f"""
To use {api.name}, you need an API key:

1. Visit: {api.signup_url}
2. Create a free account
3. Copy your API key
4. Either:
   - Set environment variable: {api.auth_config.get('env_var')}
   - Or store it securely in Minerva's credential manager
""".strip(),
    }

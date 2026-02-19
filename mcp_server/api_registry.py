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
    openapi_url: Optional[str] = None  # OpenAPI/Swagger spec URL for auto-generation
    
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
            path="/{year}/acs/acs5",
            description=(
                "American Community Survey 5-year estimates. "
                "Common 'get' variables: NAME, "
                "B01003_001E (total population), B19013_001E (median household income), "
                "B23025_004E (employed population), B23025_005E (unemployed population), "
                "B25077_001E (median home value), B25064_001E (median gross rent), "
                "B15003_022E (bachelor's degree), B15003_023E (master's degree). "
                "Geography via 'for' param: for=state:* (all states), for=county:* (all counties, "
                "requires in=state:*), for=zip code tabulation area:* (all zip codes). "
                "Use 'in' param for county queries: in=state:* for all states."
            ),
            params=["get", "for", "in", "year"],
            required_params=["get", "for"],
            example="/{year}/acs/acs5?get=NAME,B01003_001E&for=state:*"
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


# -----------------------------------------------------------------------------
# Cryptocurrency & Blockchain
# -----------------------------------------------------------------------------

_register(APIDefinition(
    id="coingecko",
    name="CoinGecko",
    description="Cryptocurrency prices, market data, and historical data for 10,000+ coins",
    base_url="https://api.coingecko.com/api/v3",
    auth_type="none",  # Free tier doesn't require key
    signup_url="https://www.coingecko.com/en/api",
    free_tier=True,
    data_types=[
        "cryptocurrency", "crypto_prices", "bitcoin", "ethereum", "altcoins",
        "market_cap", "trading_volume", "defi", "nft"
    ],
    keywords=[
        "crypto", "cryptocurrency", "bitcoin", "btc", "ethereum", "eth",
        "altcoin", "defi", "nft", "blockchain", "coin", "token", "market cap"
    ],
    endpoints=[
        APIEndpoint(
            path="/coins/markets",
            description="List coins with market data",
            params=["vs_currency", "ids", "order", "per_page", "page"],
            required_params=["vs_currency"],
            example="/coins/markets?vs_currency=usd&order=market_cap_desc"
        ),
        APIEndpoint(
            path="/coins/{id}/market_chart",
            description="Historical market data for a coin",
            params=["id", "vs_currency", "days"],
            required_params=["id", "vs_currency", "days"],
        ),
    ],
    rate_limit=30,
    docs_url="https://www.coingecko.com/en/api/documentation",
))


_register(APIDefinition(
    id="coinmarketcap",
    name="CoinMarketCap",
    description="Cryptocurrency market data, rankings, and metrics",
    base_url="https://pro-api.coinmarketcap.com/v1",
    auth_type="api_key",
    auth_config={
        "header_name": "X-CMC_PRO_API_KEY",
        "env_var": "COINMARKETCAP_API_KEY",
        "location": "header",
    },
    signup_url="https://coinmarketcap.com/api/",
    free_tier=True,
    data_types=[
        "cryptocurrency", "crypto_rankings", "market_cap", "crypto_quotes"
    ],
    keywords=[
        "coinmarketcap", "cmc", "crypto ranking", "top crypto", "market dominance"
    ],
    endpoints=[
        APIEndpoint(
            path="/cryptocurrency/listings/latest",
            description="Latest market data for all cryptocurrencies",
            params=["start", "limit", "convert", "sort"],
        ),
    ],
    rate_limit=30,
    docs_url="https://coinmarketcap.com/api/documentation/v1/",
))


# -----------------------------------------------------------------------------
# Sports Data
# -----------------------------------------------------------------------------

_register(APIDefinition(
    id="espn",
    name="ESPN API (Unofficial)",
    description="Sports scores, schedules, and standings for major leagues",
    base_url="https://site.api.espn.com/apis/site/v2",
    auth_type="none",
    signup_url=None,
    free_tier=True,
    data_types=[
        "sports", "scores", "standings", "schedules", "nfl", "nba", "mlb",
        "nhl", "soccer", "football", "basketball", "baseball"
    ],
    keywords=[
        "sports", "espn", "scores", "games", "nfl", "nba", "mlb", "nhl",
        "football", "basketball", "baseball", "hockey", "soccer", "team"
    ],
    endpoints=[
        APIEndpoint(
            path="/sports/{sport}/{league}/scoreboard",
            description="Get current scores",
            params=["sport", "league"],
            required_params=["sport", "league"],
            example="/sports/football/nfl/scoreboard"
        ),
        APIEndpoint(
            path="/sports/{sport}/{league}/standings",
            description="Get league standings",
            params=["sport", "league"],
            required_params=["sport", "league"],
        ),
    ],
    rate_limit=60,
    docs_url=None,  # Unofficial API
))


_register(APIDefinition(
    id="football_data",
    name="Football-Data.org",
    description="Soccer/Football data: leagues, teams, matches, and standings",
    base_url="https://api.football-data.org/v4",
    auth_type="api_key",
    auth_config={
        "header_name": "X-Auth-Token",
        "env_var": "FOOTBALL_DATA_API_KEY",
        "location": "header",
    },
    signup_url="https://www.football-data.org/client/register",
    free_tier=True,
    data_types=[
        "soccer", "football", "premier_league", "champions_league",
        "la_liga", "bundesliga", "serie_a"
    ],
    keywords=[
        "soccer", "football", "premier league", "champions league",
        "world cup", "match", "goal", "team", "player"
    ],
    endpoints=[
        APIEndpoint(
            path="/competitions/{code}/matches",
            description="Get matches for a competition",
            params=["code", "dateFrom", "dateTo", "status"],
            required_params=["code"],
        ),
        APIEndpoint(
            path="/competitions/{code}/standings",
            description="Get standings for a competition",
            params=["code"],
            required_params=["code"],
        ),
    ],
    rate_limit=10,
    docs_url="https://www.football-data.org/documentation/quickstart",
))


# -----------------------------------------------------------------------------
# Energy & Utilities
# -----------------------------------------------------------------------------

_register(APIDefinition(
    id="eia",
    name="US Energy Information Administration",
    description="US energy data: oil, gas, electricity, renewables, and consumption",
    base_url="https://api.eia.gov/v2",
    auth_type="api_key",
    auth_config={
        "param_name": "api_key",
        "env_var": "EIA_API_KEY",
        "location": "query",
    },
    signup_url="https://www.eia.gov/opendata/register.php",
    free_tier=True,
    data_types=[
        "energy", "oil", "petroleum", "natural_gas", "electricity",
        "coal", "renewables", "solar", "wind", "nuclear", "consumption"
    ],
    keywords=[
        "energy", "oil", "gas", "petroleum", "electricity", "power",
        "renewable", "solar", "wind", "nuclear", "coal", "fuel", "barrel"
    ],
    endpoints=[
        APIEndpoint(
            path="/seriesid/{series_id}",
            description="Get data for a specific series",
            params=["series_id"],
            required_params=["series_id"],
        ),
    ],
    rate_limit=100,
    docs_url="https://www.eia.gov/opendata/documentation.php",
))


# -----------------------------------------------------------------------------
# Social Media & Trends
# -----------------------------------------------------------------------------

_register(APIDefinition(
    id="reddit",
    name="Reddit API",
    description="Reddit posts, comments, and subreddit data",
    base_url="https://www.reddit.com",
    auth_type="none",  # .json endpoints don't require auth
    signup_url="https://www.reddit.com/prefs/apps",
    free_tier=True,
    data_types=[
        "reddit", "social_media", "posts", "comments", "trending"
    ],
    keywords=[
        "reddit", "subreddit", "post", "upvote", "trending", "viral",
        "discussion", "community"
    ],
    endpoints=[
        APIEndpoint(
            path="/r/{subreddit}/top.json",
            description="Top posts from a subreddit",
            params=["subreddit", "t", "limit"],
            required_params=["subreddit"],
            example="/r/technology/top.json?t=week"
        ),
    ],
    rate_limit=60,
    docs_url="https://www.reddit.com/dev/api/",
))


_register(APIDefinition(
    id="github",
    name="GitHub API",
    description="GitHub repositories, users, issues, and activity",
    base_url="https://api.github.com",
    auth_type="api_key",  # Personal access token
    auth_config={
        "header_name": "Authorization",
        "env_var": "GITHUB_TOKEN",
        "location": "header",
        "format": "Bearer {key}",
    },
    signup_url="https://github.com/settings/tokens",
    free_tier=True,
    data_types=[
        "github", "repositories", "code", "issues", "pull_requests",
        "stars", "forks", "contributors", "commits"
    ],
    keywords=[
        "github", "repository", "repo", "code", "open source", "programming",
        "developer", "git", "star", "fork", "issue", "pull request"
    ],
    endpoints=[
        APIEndpoint(
            path="/repos/{owner}/{repo}",
            description="Get repository information",
            params=["owner", "repo"],
            required_params=["owner", "repo"],
        ),
        APIEndpoint(
            path="/search/repositories",
            description="Search repositories",
            params=["q", "sort", "order", "per_page"],
            required_params=["q"],
        ),
    ],
    rate_limit=60,  # 5000/hour with auth
    docs_url="https://docs.github.com/en/rest",
))


# -----------------------------------------------------------------------------
# Transportation & Travel
# -----------------------------------------------------------------------------

_register(APIDefinition(
    id="aviationstack",
    name="AviationStack",
    description="Flight tracking, airline data, and airport information",
    base_url="https://api.aviationstack.com/v1",
    auth_type="api_key",
    auth_config={
        "param_name": "access_key",
        "env_var": "AVIATIONSTACK_API_KEY",
        "location": "query",
    },
    signup_url="https://aviationstack.com/signup/free",
    free_tier=True,
    data_types=[
        "flights", "airlines", "airports", "aviation", "travel"
    ],
    keywords=[
        "flight", "airline", "airport", "airplane", "travel", "aviation",
        "departure", "arrival", "route", "tracking"
    ],
    endpoints=[
        APIEndpoint(
            path="/flights",
            description="Real-time flight data",
            params=["flight_iata", "airline_name", "dep_iata", "arr_iata"],
        ),
    ],
    rate_limit=100,
    docs_url="https://aviationstack.com/documentation",
))


# -----------------------------------------------------------------------------
# Business & Company Data
# -----------------------------------------------------------------------------

_register(APIDefinition(
    id="sec_edgar",
    name="SEC EDGAR",
    description="SEC filings, company financial reports, and regulatory data",
    base_url="https://data.sec.gov",
    auth_type="none",
    signup_url=None,
    free_tier=True,
    data_types=[
        "sec_filings", "10k", "10q", "financial_statements", "company_reports",
        "regulatory", "insider_trading"
    ],
    keywords=[
        "sec", "edgar", "filing", "10k", "10q", "annual report", "quarterly",
        "financial statement", "regulatory", "insider", "company"
    ],
    endpoints=[
        APIEndpoint(
            path="/submissions/CIK{cik}.json",
            description="Company filings by CIK",
            params=["cik"],
            required_params=["cik"],
        ),
    ],
    rate_limit=10,
    docs_url="https://www.sec.gov/os/accessing-edgar-data",
))


_register(APIDefinition(
    id="openai",
    name="OpenAI API",
    description="AI models for text generation, embeddings, and analysis",
    base_url="https://api.openai.com/v1",
    auth_type="api_key",
    auth_config={
        "header_name": "Authorization",
        "env_var": "OPENAI_API_KEY",
        "location": "header",
        "format": "Bearer {key}",
    },
    signup_url="https://platform.openai.com/signup",
    free_tier=False,
    data_types=[
        "ai", "llm", "text_generation", "embeddings", "chat"
    ],
    keywords=[
        "openai", "gpt", "chatgpt", "ai", "llm", "language model",
        "text generation", "embeddings"
    ],
    endpoints=[
        APIEndpoint(
            path="/chat/completions",
            description="Generate chat completions",
            params=["model", "messages", "temperature"],
            required_params=["model", "messages"],
        ),
        APIEndpoint(
            path="/embeddings",
            description="Generate text embeddings",
            params=["model", "input"],
            required_params=["model", "input"],
        ),
    ],
    rate_limit=60,
    docs_url="https://platform.openai.com/docs/api-reference",
))


# -----------------------------------------------------------------------------
# Environment & Sustainability
# -----------------------------------------------------------------------------

_register(APIDefinition(
    id="openaq",
    name="OpenAQ",
    description="Global air quality measurements and pollution data",
    base_url="https://api.openaq.org/v2",
    auth_type="api_key",
    auth_config={
        "header_name": "X-API-Key",
        "env_var": "OPENAQ_API_KEY",
        "location": "header",
    },
    signup_url="https://openaq.org/#/api",
    free_tier=True,
    data_types=[
        "air_quality", "pollution", "pm25", "pm10", "ozone", "no2", "co"
    ],
    keywords=[
        "air quality", "pollution", "pm2.5", "pm10", "ozone", "smog",
        "aqi", "environment", "emissions"
    ],
    endpoints=[
        APIEndpoint(
            path="/measurements",
            description="Air quality measurements",
            params=["city", "country", "parameter", "date_from", "date_to"],
        ),
        APIEndpoint(
            path="/locations",
            description="Monitoring locations",
            params=["city", "country", "limit"],
        ),
    ],
    rate_limit=100,
    docs_url="https://docs.openaq.org/reference",
))


# -----------------------------------------------------------------------------
# Demographics & Public Data
# -----------------------------------------------------------------------------

_register(APIDefinition(
    id="data_usa",
    name="Data USA",
    description="US demographic, economic, and educational data visualizations",
    base_url="https://datausa.io/api",
    auth_type="none",
    signup_url=None,
    free_tier=True,
    data_types=[
        "demographics", "education", "employment", "wages", "health",
        "diversity", "skills", "industries", "occupations"
    ],
    keywords=[
        "usa", "demographic", "education", "employment", "wage",
        "occupation", "industry", "diversity", "skill"
    ],
    endpoints=[
        APIEndpoint(
            path="/data",
            description="Query demographic data",
            params=["drilldowns", "measures", "Geography", "Year"],
            example="/data?drilldowns=State&measures=Population"
        ),
    ],
    rate_limit=60,
    docs_url="https://datausa.io/about/api/",
))


_register(APIDefinition(
    id="eurostat",
    name="Eurostat",
    description="European Union statistics on economy, population, trade",
    base_url="https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0",
    auth_type="none",
    signup_url=None,
    free_tier=True,
    data_types=[
        "eu_statistics", "european_union", "trade", "population",
        "economy", "employment", "inflation", "gdp"
    ],
    keywords=[
        "eurostat", "europe", "eu", "european union", "euro",
        "european economy", "eu trade", "european population"
    ],
    endpoints=[
        APIEndpoint(
            path="/data/{dataset}",
            description="Get dataset",
            params=["dataset", "geo", "time"],
            required_params=["dataset"],
        ),
    ],
    rate_limit=60,
    docs_url="https://ec.europa.eu/eurostat/web/sdmx-infospace/welcome",
))


# -----------------------------------------------------------------------------
# Food & Agriculture
# -----------------------------------------------------------------------------

_register(APIDefinition(
    id="usda_fdc",
    name="USDA FoodData Central",
    description="Detailed nutritional information for foods and ingredients",
    base_url="https://api.nal.usda.gov/fdc/v1",
    auth_type="api_key",
    auth_config={
        "param_name": "api_key",
        "env_var": "USDA_API_KEY",
        "location": "query",
    },
    signup_url="https://fdc.nal.usda.gov/api-key-signup.html",
    free_tier=True,
    data_types=[
        "nutrition", "food", "calories", "protein", "vitamins",
        "ingredients", "diet"
    ],
    keywords=[
        "food", "nutrition", "calories", "protein", "vitamin",
        "diet", "ingredient", "usda", "nutrient"
    ],
    endpoints=[
        APIEndpoint(
            path="/foods/search",
            description="Search foods",
            params=["query", "pageSize", "pageNumber"],
            required_params=["query"],
        ),
        APIEndpoint(
            path="/food/{fdcId}",
            description="Get food details",
            params=["fdcId"],
            required_params=["fdcId"],
        ),
    ],
    rate_limit=1000,
    docs_url="https://fdc.nal.usda.gov/api-guide.html",
))


# -----------------------------------------------------------------------------
# Movies & Entertainment
# -----------------------------------------------------------------------------

_register(APIDefinition(
    id="omdb",
    name="OMDb API",
    description="Movie and TV show information, ratings, and metadata",
    base_url="https://www.omdbapi.com",
    auth_type="api_key",
    auth_config={
        "param_name": "apikey",
        "env_var": "OMDB_API_KEY",
        "location": "query",
    },
    signup_url="https://www.omdbapi.com/apikey.aspx",
    free_tier=True,
    data_types=[
        "movies", "tv_shows", "ratings", "imdb", "actors", "directors"
    ],
    keywords=[
        "movie", "film", "tv", "television", "imdb", "rating",
        "actor", "director", "cinema", "show"
    ],
    endpoints=[
        APIEndpoint(
            path="/",
            description="Search movies and shows",
            params=["t", "s", "i", "y", "type", "plot"],
            example="/?t=Inception"
        ),
    ],
    rate_limit=1000,
    docs_url="https://www.omdbapi.com/",
))


_register(APIDefinition(
    id="tmdb",
    name="The Movie Database (TMDb)",
    description="Extensive movie and TV database with images and metadata",
    base_url="https://api.themoviedb.org/3",
    auth_type="api_key",
    auth_config={
        "param_name": "api_key",
        "env_var": "TMDB_API_KEY",
        "location": "query",
    },
    signup_url="https://www.themoviedb.org/signup",
    free_tier=True,
    data_types=[
        "movies", "tv_shows", "actors", "images", "trailers", "reviews"
    ],
    keywords=[
        "tmdb", "movie database", "film", "tv", "cast", "crew",
        "poster", "trailer", "review"
    ],
    endpoints=[
        APIEndpoint(
            path="/search/movie",
            description="Search movies",
            params=["query", "year", "language"],
            required_params=["query"],
        ),
        APIEndpoint(
            path="/movie/{movie_id}",
            description="Get movie details",
            params=["movie_id"],
            required_params=["movie_id"],
        ),
    ],
    rate_limit=40,
    docs_url="https://developers.themoviedb.org/3",
))


# -----------------------------------------------------------------------------
# Books & Literature
# -----------------------------------------------------------------------------

_register(APIDefinition(
    id="open_library",
    name="Open Library",
    description="Book metadata, covers, and lending library data",
    base_url="https://openlibrary.org",
    auth_type="none",
    signup_url=None,
    free_tier=True,
    data_types=[
        "books", "authors", "isbn", "library", "literature"
    ],
    keywords=[
        "book", "author", "isbn", "library", "reading", "literature",
        "novel", "publication"
    ],
    endpoints=[
        APIEndpoint(
            path="/search.json",
            description="Search books",
            params=["q", "title", "author", "limit"],
        ),
        APIEndpoint(
            path="/api/books",
            description="Get book metadata by ISBN/OLID",
            params=["bibkeys", "format", "jscmd"],
            required_params=["bibkeys"],
        ),
    ],
    rate_limit=100,
    docs_url="https://openlibrary.org/developers/api",
))


# -----------------------------------------------------------------------------
# Exchange Rates & Currency
# -----------------------------------------------------------------------------

_register(APIDefinition(
    id="exchangerate",
    name="ExchangeRate-API",
    description="Currency exchange rates and conversion",
    base_url="https://v6.exchangerate-api.com/v6",
    auth_type="api_key",
    auth_config={
        "param_name": None,  # Key is in URL path
        "env_var": "EXCHANGERATE_API_KEY",
        "location": "path",
    },
    signup_url="https://www.exchangerate-api.com/",
    free_tier=True,
    data_types=[
        "exchange_rates", "currency", "forex", "conversion"
    ],
    keywords=[
        "exchange rate", "currency", "forex", "convert", "usd", "eur",
        "gbp", "jpy", "money", "foreign exchange"
    ],
    endpoints=[
        APIEndpoint(
            path="/{api_key}/latest/{base}",
            description="Latest exchange rates",
            params=["base"],
            required_params=["base"],
            example="/{api_key}/latest/USD"
        ),
    ],
    rate_limit=100,
    docs_url="https://www.exchangerate-api.com/docs",
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
   - Or store it securely in Assay's credential manager
""".strip(),
    }

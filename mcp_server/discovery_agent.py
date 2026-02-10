"""
API Discovery Agent - Fully autonomous API discovery and connector generation.

This agent takes a natural language description of data needs and:
1. Searches the registry for matching APIs
2. Uses web search to find API documentation
3. Auto-generates connectors
4. Returns ready-to-use data fetching capability

This enables a completely hands-off experience where users just describe
what data they want and the system figures out how to get it.
"""
import os
import re
import logging
import requests
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredAPI:
    """Result of API discovery."""
    name: str
    description: str
    base_url: str
    docs_url: Optional[str] = None
    openapi_url: Optional[str] = None
    auth_type: str = "unknown"
    signup_url: Optional[str] = None
    source: str = "registry"  # "registry", "web_search", "llm"
    confidence: float = 0.0


@dataclass
class AutoConnectResult:
    """Result of auto-connect flow."""
    success: bool
    api_name: str = ""
    connector_code: Optional[str] = None
    sample_data: Optional[Any] = None
    error: Optional[str] = None
    needs_auth: bool = False
    auth_instructions: Optional[str] = None
    signup_url: Optional[str] = None


@dataclass
class FetchResult:
    """Result from one_click_fetch with rich auth info for UI."""
    success: bool
    data: Optional[Any] = None
    status: str = ""
    # Auth info for guided UI
    needs_auth: bool = False
    api_name: str = ""
    api_id: str = ""
    auth_type: str = ""
    signup_url: Optional[str] = None
    auth_instructions: Optional[str] = None


class WebSearcher:
    """
    Search the web for API documentation.
    
    Uses multiple strategies:
    1. Common API documentation patterns
    2. Search engines (if available)
    3. Known API directories
    """
    
    # Common OpenAPI/Swagger URL patterns
    OPENAPI_PATTERNS = [
        "{base}/swagger.json",
        "{base}/openapi.json",
        "{base}/api-docs",
        "{base}/v1/swagger.json",
        "{base}/v2/swagger.json",
        "{base}/v3/swagger.json",
        "{base}/api/swagger.json",
        "{base}/docs/swagger.json",
        "{base}/spec/openapi.json",
    ]
    
    # Known API directories with free APIs
    API_DIRECTORIES = {
        "rapidapi": "https://rapidapi.com/search/{query}",
        "publicapis": "https://api.publicapis.org/entries?title={query}",
        "apilist": "https://apilist.fun/api/search?q={query}",
    }
    
    def search_for_api_docs(self, api_name: str, base_url: str = "") -> Optional[str]:
        """
        Try to find OpenAPI/Swagger documentation for an API.
        
        Args:
            api_name: Name of the API to search for
            base_url: Base URL if known
            
        Returns:
            URL to OpenAPI spec if found, None otherwise
        """
        if base_url:
            # Try common patterns on the base URL
            for pattern in self.OPENAPI_PATTERNS:
                url = pattern.format(base=base_url.rstrip('/'))
                if self._is_valid_openapi(url):
                    logger.info(f"Found OpenAPI spec at {url}")
                    return url
        
        # Try to construct URLs from API name
        sanitized_name = api_name.lower().replace(' ', '').replace('-', '')
        possible_bases = [
            f"https://api.{sanitized_name}.com",
            f"https://{sanitized_name}.api.com",
            f"https://api.{sanitized_name}.io",
        ]
        
        for base in possible_bases:
            for pattern in self.OPENAPI_PATTERNS:
                url = pattern.format(base=base)
                if self._is_valid_openapi(url):
                    return url
        
        return None
    
    def _is_valid_openapi(self, url: str) -> bool:
        """Check if URL returns a valid OpenAPI spec."""
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                content = response.text
                # Check for OpenAPI markers
                if '"openapi"' in content or '"swagger"' in content:
                    return True
        except Exception:
            pass
        return False
    
    def search_public_apis(self, query: str) -> List[DiscoveredAPI]:
        """
        Search public API directories for matching APIs.
        
        Args:
            query: Search query describing desired data
            
        Returns:
            List of discovered APIs
        """
        results = []
        
        # Try publicapis.org
        try:
            response = requests.get(
                f"https://api.publicapis.org/entries",
                params={"title": query},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                for entry in data.get("entries", [])[:5]:
                    results.append(DiscoveredAPI(
                        name=entry.get("API", "Unknown"),
                        description=entry.get("Description", ""),
                        base_url=entry.get("Link", ""),
                        auth_type="api_key" if entry.get("Auth") else "none",
                        source="web_search",
                        confidence=0.7,
                    ))
        except Exception as e:
            logger.debug(f"publicapis.org search failed: {e}")
        
        return results


class APIDiscoveryAgent:
    """
    Fully autonomous API discovery agent.
    
    Workflow:
    1. User describes data need
    2. Agent searches registry
    3. If no match, searches web for APIs
    4. Fetches API documentation
    5. Generates connector
    6. Attempts to fetch data
    7. Returns results
    """
    
    # Default/fallback vertical weights (used if Kaggle API unavailable)
    # These are based on domain knowledge of common data sources
    DEFAULT_VERTICAL_SOURCES = {
        "finance": {
            "preferred": ["alpha_vantage", "fred", "sec_edgar", "yahoo_finance"],
            "boost": 0.3,
        },
        "economics": {
            "preferred": ["fred", "world_bank", "bls", "census", "eurostat"],
            "boost": 0.3,
        },
        "weather": {
            "preferred": ["openweathermap", "noaa"],
            "boost": 0.3,
        },
        "environment": {
            "preferred": ["noaa", "openaq", "world_bank"],
            "boost": 0.3,
        },
        "health": {
            "preferred": ["who", "cdc", "world_bank"],
            "boost": 0.3,
        },
        "demographics": {
            "preferred": ["census", "world_bank", "data_usa", "eurostat"],
            "boost": 0.3,
        },
        "sports": {
            "preferred": ["espn", "football_data"],
            "boost": 0.3,
        },
        "crypto": {
            "preferred": ["coingecko", "coinmarketcap"],
            "boost": 0.3,
        },
        "energy": {
            "preferred": ["eia"],
            "boost": 0.3,
        },
        "food": {
            "preferred": ["usda_fdc"],
            "boost": 0.3,
        },
        "movies": {
            "preferred": ["tmdb", "omdb", "imdb"],
            "boost": 0.3,
        },
        "social": {
            "preferred": ["reddit", "twitter"],
            "boost": 0.3,
        },
    }
    
    # Keywords that indicate vertical - organized by priority
    # Primary keywords have stronger weight (matched first)
    VERTICAL_KEYWORDS = {
        "finance": {
            "primary": ["stock", "trading", "nasdaq", "nyse", "s&p", "dow jones", "hedge fund", "ipo"],
            "secondary": ["investment", "portfolio", "market cap", "dividend", "earnings", "ticker", 
                         "aapl", "msft", "googl", "tsla", "amzn", "shares", "equity", "bull", "bear"]
        },
        "economics": {
            "primary": ["gdp", "unemployment", "inflation", "federal reserve", "fed", "interest rate",
                       "social mobility", "income inequality", "wealth inequality", "gini"],
            "secondary": ["economic", "macro", "recession", "monetary", "fiscal", "trade deficit",
                         "consumer spending", "labor market", "jobs report", "cpi", "ppi",
                         "poverty", "wealth quintile", "income quintile", "minimum wage",
                         "economic mobility", "generational wealth", "median income", "wage gap"]
        },
        "weather": {
            "primary": ["weather", "temperature", "forecast", "hurricane", "storm"],
            "secondary": ["rain", "snow", "climate", "atmospheric", "wind speed", "humidity", 
                         "precipitation", "sunny", "cloudy", "celsius", "fahrenheit"]
        },
        "environment": {
            "primary": ["pollution", "air quality", "emissions", "carbon", "climate change"],
            "secondary": ["greenhouse", "co2", "renewable", "sustainability", "epa", "ozone",
                         "deforestation", "biodiversity", "wildfire", "sea level"]
        },
        "health": {
            "primary": ["disease", "covid", "pandemic", "hospital", "vaccine", "mortality"],
            "secondary": ["vaccination", "cases", "outbreak", "symptoms", "cdc", "who",
                         "healthcare", "medical", "therapy", "treatment", "clinical"]
        },
        "demographics": {
            "primary": ["population", "census", "demographic", "social mobility"],
            "secondary": ["age distribution", "gender", "income", "household", "birth rate",
                         "death rate", "migration", "ethnicity", "education level",
                         "socioeconomic", "quintile", "decile", "upward mobility", 
                         "intergenerational", "class", "middle class", "poverty rate"]
        },
        "sports": {
            "primary": ["nfl", "nba", "mlb", "nhl", "fifa", "espn", "sports"],
            "secondary": ["score", "game", "player", "team", "stadium", "championship", "playoff",
                         "touchdown", "homerun", "goal", "assist", "lebron", "curry", "mahomes",
                         "messi", "ronaldo", "basketball", "football", "soccer", "baseball", 
                         "hockey", "tennis", "golf", "olympics", "world cup", "super bowl",
                         "standings", "stats", "athlete", "coach", "roster"]
        },
        "crypto": {
            "primary": ["bitcoin", "ethereum", "crypto", "blockchain"],
            "secondary": ["btc", "eth", "defi", "nft", "binance", "coinbase", "wallet",
                         "mining", "altcoin", "token", "smart contract", "web3", "airdrop"]
        },
        "energy": {
            "primary": ["oil price", "natural gas", "electricity", "renewable energy"],
            "secondary": ["oil", "gas", "solar", "wind", "nuclear", "coal", "power grid",
                         "barrel", "opec", "energy sector", "utilities", "kwh", "megawatt"]
        },
        "food": {
            "primary": ["nutrition", "calories", "food data", "ingredients"],
            "secondary": ["food", "diet", "protein", "carbs", "fat", "vitamins", "usda",
                         "recipe", "allergen", "organic", "gmo", "meal"]
        },
        "movies": {
            "primary": ["movie", "film", "imdb", "tmdb", "box office"],
            "secondary": ["actor", "actress", "director", "rating", "cinema", "hollywood",
                         "netflix", "streaming", "oscar", "screenplay", "genre", "sequel"]
        },
        "social": {
            "primary": ["reddit", "twitter", "trending", "viral", "social media"],
            "secondary": ["posts", "likes", "followers", "hashtag", "meme",
                         "influencer", "engagement", "sentiment", "news", "opinion"]
        },
    }
    
    def __init__(self, use_kaggle: bool = True):
        """
        Initialize the discovery agent.
        
        Args:
            use_kaggle: If True, enable Kaggle integration (weights loaded lazily).
        """
        self.web_searcher = WebSearcher()
        self._connector_cache: Dict[str, Any] = {}
        self._kaggle_analyzer = None
        self._vertical_sources = self.DEFAULT_VERTICAL_SOURCES.copy()
        self._use_kaggle = use_kaggle
        self._kaggle_loaded = False  # Track if we've attempted Kaggle load
    
    def clear_credentials(self):
        """Clear cached credentials to lock session."""
        if self._kaggle_analyzer:
            self._kaggle_analyzer.clear_credentials()

    def _load_kaggle_weights(self, master_password: Optional[str] = None):
        """
        Load dynamic weights from Kaggle analyzer.
        
        Args:
            master_password: Required to decrypt stored Kaggle credentials
        """
        # Allow retry if we have an analyzer but it wasn't authenticated
        if self._kaggle_loaded and self._kaggle_analyzer:
            # Already loaded successfully, just update password if provided
            if master_password:
                self._kaggle_analyzer.set_master_password(master_password)
                
                # If we haven't loaded weights yet, try again with new password
                if not self._kaggle_loaded:
                    logger.info("Retrying Kaggle weight loading with updated password...")
                    kaggle_weights = self._kaggle_analyzer.export_for_discovery_agent()
                    if kaggle_weights:
                        for vertical, config in kaggle_weights.items():
                            if config.get("preferred"):
                                self._vertical_sources[vertical] = config
                                logger.info(f"Loaded Kaggle weights for '{vertical}': {config['preferred'][:3]}")
                        self._kaggle_loaded = True
            return
        
        logger.info(f"Attempting to load Kaggle weights (password: {'provided' if master_password else 'not provided'})")
        
        try:
            from mcp_server.kaggle_source_analyzer import KaggleSourceAnalyzer
            
            # Create analyzer with master password for credential decryption
            self._kaggle_analyzer = KaggleSourceAnalyzer(master_password=master_password)
            
            # Export Kaggle-derived weights
            kaggle_weights = self._kaggle_analyzer.export_for_discovery_agent()
            
            if kaggle_weights:
                # Merge with defaults (Kaggle takes priority)
                for vertical, config in kaggle_weights.items():
                    if config.get("preferred"):  # Only update if we got real data
                        self._vertical_sources[vertical] = config
                        logger.info(f"Loaded Kaggle weights for '{vertical}': {config['preferred'][:3]}")
                
                # Mark as successfully loaded
                self._kaggle_loaded = True
            else:
                logger.info("No Kaggle weights returned (may retry later with password)")
            
        except ImportError as e:
            logger.warning(f"Kaggle module not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to load Kaggle weights: {e}")
    
    def refresh_kaggle_weights(self) -> bool:
        """
        Refresh weights from Kaggle API.
        
        Call this periodically to update source rankings.
        
        Returns:
            True if refresh succeeded, False otherwise
        """
        if self._kaggle_analyzer is None:
            self._load_kaggle_weights()
            return self._kaggle_analyzer is not None
        
        try:
            self._kaggle_analyzer.refresh_all_verticals()
            kaggle_weights = self._kaggle_analyzer.export_for_discovery_agent()
            
            for vertical, config in kaggle_weights.items():
                if config.get("preferred"):
                    self._vertical_sources[vertical] = config
            
            return True
        except Exception as e:
            logger.error(f"Kaggle refresh failed: {e}")
            return False
    
    @property
    def vertical_sources(self) -> Dict[str, Dict]:
        """Get current vertical sources (Kaggle or defaults)."""
        return self._vertical_sources
    
    def _detect_vertical(self, query: str) -> Optional[str]:
        """
        Detect the data vertical from query keywords using weighted scoring.
        
        Primary keywords (e.g., 'NBA', 'stock') get higher weight (3 points)
        Secondary keywords (e.g., 'LeBron', 'ticker') get lower weight (1 point)
        
        This allows contextual clues to contribute while direct keywords dominate.
        """
        query_lower = query.lower()
        
        best_match = None
        best_score = 0
        
        for vertical, keywords_config in self.VERTICAL_KEYWORDS.items():
            score = 0
            
            # Primary keywords have higher weight
            primary = keywords_config.get("primary", [])
            for kw in primary:
                if kw in query_lower:
                    score += 3  # Strong signal
            
            # Secondary keywords contribute less
            secondary = keywords_config.get("secondary", [])
            for kw in secondary:
                if kw in query_lower:
                    score += 1  # Weak signal
            
            if score > best_score:
                best_score = score
                best_match = vertical
        
        # Require at least 1 point to match (any keyword)
        return best_match if best_score > 0 else None
    
    def _apply_vertical_boost(self, apis: List[DiscoveredAPI], vertical: str) -> List[DiscoveredAPI]:
        """
        Apply confidence boost to preferred sources for a vertical.
        
        Uses dynamically-loaded Kaggle weights if available,
        otherwise falls back to default weights.
        """
        if vertical not in self._vertical_sources:
            return apis
        
        config = self._vertical_sources[vertical]
        preferred = config.get("preferred", [])
        boost = config.get("boost", 0.3)
        
        # Log if using Kaggle-derived weights
        if config.get("kaggle_derived"):
            logger.info(f"Using Kaggle-derived weights for '{vertical}'")
        
        for api in apis:
            # Check if this API is in preferred list for the vertical
            api_id = api.name.lower().replace(' ', '_').replace('-', '_')
            for pref in preferred:
                if pref in api_id or api_id in pref:
                    api.confidence = min(1.0, api.confidence + boost)
                    logger.debug(f"Boosted {api.name} for vertical '{vertical}'")
                    break
        
        return apis
    
    def discover_api(self, user_query: str) -> List[DiscoveredAPI]:
        """
        Discover APIs matching user's data needs.
        
        Uses Kaggle-derived vertical weighting to boost commonly-used
        data sources per domain.
        
        Args:
            user_query: Natural language description of data needed
            
        Returns:
            List of discovered APIs, ranked by relevance
        """
        results = []
        
        # Detect vertical for smart boosting
        vertical = self._detect_vertical(user_query)
        if vertical:
            logger.info(f"Detected vertical: {vertical}")
        
        # 1. Search registry first
        # 1. Search registry using Semantic Router (Embeddings + LLM Fallback)
        try:
            from mcp_server.semantic_router import get_router
            from mcp_server.api_registry import get_api
            
            router = get_router()
            # Search with LLM fallback enabled for low confidence matches
            matches = router.search(user_query, top_k=5, llm_fallback=True)
            
            for match in matches:
                api_def = get_api(match.api_id)
                if api_def:
                    # Normalize confidence score
                    # Semantic router returns 0-1 score, we want it to reflect in our confidence
                    confidence = match.score
                    
                    # Boost confidence if matched via LLM (router gives 0.9, but we can trust it significantly)
                    if match.matched_via == "llm":
                        logger.info(f"LLM matched query '{user_query}' to API '{match.name}'")
                    
                    results.append(DiscoveredAPI(
                        name=api_def.name,
                        description=api_def.description,
                        base_url=api_def.base_url,
                        docs_url=api_def.docs_url,
                        openapi_url=api_def.openapi_url if hasattr(api_def, 'openapi_url') else None,
                        auth_type=api_def.auth_type,
                        signup_url=api_def.signup_url,
                        source="registry",
                        confidence=confidence,
                    ))
                    
        except Exception as e:
            logger.error(f"Semantic registry search failed: {e}")
            # Fallback to simple keyword search if semantic router fails
            try:
                from mcp_server.api_registry import search_apis_by_query, get_api
                logger.info("Falling back to keyword search")
                registry_matches = search_apis_by_query(user_query)
                
                for match in registry_matches[:5]:
                    api_def = get_api(match["api_id"])
                    if api_def:
                        results.append(DiscoveredAPI(
                            name=api_def.name,
                            description=api_def.description,
                            base_url=api_def.base_url,
                            docs_url=api_def.docs_url,
                            openapi_url=api_def.openapi_url if hasattr(api_def, 'openapi_url') else None,
                            auth_type=api_def.auth_type,
                            signup_url=api_def.signup_url,
                            source="registry",
                            confidence=min(1.0, match["score"] / 20),
                        ))
            except Exception as e2:
                logger.error(f"Keyword fallback failed: {e2}")
        
        # 2. If no good matches, search web
        if not results or results[0].confidence < 0.5:
            web_results = self.web_searcher.search_public_apis(user_query)
            results.extend(web_results)
        
        # 3. Apply vertical-specific boost (Kaggle-informed weighting)
        if vertical:
            results = self._apply_vertical_boost(results, vertical)
        
        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        return results
    
    def auto_connect(
        self, 
        api: DiscoveredAPI,
        api_key: Optional[str] = None
    ) -> AutoConnectResult:
        """
        Automatically generate a connector and attempt data fetch.
        
        Args:
            api: Discovered API to connect to
            api_key: API key if auth is required
            
        Returns:
            AutoConnectResult with connector and sample data
        """
        result = AutoConnectResult(
            success=False,
            api_name=api.name,
        )
        
        # Check if auth is needed
        if api.auth_type not in ("none", "unknown") and not api_key:
            result.needs_auth = True
            result.auth_instructions = f"This API requires authentication ({api.auth_type})."
            result.signup_url = api.signup_url
            # Still try to generate connector
        
        # Find documentation URL
        docs_url = api.openapi_url or api.docs_url
        
        if not docs_url:
            # Try to find it
            docs_url = self.web_searcher.search_for_api_docs(api.name, api.base_url)
        
        if not docs_url:
            # Fall back to base URL
            docs_url = api.base_url or api.docs_url
        
        if not docs_url:
            result.error = "Could not find API documentation"
            return result
        
        # Generate connector
        try:
            from mcp_server.dynamic_connector import get_connector_manager
            
            manager = get_connector_manager()
            generated = manager.generate_connector(docs_url, api_key)
            
            if generated.validated:
                result.success = True
                result.connector_code = generated.code
                
                # Try to fetch sample data if we have auth
                if api_key or api.auth_type == "none":
                    try:
                        connector = manager.get_connector_instance(docs_url, api_key)
                        if connector and hasattr(connector, 'fetch_data'):
                            # Try fetching from first endpoint
                            sample = connector.fetch_data("/", params={})
                            if sample is not None and len(sample) > 0:
                                result.sample_data = sample.head(5).to_dict()
                    except Exception as e:
                        logger.debug(f"Sample fetch failed: {e}")
                
            else:
                result.error = generated.error
                
        except Exception as e:
            logger.error(f"Connector generation failed: {e}")
            result.error = str(e)
        
        return result
    
    def one_click_fetch(
        self,
        user_query: str,
        api_key: Optional[str] = None
    ) -> Tuple[Optional[Any], str]:
        """
        Complete one-click flow: describe data -> get data.
        
        Args:
            user_query: Natural language description of desired data
            api_key: Optional API key for authenticated APIs
            
        Returns:
            Tuple of (data or None, status message)
        """
        # 1. Discover APIs
        apis = self.discover_api(user_query)
        
        if not apis:
            return None, "No APIs found matching your query. Try being more specific about the data you need."
        
        # 2. Try each API until one works
        for api in apis[:3]:  # Try top 3
            logger.info(f"Trying {api.name}...")
            
            result = self.auto_connect(api, api_key)
            
            if result.success:
                if result.sample_data:
                    return result.sample_data, f"Successfully fetched data from {api.name}"
                elif result.needs_auth:
                    return None, f"Found {api.name} but authentication required. Visit: {result.signup_url}"
                else:
                    return None, f"Connected to {api.name} but no data returned. The connector is ready for use."
        
        # All failed
        best_api = apis[0]
        return None, f"Found {best_api.name} but couldn't fetch data automatically. You may need to configure authentication."

    def one_click_fetch_rich(
        self,
        user_query: str,
        api_key: Optional[str] = None
    ) -> FetchResult:
        """
        Complete one-click flow with rich result for guided auth UI.
        
        Returns FetchResult with structured auth info when authentication
        is needed, allowing the UI to prompt the user inline.
        """
        # 1. Discover APIs
        apis = self.discover_api(user_query)
        
        if not apis:
            return FetchResult(
                success=False,
                status="No APIs found matching your query. Try being more specific."
            )
        
        # 2. Try each API until one works
        for api in apis[:3]:
            logger.info(f"Trying {api.name}...")
            
            result = self.auto_connect(api, api_key)
            
            if result.success:
                if result.sample_data:
                    return FetchResult(
                        success=True,
                        data=result.sample_data,
                        status=f"Successfully fetched data from {api.name}",
                        api_name=api.name,
                        api_id=getattr(api, 'api_id', api.name.lower().replace(' ', '_'))
                    )
                elif result.needs_auth:
                    # Return rich auth info for guided UI
                    return FetchResult(
                        success=False,
                        needs_auth=True,
                        api_name=api.name,
                        api_id=getattr(api, 'api_id', api.name.lower().replace(' ', '_')),
                        auth_type=api.auth_type,
                        signup_url=result.signup_url or api.signup_url,
                        auth_instructions=result.auth_instructions,
                        status=f"Found {api.name} - authentication required"
                    )
                else:
                    return FetchResult(
                        success=True,
                        data=None,
                        status=f"Connected to {api.name} but no data returned. Connector is ready.",
                        api_name=api.name
                    )
        
        # All failed - return best match with auth info if known
        best_api = apis[0]
        return FetchResult(
            success=False,
            api_name=best_api.name,
            api_id=getattr(best_api, 'api_id', best_api.name.lower().replace(' ', '_')),
            auth_type=best_api.auth_type,
            signup_url=best_api.signup_url,
            needs_auth=best_api.auth_type not in ('none', 'unknown'),
            status=f"Found {best_api.name} but couldn't fetch data. Authentication may be required."
        )


# Singleton instance
_agent: Optional[APIDiscoveryAgent] = None


def get_discovery_agent(master_password: Optional[str] = None) -> APIDiscoveryAgent:
    """
    Get the global API discovery agent.
    
    Args:
        master_password: Optional password to unlock encrypted Kaggle credentials.
                        If provided, triggers lazy loading of Kaggle weights.
                        If None, ensures credentials are cleared (for session safety).
    """
    global _agent
    if _agent is None:
        _agent = APIDiscoveryAgent(use_kaggle=True)
    
    if _agent._use_kaggle:
        if master_password:
            # If master password provided, trigger loading or update password
            _agent._load_kaggle_weights(master_password=master_password)
        else:
            # No password provided for this request/session
            # Clear any existing credentials to prevent session leakage
            _agent.clear_credentials()
    
    return _agent


def discover_and_fetch(query: str, api_key: Optional[str] = None) -> Tuple[Optional[Any], str]:
    """
    Convenience function for one-click data fetching.
    
    Example:
        data, status = discover_and_fetch("US GDP data")
        if data:
            print(data)
        else:
            print(status)
    """
    return get_discovery_agent().one_click_fetch(query, api_key)

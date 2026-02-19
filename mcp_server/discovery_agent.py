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
    registry_id: Optional[str] = None  # Original ID from API registry (e.g. 'census')


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

    def ask_llm_for_api(self, query: str) -> List[DiscoveredAPI]:
        """
        Ask LLM to identify the best official API for a query.
        """
        try:
            from llm_manager.llm_interface import get_llm_completion
            
            prompt = f"""You are an expert Data Engineer. Identify the official public API for this data request: "{query}".
            
            Return a JSON object with:
            - name: Exact name of the API
            - base_url: The base URL of the API
            - description: Brief description
            - auth_type: "api_key", "oauth2", or "none"
            - home_url: Main website URL
            
            Example:
            {{
                "name": "Steam Web API",
                "base_url": "https://api.steampowered.com",
                "description": "Valve's official API for Steam data",
                "auth_type": "api_key",
                "home_url": "https://steamcommunity.com/dev"
            }}
            
            If no public API exists, return null.
            JSON:"""
            
            response = get_llm_completion(prompt, max_tokens=256, temperature=0.0)
            
            # Simple JSON extraction
            import json
            import re
            
            # Find JSON block
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                if data and data.get("name"):
                    return [DiscoveredAPI(
                        name=data["name"],
                        description=data.get("description", ""),
                        base_url=data.get("base_url", ""),
                        docs_url=data.get("home_url", ""),
                        auth_type=data.get("auth_type", "unknown"),
                        source="llm_knowledge",
                        confidence=0.85  # High confidence for LLM knowledge
                    )]
            
        except Exception as e:
            logger.error(f"LLM API discovery failed: {e}")
            
        return []


class APIDiscoveryAgent:
    """
    Fully autonomous API discovery agent.
    
    Workflow:
    1. User describes data need
    2. Agent searches registry for matching APIs
    3. If registry match: uses LLM to map query → endpoint parameters,
       constructs the API call directly, and fetches data
    4. If no registry match: searches web for API docs
    5. Falls back to dynamic connector generation (OpenAPI)
    6. Returns results
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
        "gaming": {
            "preferred": ["steam", "igdb", "rawg"],
            "boost": 0.4,
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
        "gaming": {
            "primary": ["steam", "valve", "twitch", "gaming", "esports", "minecraft", "roblox", "nintendo", "xbox", "playstation"],
            "secondary": ["game", "games", "player", "score", "achievement", "level", "boss", "quest", "rpg", "fps", "skin", "loot"]
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
                        registry_id=api_def.id,
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
            registry_id=api_def.id,
                        ))
            except Exception as e2:
                logger.error(f"Keyword fallback failed: {e2}")
        
        # 2. LLM-based intelligent fallback (High Confidence)
        # If registry didn't give a perfect match (confidence > 0.9), ask the LLM for the real API
        if not results or results[0].confidence < 0.9:
            logger.info("Checking LLM knowledge for API...")
            llm_results = self.web_searcher.ask_llm_for_api(user_query)
            if llm_results:
                logger.info(f"LLM suggested API: {llm_results[0].name}")
                results.extend(llm_results)

        # 3. Web Search fallback (Public lists)
        if not results or results[0].confidence < 0.5:
            web_results = self.web_searcher.search_public_apis(user_query)
            results.extend(web_results)
        
        # 4. Apply vertical-specific boost (Kaggle-informed weighting)
        if vertical:
            results = self._apply_vertical_boost(results, vertical)
        
        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        return results
    
    
    def _resolve_api_key(self, api_name: str) -> Optional[str]:
        """
        Attempt to resolve an API key from environment variables based on the API name.
        
        This checks the registry for known mappings first, then tries common patterns.
        """
        import os
        from mcp_server.api_registry import API_REGISTRY
        
        # 1. Check Registry for exact or fuzzy match
        # Normalize name for comparison
        clean_name = api_name.lower().replace(" ", "")
        
        for api_id, api_def in API_REGISTRY.items():
            # Check ID match
            if api_id.lower() in clean_name or clean_name in api_id.lower().replace("_", ""):
                if api_def.auth_config and api_def.auth_config.get("env_var"):
                    env_var = api_def.auth_config["env_var"]
                    key = os.environ.get(env_var)
                    if key:
                        logger.info(f"Resolved API key for '{api_name}' from {env_var}")
                        return key
            
            # Check Name match
            registry_clean_name = api_def.name.lower().replace(" ", "")
            if registry_clean_name in clean_name or clean_name in registry_clean_name:
                if api_def.auth_config and api_def.auth_config.get("env_var"):
                    env_var = api_def.auth_config["env_var"]
                    key = os.environ.get(env_var)
                    if key:
                        logger.info(f"Resolved API key for '{api_name}' from {env_var}")
                        return key

        # 2. Heuristic fallback (e.g. "OpenWeatherMap" -> OPENWEATHERMAP_API_KEY)
        # Convert "US Census Bureau" -> "US_CENSUS_BUREAU_API_KEY" or "CENSUS_API_KEY"
        
        # Try exact upper case with _API_KEY
        env_guess_1 = api_name.upper().replace(" ", "_") + "_API_KEY"
        if os.environ.get(env_guess_1):
            return os.environ.get(env_guess_1)
            
        # Try without "API" (e.g. just "CENSUS" if user named it that, though less likely)
        # But also try removing common words like "THE", "US", "BUREAU", "API" for the variable name
        short_name = api_name.upper().replace("THE", "").replace("US", "").replace("BUREAU", "").replace("API", "").replace("WEB", "").strip().replace("  ", " ").replace(" ", "_")
        env_guess_2 = short_name + "_API_KEY"
        if os.environ.get(env_guess_2):
             return os.environ.get(env_guess_2)
             
        # Try specifically for Census since that was the user report
        if "CENSUS" in api_name.upper():
            if os.environ.get("CENSUS_API_KEY"):
                return os.environ.get("CENSUS_API_KEY")

        return None

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
        # Attempt to auto-resolve key if not provided
        if not api_key:
            api_key = self._resolve_api_key(api.name)
            
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
    ) -> FetchResult:
        """
        Complete one-click flow: describe data -> get data.
        
        Args:
            user_query: Natural language description of desired data
            api_key: Optional API key for authenticated APIs
            
        Returns:
            FetchResult object with data, status, and auth info
        """
        # 1. Discover APIs
        apis = self.discover_api(user_query)
        
        if not apis:
             return FetchResult(success=False, status="No relevant APIs found.")
             
        # 2. Try each API until one works
        for api in apis[:3]:  # Try top 3
            logger.info(f"Trying {api.name}...")
            
            result = self.auto_connect(api, api_key)
            
            if result.success:
                if result.sample_data:
                    return FetchResult(
                        success=True, 
                        data=result.sample_data, 
                        status=f"Successfully fetched data from {api.name}",
                        api_name=api.name
                    )
                elif result.needs_auth:
                    # Return rich auth info so UI can prompt user
                    return FetchResult(
                        success=False,
                        status=f"Authentication required for {api.name}",
                        needs_auth=True,
                        api_name=api.name,
                        api_id=api.registry_id or ""
                    )
                else:
                    logger.warning(f"Connected to {api.name} but no data returned. Trying next API...")
                    continue
            else:
                if result.needs_auth:
                     return FetchResult(
                        success=False,
                        status=f"Authentication required for {api.name}",
                        needs_auth=True,
                        api_name=api.name,
                        api_id=api.registry_id or ""
                    )

        return FetchResult(success=False, status="No data could be fetched from any discovered API.")

    def _registry_fetch(
        self,
        api: 'DiscoveredAPI',
        user_query: str,
        api_key: Optional[str] = None
    ) -> Optional['FetchResult']:
        """
        Autonomously fetch data from a registry API using LLM parameter mapping.
        
        Uses the API's registry definition (endpoints, params, examples) and
        asks the LLM to map the user's query to the correct parameter values.
        Falls back to the endpoint's example if no LLM is available.
        
        Args:
            api: Discovered API (must have registry_id)
            user_query: Natural language data request
            api_key: Optional API key
            
        Returns:
            FetchResult with data if successful, None if this path can't handle it
        """
        if not api.registry_id:
            return None
        
        from mcp_server.api_registry import get_api
        api_def = get_api(api.registry_id)
        if not api_def or not api_def.endpoints:
            return None
        
        api_id = api.registry_id
        
        # -- Pick the best endpoint -----------------------------------------------
        endpoint = api_def.endpoints[0]  # Default to first
        if len(api_def.endpoints) > 1:
            # Ask LLM to choose the most relevant endpoint
            ep_descriptions = "\n".join(
                f"{i+1}. {ep.path} — {ep.description}"
                for i, ep in enumerate(api_def.endpoints)
            )
            try:
                from llm_manager.llm_interface import get_llm_completion
                choice = get_llm_completion(
                    f"Which API endpoint best answers this data request?\n"
                    f"Request: {user_query}\n\n"
                    f"Endpoints:\n{ep_descriptions}\n\n"
                    f"Reply with ONLY the endpoint number (e.g. 1).",
                    max_tokens=8, temperature=0.0
                ).strip()
                idx = int(choice) - 1
                if 0 <= idx < len(api_def.endpoints):
                    endpoint = api_def.endpoints[idx]
            except Exception:
                pass  # Keep default
        
        logger.info(f"Registry fetch {api_id}: endpoint={endpoint.path}")
        
        # -- Build request URL and parameters ------------------------------------
        base_url = api_def.base_url.rstrip("/")
        path = endpoint.path
        params = {}
        
        # Try LLM-driven parameter mapping
        llm_params = self._llm_map_params(api_def, endpoint, user_query)
        
        if llm_params:
            # LLM produced parameter values — use them
            # Some params might be path params (e.g. {year}, {api_key})
            for key, value in llm_params.items():
                placeholder = "{" + key + "}"
                if placeholder in path:
                    path = path.replace(placeholder, str(value))
                else:
                    params[key] = value
        elif endpoint.example:
            # No LLM available — parse the example URL as fallback
            logger.info(f"No LLM params, falling back to endpoint example")
            example = endpoint.example
            if "?" in example:
                example_path, query_string = example.split("?", 1)
                path = example_path
                for part in query_string.split("&"):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        params[k] = v
            else:
                path = example
        
        # Substitute {api_key} in path if present (e.g. ExchangeRate-API)
        if "{api_key}" in path and api_key:
            path = path.replace("{api_key}", api_key)
        
        # -- Add authentication ---------------------------------------------------
        headers = {}
        if api_key and api_def.auth_config:
            location = api_def.auth_config.get("location", "query")
            param_name = api_def.auth_config.get("param_name")
            
            if location == "query" and param_name:
                params[param_name] = api_key
            elif location == "header":
                header_name = api_def.auth_config.get("header_name", "Authorization")
                header_prefix = api_def.auth_config.get("prefix", "Bearer")
                headers[header_name] = f"{header_prefix} {api_key}"
            # "path" location already handled above via {api_key} substitution
        elif not api_key and api_def.auth_type not in ("none", "unknown"):
            # Try env var
            import os
            env_var = api_def.auth_config.get("env_var", "")
            env_key = os.environ.get(env_var, "") if env_var else ""
            if env_key:
                api_key = env_key
                location = api_def.auth_config.get("location", "query")
                param_name = api_def.auth_config.get("param_name")
                if location == "query" and param_name:
                    params[param_name] = api_key
                elif location == "header":
                    header_name = api_def.auth_config.get("header_name", "Authorization")
                    header_prefix = api_def.auth_config.get("prefix", "Bearer")
                    headers[header_name] = f"{header_prefix} {api_key}"
        
        # -- Resolve any remaining path placeholders with defaults ----------------
        import re
        unresolved = re.findall(r'\{(\w+)\}', path)
        if unresolved:
            # Sensible defaults for common path parameters
            defaults = {"year": "2022", "version": "v1", "format": "json"}
            for placeholder in unresolved:
                value = defaults.get(placeholder, "")
                if value:
                    path = path.replace("{" + placeholder + "}", value)
                    logger.info(f"Resolved {{{placeholder}}} to default: {value}")
                else:
                    logger.warning(f"Unresolved path placeholder: {{{placeholder}}}")
        
        # -- Make the HTTP request ------------------------------------------------
        url = f"{base_url}{path}"
        logger.info(f"Registry fetch {api_id}: GET {url} params={list(params.keys())}")
        
        try:
            import requests as req
            response = req.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # -- Parse response into DataFrame ------------------------------------
            import pandas as pd
            from preprocessing.data_cleaning import find_table_data
            
            table_data = find_table_data(data)
            
            if table_data:
                df = pd.DataFrame(table_data)
                if not df.empty:
                    return FetchResult(
                        success=True,
                        data=df.head(100).to_dict(orient="list"),
                        status=f"Fetched {len(df)} rows from {api_def.name} ({endpoint.path})",
                        api_name=api_def.name,
                        api_id=api_id
                    )
            
            # Census-style response: list-of-lists where first row = headers
            if isinstance(data, list) and len(data) >= 2 and isinstance(data[0], list):
                headers_row = data[0]
                rows = data[1:]
                df = pd.DataFrame(rows, columns=headers_row)
                if not df.empty:
                    return FetchResult(
                        success=True,
                        data=df.head(100).to_dict(orient="list"),
                        status=f"Fetched {len(df)} rows from {api_def.name} ({endpoint.path})",
                        api_name=api_def.name,
                        api_id=api_id
                    )
            
            logger.info(f"Registry fetch {api_id}: got response but could not parse into table")
            return None  # Let caller fall through to dynamic connector
            
        except Exception as e:
            logger.warning(f"Registry fetch {api_id} failed: {e}")
            return None
    
    def _llm_map_params(
        self,
        api_def: 'APIDefinition',
        endpoint: 'APIEndpoint',
        user_query: str
    ) -> Optional[Dict[str, Any]]:
        """
        Use the LLM to map a user query to API endpoint parameters.
        
        Returns a dict of param_name → value, or None if LLM is unavailable.
        """
        try:
            from llm_manager.llm_interface import get_llm_completion, is_llm_available
            
            if not is_llm_available():
                return None
            
            # Build a focused prompt with all context the LLM needs
            example_text = f"\nExample: {endpoint.example}" if endpoint.example else ""
            required_text = f"\nRequired params: {', '.join(endpoint.required_params)}" if endpoint.required_params else ""
            
            prompt = (
                f"You are an API parameter generator. Given a user's data request and an API endpoint, "
                f"generate the EXACT parameter values needed for the HTTP request.\n\n"
                f"API: {api_def.name}\n"
                f"Base URL: {api_def.base_url}\n"
                f"Endpoint: {endpoint.path}\n"
                f"Description: {endpoint.description}\n"
                f"Available params: {', '.join(endpoint.params)}{required_text}{example_text}\n\n"
                f"User request: {user_query}\n\n"
                f"Reply with ONLY a valid JSON object mapping parameter names to values. "
                f"Include path parameters (like {{year}}) as keys too. "
                f"Do NOT include any explanation, just the JSON object."
            )
            
            response = get_llm_completion(prompt, max_tokens=256, temperature=0.1)
            
            if not response:
                return None
            
            # Extract JSON from the response (handle markdown code fences)
            import json
            text = response.strip()
            if text.startswith("```"):
                # Strip code fences
                lines = text.split("\n")
                text = "\n".join(
                    line for line in lines 
                    if not line.strip().startswith("```")
                )
            
            # Find the JSON object in the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                params = json.loads(text[start:end])
                if isinstance(params, dict):
                    logger.info(f"LLM mapped params for {api_def.id}: {params}")
                    return params
            
            logger.warning(f"LLM response was not valid JSON: {text[:200]}")
            return None
            
        except Exception as e:
            logger.warning(f"LLM parameter mapping failed: {e}")
            return None

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
        
        # 2. Try each discovered API
        for api in apis[:3]:
            api_id = api.registry_id or api.name.lower().replace(' ', '_')
            logger.info(f"Trying {api.name} (id={api_id})...")
            
            # 2a. Registry-driven fetch: use endpoint definitions + LLM
            if api.registry_id:
                registry_result = self._registry_fetch(api, user_query, api_key)
                if registry_result and registry_result.success and registry_result.data:
                    return registry_result
                elif registry_result:
                    logger.info(f"Registry fetch partial: {registry_result.status}")
            
            # 2b. Dynamic connector fallback (OpenAPI generation)
            result = self.auto_connect(api, api_key)
            
            if result.success:
                if result.sample_data:
                    return FetchResult(
                        success=True,
                        data=result.sample_data,
                        status=f"Successfully fetched data from {api.name}",
                        api_name=api.name,
                        api_id=api_id
                    )
                elif result.needs_auth:
                    return FetchResult(
                        success=False,
                        needs_auth=True,
                        api_name=api.name,
                        api_id=api_id,
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
            api_id=best_api.registry_id or best_api.name.lower().replace(' ', '_'),
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

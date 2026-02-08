"""
Kaggle Source Analyzer - Dynamic vertical weighting from Kaggle datasets.

This module pulls dataset metadata from Kaggle's API to determine which
data sources are most commonly used per vertical/domain. This enables
data-driven API discovery rather than relying on hardcoded rankings.

Flow:
1. Query Kaggle API for popular datasets by tag (e.g., "finance", "economics")
2. Extract source URLs and names from dataset metadata
3. Count frequency to build weighted rankings
4. Cache results to avoid excessive API calls
5. Export weights for use by APIDiscoveryAgent

Requires: KAGGLE_USERNAME and KAGGLE_KEY environment variables
(or ~/.kaggle/kaggle.json credential file)
"""
import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import Counter

logger = logging.getLogger(__name__)

# Cache file for storing Kaggle-derived weights
CACHE_DIR = Path.home() / ".minerva" / "cache"
WEIGHTS_CACHE_FILE = CACHE_DIR / "kaggle_source_weights.json"
CACHE_TTL_HOURS = 24 * 7  # Refresh weekly


@dataclass
class SourceRanking:
    """Ranking of a data source within a vertical."""
    source_name: str
    frequency: int
    normalized_score: float = 0.0
    example_datasets: List[str] = field(default_factory=list)


@dataclass
class VerticalWeights:
    """Weights for a specific vertical/domain."""
    vertical: str
    sources: List[SourceRanking]
    total_datasets_analyzed: int
    last_updated: datetime
    
    def get_preferred(self, top_n: int = 5) -> List[str]:
        """Get top N preferred sources."""
        sorted_sources = sorted(self.sources, key=lambda x: x.frequency, reverse=True)
        return [s.source_name for s in sorted_sources[:top_n]]
    
    def to_dict(self) -> Dict:
        return {
            "vertical": self.vertical,
            "sources": [
                {
                    "name": s.source_name,
                    "frequency": s.frequency,
                    "score": s.normalized_score,
                    "examples": s.example_datasets[:3],
                }
                for s in self.sources
            ],
            "total_analyzed": self.total_datasets_analyzed,
            "last_updated": self.last_updated.isoformat(),
        }


# Known source patterns to extract from Kaggle dataset descriptions/sources
SOURCE_PATTERNS = {
    # Government & Economic
    "fred": [r"fred\.stlouisfed\.org", r"federal reserve", r"FRED"],
    "world_bank": [r"worldbank", r"world bank", r"data\.worldbank\.org"],
    "bls": [r"bls\.gov", r"bureau of labor statistics"],
    "census": [r"census\.gov", r"us census", r"american community survey"],
    "eurostat": [r"eurostat", r"ec\.europa\.eu"],
    "eia": [r"eia\.gov", r"energy information administration"],
    
    # Finance
    "alpha_vantage": [r"alphavantage", r"alpha vantage"],
    "yahoo_finance": [r"yahoo finance", r"finance\.yahoo\.com", r"yfinance"],
    "sec_edgar": [r"sec\.gov", r"edgar", r"SEC filings"],
    
    # Crypto
    "coingecko": [r"coingecko", r"coin gecko"],
    "coinmarketcap": [r"coinmarketcap", r"coin market cap"],
    
    # Weather & Environment
    "noaa": [r"noaa\.gov", r"NOAA", r"national ocean"],
    "openweathermap": [r"openweathermap", r"open weather"],
    "openaq": [r"openaq", r"air quality"],
    
    # Sports
    "espn": [r"espn\.com", r"ESPN"],
    "kaggle_sports": [r"sports-reference", r"basketball-reference"],
    
    # Movies/Entertainment
    "imdb": [r"imdb\.com", r"IMDB", r"internet movie database"],
    "tmdb": [r"themoviedb", r"TMDB"],
    
    # Social
    "twitter": [r"twitter\.com", r"Twitter API", r"tweets"],
    "reddit": [r"reddit\.com", r"Reddit API", r"pushshift"],
    
    # Health
    "who": [r"who\.int", r"World Health Organization", r"WHO"],
    "cdc": [r"cdc\.gov", r"CDC"],
    
    # Other
    "github": [r"github\.com", r"GitHub API"],
    "wikipedia": [r"wikipedia", r"wiki"],
}


# Mapping from Kaggle tags to our verticals
TAG_TO_VERTICAL = {
    # Finance
    "finance": "finance",
    "stock-market": "finance",
    "investing": "finance",
    "trading": "finance",
    "financial": "finance",
    
    # Economics
    "economics": "economics",
    "economic-data": "economics",
    "macroeconomics": "economics",
    
    # Weather
    "weather": "weather",
    "climate": "weather",
    "meteorology": "weather",
    
    # Environment
    "environment": "environment",
    "pollution": "environment",
    "air-quality": "environment",
    "emissions": "environment",
    
    # Health
    "health": "health",
    "healthcare": "health",
    "medical": "health",
    "covid-19": "health",
    "disease": "health",
    
    # Demographics
    "demographics": "demographics",
    "census": "demographics",
    "population": "demographics",
    
    # Sports
    "sports": "sports",
    "football": "sports",
    "basketball": "sports",
    "soccer": "sports",
    "baseball": "sports",
    
    # Crypto
    "cryptocurrency": "crypto",
    "bitcoin": "crypto",
    "ethereum": "crypto",
    "blockchain": "crypto",
    
    # Energy
    "energy": "energy",
    "oil-and-gas": "energy",
    "renewable-energy": "energy",
    
    # Food
    "food": "food",
    "nutrition": "food",
    
    # Movies
    "movies": "movies",
    "film": "movies",
    "entertainment": "movies",
    
    # Social
    "social-media": "social",
    "twitter": "social",
    "reddit": "social",
}


class KaggleSourceAnalyzer:
    """
    Analyzes Kaggle datasets to determine popular data sources per vertical.
    
    Uses the Kaggle API to:
    1. Fetch popular datasets by tag
    2. Extract source information from descriptions and metadata
    3. Build frequency-based rankings
    4. Cache results for efficiency
    """
    
    def __init__(self, master_password: Optional[str] = None):
        self._kaggle_api = None
        self._master_password = master_password
        self._weights_cache: Dict[str, VerticalWeights] = {}
        self._load_cache()
    
    def set_master_password(self, password: str):
        """
        Set master password for credential decryption.
        
        This also clears any cached API instance so the next call
        will retry authentication with the new password.
        """
        if password != self._master_password:
            self._master_password = password
            # Clear cached API to force re-authentication with new password
            self._kaggle_api = None
            logger.info("Master password updated, will retry Kaggle auth on next API call")

    def clear_credentials(self):
        """
        Clear master password and API instance.
        
        Call this to lock the analyzer's access to Kaggle.
        """
        self._master_password = None
        self._kaggle_api = None
        # Also clear env vars to be safe
        if "KAGGLE_USERNAME" in os.environ:
            del os.environ["KAGGLE_USERNAME"]
        if "KAGGLE_KEY" in os.environ:
            del os.environ["KAGGLE_KEY"]
        logger.info("Credentials cleared, Kaggle access locked")
    
    def _get_kaggle_api(self):
        """
        Lazy-load Kaggle API with credential manager integration.
        
        Credential priority:
        1. Minerva encrypted storage (~/.minerva/credentials.json)
        2. Standard Kaggle env vars (KAGGLE_USERNAME, KAGGLE_KEY)
        3. Standard Kaggle file (~/.kaggle/kaggle.json)
        
        IMPORTANT: The Kaggle package checks credentials during import,
        so we MUST set environment variables BEFORE importing KaggleApi!
        """
        if self._kaggle_api is not None:
            return self._kaggle_api
        
        logger.debug(f"_get_kaggle_api called, master_password={'SET' if self._master_password else 'NOT SET'}")
        
        # CRITICAL: Set credentials BEFORE importing Kaggle API
        # The Kaggle package validates credentials during import!
        if self._master_password:
            logger.info("Loading Kaggle credentials from Minerva storage...")
            try:
                from mcp_server.credential_manager import get_kaggle_credentials
                creds = get_kaggle_credentials(self._master_password)
                
                if creds:
                    source = creds.get("source", "unknown")
                    logger.info(f"Got Kaggle credentials from {source} for user: {creds.get('username')}")
                    
                    # Set env vars BEFORE importing Kaggle API
                    os.environ["KAGGLE_USERNAME"] = creds["username"]
                    os.environ["KAGGLE_KEY"] = creds["key"]
                else:
                    logger.warning("Kaggle credential decryption failed - check master password")
            except Exception as e:
                logger.warning(f"Could not load Kaggle credentials: {e}")
        else:
            logger.debug("No master password provided, skipping Minerva credential check")
        
        # Now import and authenticate Kaggle API
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            api = KaggleApi()
            api.authenticate()
            self._kaggle_api = api
            logger.info("Kaggle API authenticated successfully")
            return api
            
        except ImportError:
            logger.warning("kaggle package not installed. Run: pip install kaggle")
            return None
        except Exception as e:
            logger.warning(f"Kaggle API auth failed: {e}")
            return None
    
    def _load_cache(self):
        """Load cached weights from disk."""
        if WEIGHTS_CACHE_FILE.exists():
            try:
                with open(WEIGHTS_CACHE_FILE, "r") as f:
                    data = json.load(f)
                
                for vertical, weights_data in data.items():
                    sources = [
                        SourceRanking(
                            source_name=s["name"],
                            frequency=s["frequency"],
                            normalized_score=s.get("score", 0),
                            example_datasets=s.get("examples", []),
                        )
                        for s in weights_data.get("sources", [])
                    ]
                    
                    self._weights_cache[vertical] = VerticalWeights(
                        vertical=vertical,
                        sources=sources,
                        total_datasets_analyzed=weights_data.get("total_analyzed", 0),
                        last_updated=datetime.fromisoformat(weights_data.get("last_updated", datetime.now().isoformat())),
                    )
                
                logger.info(f"Loaded cached weights for {len(self._weights_cache)} verticals")
            except Exception as e:
                logger.warning(f"Failed to load weights cache: {e}")
    
    def _save_cache(self):
        """Save weights cache to disk."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        try:
            data = {v: w.to_dict() for v, w in self._weights_cache.items()}
            with open(WEIGHTS_CACHE_FILE, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved weights cache for {len(data)} verticals")
        except Exception as e:
            logger.warning(f"Failed to save weights cache: {e}")
    
    def _is_cache_valid(self, vertical: str) -> bool:
        """Check if cached weights for a vertical are still valid."""
        if vertical not in self._weights_cache:
            return False
        
        weights = self._weights_cache[vertical]
        age = datetime.now() - weights.last_updated
        return age < timedelta(hours=CACHE_TTL_HOURS)
    
    def _extract_sources(self, text: str) -> List[str]:
        """Extract known data sources from text using pattern matching."""
        sources = []
        text_lower = text.lower()
        
        for source_id, patterns in SOURCE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    sources.append(source_id)
                    break
        
        return sources
    
    def analyze_vertical(self, vertical: str, max_datasets: int = 100) -> Optional[VerticalWeights]:
        """
        Analyze Kaggle datasets for a vertical to find popular sources.
        
        Args:
            vertical: The vertical to analyze (e.g., "finance", "health")
            max_datasets: Maximum number of datasets to analyze
            
        Returns:
            VerticalWeights with source rankings, or None if analysis fails
        """
        # Check cache first
        if self._is_cache_valid(vertical):
            logger.info(f"Using cached weights for '{vertical}'")
            return self._weights_cache[vertical]
        
        # Find Kaggle tags for this vertical
        tags = [tag for tag, v in TAG_TO_VERTICAL.items() if v == vertical]
        if not tags:
            tags = [vertical]  # Use vertical name as tag if no mapping
        
        api = self._get_kaggle_api()
        if api is None:
            # Fallback to cached or default weights
            if vertical in self._weights_cache:
                return self._weights_cache[vertical]
            return None
        
        # Collect sources from datasets
        source_counter: Counter = Counter()
        source_examples: Dict[str, List[str]] = {}
        total_datasets = 0
        
        for tag in tags[:3]:  # Limit tags to avoid too many API calls
            try:
                datasets = api.dataset_list(
                    search=tag,
                    sort_by="hottest",
                    page=1,
                    max_size=1000000000,  # No size limit
                )
                
                for dataset in datasets[:max_datasets // len(tags)]:
                    total_datasets += 1
                    
                    # Combine description and other metadata for analysis
                    text_to_analyze = f"{dataset.title} {dataset.subtitle or ''}"
                    
                    # Try to get full description if available
                    try:
                        # Note: This makes an additional API call per dataset
                        # Only do this for top datasets
                        if total_datasets <= 20:
                            full_dataset = api.dataset_view(dataset.ref)
                            if hasattr(full_dataset, 'description'):
                                text_to_analyze += f" {full_dataset.description}"
                    except Exception:
                        pass
                    
                    # Extract sources
                    found_sources = self._extract_sources(text_to_analyze)
                    
                    for source in found_sources:
                        source_counter[source] += 1
                        if source not in source_examples:
                            source_examples[source] = []
                        if len(source_examples[source]) < 5:
                            source_examples[source].append(dataset.title)
                            
            except Exception as e:
                logger.warning(f"Error fetching datasets for tag '{tag}': {e}")
        
        if not source_counter:
            logger.warning(f"No sources found for vertical '{vertical}'")
            return None
        
        # Normalize scores
        max_freq = max(source_counter.values())
        
        sources = [
            SourceRanking(
                source_name=source,
                frequency=freq,
                normalized_score=freq / max_freq,
                example_datasets=source_examples.get(source, []),
            )
            for source, freq in source_counter.most_common()
        ]
        
        weights = VerticalWeights(
            vertical=vertical,
            sources=sources,
            total_datasets_analyzed=total_datasets,
            last_updated=datetime.now(),
        )
        
        # Cache results
        self._weights_cache[vertical] = weights
        self._save_cache()
        
        logger.info(f"Analyzed {total_datasets} datasets for '{vertical}', found {len(sources)} sources")
        
        return weights
    
    def get_weights_for_vertical(self, vertical: str) -> Dict[str, float]:
        """
        Get source weights for a vertical as a simple dict.
        
        Returns:
            Dict mapping source_id to weight (0.0-1.0)
        """
        weights = self.analyze_vertical(vertical)
        
        if weights is None:
            return {}
        
        return {
            s.source_name: s.normalized_score
            for s in weights.sources
        }
    
    def get_preferred_sources(self, vertical: str, top_n: int = 5) -> List[str]:
        """Get top N preferred sources for a vertical."""
        weights = self.analyze_vertical(vertical)
        
        if weights is None:
            return []
        
        return weights.get_preferred(top_n)
    
    def refresh_all_verticals(self) -> Dict[str, VerticalWeights]:
        """
        Refresh weights for all known verticals.
        
        Call this periodically (e.g., weekly) to update rankings.
        """
        verticals = set(TAG_TO_VERTICAL.values())
        results = {}
        
        for vertical in verticals:
            logger.info(f"Refreshing weights for '{vertical}'...")
            weights = self.analyze_vertical(vertical)
            if weights:
                results[vertical] = weights
        
        return results
    
    def export_for_discovery_agent(self) -> Dict[str, Dict]:
        """
        Export weights in format expected by APIDiscoveryAgent.VERTICAL_SOURCES.
        
        Returns:
            Dict in the format:
            {
                "finance": {
                    "preferred": ["alpha_vantage", "fred", ...],
                    "boost": 0.3,
                    "kaggle_derived": True,
                },
                ...
            }
        """
        result = {}
        
        for vertical in set(TAG_TO_VERTICAL.values()):
            weights = self.analyze_vertical(vertical)
            
            if weights:
                result[vertical] = {
                    "preferred": weights.get_preferred(5),
                    "boost": 0.3,
                    "kaggle_derived": True,
                    "datasets_analyzed": weights.total_datasets_analyzed,
                    "last_updated": weights.last_updated.isoformat(),
                }
        
        return result


# Singleton instance
_analyzer: Optional[KaggleSourceAnalyzer] = None


def get_kaggle_analyzer() -> KaggleSourceAnalyzer:
    """Get the global Kaggle source analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = KaggleSourceAnalyzer()
    return _analyzer


def refresh_kaggle_weights() -> Dict[str, VerticalWeights]:
    """
    Convenience function to refresh all Kaggle weights.
    
    Can be called from a scheduled job or manually.
    """
    return get_kaggle_analyzer().refresh_all_verticals()


def get_vertical_sources(vertical: str) -> List[str]:
    """
    Get preferred data sources for a vertical.
    
    Example:
        sources = get_vertical_sources("finance")
        # ["alpha_vantage", "yahoo_finance", "fred", "sec_edgar"]
    """
    return get_kaggle_analyzer().get_preferred_sources(vertical)

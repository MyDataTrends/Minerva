"""
Semantic API Search - Embedding-based intent matching with LLM fallback.

Uses vector similarity to match user queries to APIs, with a fallback
to LLM interpretation for low-confidence matches or novel queries.

Architecture designed for future expansion:
- Current: Match against curated registry of 10 APIs
- Future: Low-confidence queries trigger dynamic API discovery/generation
"""
import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Confidence thresholds
HIGH_CONFIDENCE = 0.85  # Direct match, use without LLM
LOW_CONFIDENCE = 0.60   # Below this, LLM interpretation or discovery needed


@dataclass
class APIMatch:
    """Result of semantic API matching."""
    api_id: str
    name: str
    description: str
    score: float  # Cosine similarity, 0-1
    confidence: str  # "high", "medium", "low"
    matched_via: str  # "embedding", "llm", "keyword"


class SemanticAPIRouter:
    """
    Semantic router for matching user queries to APIs.
    
    Uses embeddings for fast similarity matching, with optional
    LLM fallback for ambiguous queries.
    """
    
    def __init__(self, use_llm_fallback: bool = True):
        self.use_llm_fallback = use_llm_fallback
        self._embedding_model = None
        self._api_embeddings: Dict[str, np.ndarray] = {}
        self._api_descriptions: Dict[str, str] = {}
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization of embedding model and API vectors."""
        if self._initialized:
            return
        
        try:
            from learning.embeddings import EmbeddingModel
            self._embedding_model = EmbeddingModel()
            self._build_api_index()
            self._initialized = True
            logger.info(f"Semantic API router initialized with {len(self._api_embeddings)} APIs")
        except Exception as e:
            logger.warning(f"Failed to initialize embedding model: {e}")
            self._initialized = False
    
    def _build_api_index(self):
        """Build embedding index for all APIs in registry."""
        from mcp_server.api_registry import get_all_apis
        
        apis = get_all_apis()
        
        for api in apis:
            # Create a rich description for embedding
            description = self._create_api_description(api)
            self._api_descriptions[api.id] = description
        
        # Batch embed all descriptions
        if self._api_descriptions:
            descriptions = list(self._api_descriptions.values())
            api_ids = list(self._api_descriptions.keys())
            
            embeddings = self._embedding_model.embed(descriptions)
            
            for i, api_id in enumerate(api_ids):
                self._api_embeddings[api_id] = embeddings[i]
    
    def _create_api_description(self, api) -> str:
        """Create a rich text description for embedding."""
        parts = [
            api.name,
            api.description,
            "Data types: " + ", ".join(api.data_types),
            "Keywords: " + ", ".join(api.keywords[:10]),  # Limit keywords
        ]
        return " | ".join(parts)
    
    def add_api(self, api_id: str, api):
        """Dynamically add a new API to the index (for future dynamic generation)."""
        self._ensure_initialized()
        
        if not self._embedding_model:
            logger.warning("Cannot add API - embedding model not available")
            return
        
        description = self._create_api_description(api)
        self._api_descriptions[api_id] = description
        embedding = self._embedding_model.embed([description])[0]
        self._api_embeddings[api_id] = embedding
        
        logger.info(f"Added API '{api_id}' to semantic index")
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        llm_fallback: bool = None
    ) -> List[APIMatch]:
        """
        Search for APIs matching a natural language query.
        
        Args:
            query: User's query (e.g., "US unemployment trends")
            top_k: Maximum number of results
            llm_fallback: Override default LLM fallback behavior
            
        Returns:
            List of APIMatch objects sorted by relevance
        """
        self._ensure_initialized()
        
        use_fallback = llm_fallback if llm_fallback is not None else self.use_llm_fallback
        
        # If embedding model unavailable, fall back to keyword search
        if not self._embedding_model:
            return self._keyword_search(query, top_k)
        
        # Embed the query
        query_embedding = self._embedding_model.embed_query(query)
        
        # Calculate cosine similarity to all APIs
        scores = []
        for api_id, api_embedding in self._api_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, api_embedding)
            scores.append((api_id, similarity))
        
        # Sort by similarity (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        from mcp_server.api_registry import get_api
        
        results = []
        for api_id, score in scores[:top_k]:
            api = get_api(api_id)
            if not api:
                continue
            
            # Determine confidence level
            if score >= HIGH_CONFIDENCE:
                confidence = "high"
            elif score >= LOW_CONFIDENCE:
                confidence = "medium"
            else:
                confidence = "low"
            
            results.append(APIMatch(
                api_id=api_id,
                name=api.name,
                description=api.description,
                score=float(score),
                confidence=confidence,
                matched_via="embedding",
            ))
        
        # If top match is low confidence and LLM fallback enabled, try LLM
        if results and results[0].confidence == "low" and use_fallback:
            llm_result = self._llm_interpret(query, results)
            if llm_result:
                # Prepend LLM result if it's different from top embedding match
                if llm_result.api_id != results[0].api_id:
                    results.insert(0, llm_result)
        
        return results
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def _keyword_search(self, query: str, top_k: int) -> List[APIMatch]:
        """Fallback keyword-based search when embeddings unavailable."""
        from mcp_server.api_registry import search_apis_by_query, get_api
        
        matches = search_apis_by_query(query)
        
        results = []
        for match in matches[:top_k]:
            api = get_api(match["api_id"])
            if api:
                # Normalize score to 0-1 range (original is 0-100ish)
                normalized_score = min(match["score"] / 50.0, 1.0)
                results.append(APIMatch(
                    api_id=match["api_id"],
                    name=api.name,
                    description=api.description,
                    score=normalized_score,
                    confidence="medium" if normalized_score > 0.5 else "low",
                    matched_via="keyword",
                ))
        
        return results
    
    def _llm_interpret(self, query: str, candidates: List[APIMatch]) -> Optional[APIMatch]:
        """
        Use LLM to interpret a query when embedding confidence is low.
        
        This is the bridge to future dynamic discovery - if LLM can't
        find a match in candidates, it could trigger API generation.
        """
        try:
            from llm_manager.llm_interface import get_llm_completion
            from mcp_server.api_registry import get_api
            
            # Build prompt with candidate APIs
            candidate_list = "\n".join([
                f"- {c.api_id}: {c.name} - {c.description}"
                for c in candidates[:5]
            ])
            
            prompt = f"""Given this user query about data:
"{query}"

Which of these APIs would best provide the requested data?

Available APIs:
{candidate_list}

Respond with ONLY the api_id (e.g., "fred" or "world_bank"). 
If none of these APIs can provide the data, respond with "NONE".
"""
            
            response = get_llm_completion(prompt, max_tokens=20, temperature=0.0)
            
            if response:
                api_id = response.strip().lower().replace('"', '').replace("'", "")
                
                if api_id == "none":
                    # Future: trigger dynamic discovery here
                    logger.info(f"LLM found no match for query: {query}")
                    return None
                
                # Find the matched API
                api = get_api(api_id)
                if api:
                    return APIMatch(
                        api_id=api_id,
                        name=api.name,
                        description=api.description,
                        score=0.90,  # LLM match gets high score
                        confidence="high",
                        matched_via="llm",
                    )
            
        except Exception as e:
            logger.warning(f"LLM fallback failed: {e}")
        
        return None
    
    def get_confidence_report(self, query: str) -> Dict[str, Any]:
        """
        Get detailed matching report for debugging/transparency.
        
        Useful for understanding why a particular API was selected.
        """
        results = self.search(query, top_k=5, llm_fallback=False)
        
        return {
            "query": query,
            "matches": [
                {
                    "api_id": r.api_id,
                    "name": r.name,
                    "score": round(r.score, 4),
                    "confidence": r.confidence,
                    "matched_via": r.matched_via,
                }
                for r in results
            ],
            "top_match": results[0].api_id if results else None,
            "would_use_llm": results[0].confidence == "low" if results else True,
            "thresholds": {
                "high_confidence": HIGH_CONFIDENCE,
                "low_confidence": LOW_CONFIDENCE,
            }
        }


# =============================================================================
# Module-level convenience functions
# =============================================================================

_router: Optional[SemanticAPIRouter] = None


def get_router() -> SemanticAPIRouter:
    """Get or create the global semantic router instance."""
    global _router
    if _router is None:
        _router = SemanticAPIRouter()
    return _router


def semantic_search_apis(query: str, top_k: int = 5) -> List[APIMatch]:
    """Convenience function for semantic API search."""
    return get_router().search(query, top_k)


def get_best_api_for_query(query: str) -> Optional[APIMatch]:
    """Get the single best API match for a query."""
    results = get_router().search(query, top_k=1)
    return results[0] if results else None

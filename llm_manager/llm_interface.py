"""
LLM Interface - Bridge between LLM Manager and existing code.

Provides a unified interface that:
1. Uses LLM Manager's active model when available
2. Falls back to existing singleton pattern
3. Exposes simple completion/chat methods

This allows gradual migration from the old LLM loading pattern.
"""
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


def get_llm_completion(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """
    Get a completion from the active LLM.
    
    Tries LLM Manager first, falls back to existing singleton.
    
    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated text, or empty string if no LLM available
    """
    # Try LLM Manager first
    try:
        from llm_manager.registry import get_registry
        registry = get_registry()
        active_model = registry.get_active_model()
        
        # Auto-select first available model if none is active
        if active_model is None:
            local_models = registry.get_local_models()
            if local_models:
                first_model = local_models[0]
                logger.info(f"Auto-selecting model: {first_model.name}")
                registry.set_active_model(first_model.id)
                active_model = first_model
        
        if active_model is not None:
            logger.debug(f"Active model configured: {active_model.name}")
            
            # Get cached provider (doesn't auto-load)
            provider = registry.get_active_provider(auto_load=False)
            
            # If not loaded, try to load now
            if provider is None:
                logger.info(f"Model not loaded, attempting to load: {active_model.name}")
                provider = registry.load_active_model()
            
            if provider is not None:
                logger.debug(f"Using LLM Manager provider: {active_model.name}")
                result = provider.complete(prompt, max_tokens, temperature, **kwargs)
                if result:
                    return result
            else:
                logger.warning(f"Could not load {active_model.name}")
    except Exception as e:
        logger.error(f"LLM Manager error: {e}")
    
    # Fallback: Only try subprocess if NO active model is configured (first run scenario)
    if not active_model:
        try:
            from pathlib import Path
            from llm_manager.subprocess_manager import get_llm_subprocess
            
            model_dir = Path(__file__).resolve().parents[1] / "adm" / "llm_backends" / "local_model"
            gguf_files = list(model_dir.glob("*.gguf"))
            
            if gguf_files:
                logger.info(f"No active model config, falling back to found model: {gguf_files[0].name}")
                subprocess_mgr = get_llm_subprocess()
                if subprocess_mgr.load_model(str(gguf_files[0])):
                    result = subprocess_mgr.complete(prompt, max_tokens=max_tokens, temperature=temperature)
                    if result:
                        return result
        except Exception as e:
            logger.error(f"Subprocess fallback error: {e}")
    
    return ""


def get_llm_chat(
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """
    Get a chat response from the active LLM.
    
    Args:
        messages: List of {"role": "...", "content": "..."} dicts
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Assistant's response text
    """
    # Try LLM Manager first
    try:
        from llm_manager.registry import get_registry
        registry = get_registry()
        provider = registry.get_active_provider()
        
        if provider is not None:
            return provider.chat(messages, max_tokens, temperature, **kwargs)
    except Exception as e:
        logger.debug(f"LLM Manager not available: {e}")
    
    # Fallback: convert to prompt
    prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
    prompt += "\nASSISTANT:"
    return get_llm_completion(prompt, max_tokens, temperature, **kwargs)


def is_llm_available() -> bool:
    """
    Check if any LLM is available (configured or discoverable).
    
    This checks:
    1. If an active model is selected in the registry
    2. If any gguf files exist in the default location
    
    Loading happens lazily when the user actually runs a query.
    """
    # Check if gguf files exist (most reliable check)
    try:
        from pathlib import Path
        model_dir = Path(__file__).resolve().parents[1] / "adm" / "llm_backends" / "local_model"
        if model_dir.exists() and list(model_dir.glob("*.gguf")):
            return True
    except Exception:
        pass
    
    # Check LLM Manager config
    try:
        from llm_manager.registry import get_registry
        registry = get_registry()
        if registry.get_active_model() is not None:
            return True
        # Also check if any local models are registered
        if registry.get_local_models():
            return True
    except Exception:
        pass
    
    return False


def get_active_model_name() -> str:
    """Get the name of the active model."""
    try:
        from llm_manager.registry import get_registry
        model = get_registry().get_active_model()
        if model:
            return model.name
    except Exception:
        pass
    
    return "Local LLM" if is_llm_available() else "None"


# Analysis-specific helpers

def analyze_data_with_llm(df, question: str) -> str:
    """
    Use LLM to analyze data and answer a question.
    
    Args:
        df: DataFrame to analyze
        question: Natural language question about the data
        
    Returns:
        LLM's analysis response
    """
    # Build context about the data
    stats_summary = df.describe().to_string()
    columns = ", ".join(df.columns.tolist())
    sample = df.head(5).to_string()
    
    prompt = f"""You are a data analyst. Analyze this dataset and answer the question.

DATASET INFO:
- Columns: {columns}
- Shape: {df.shape[0]} rows x {df.shape[1]} columns

SAMPLE DATA:
{sample}

STATISTICS:
{stats_summary}

QUESTION: {question}

Provide a clear, concise analysis based on the data:"""

    return get_llm_completion(prompt, max_tokens=1024, temperature=0.3)


def suggest_visualizations(df) -> List[Dict[str, Any]]:
    """
    Use LLM to suggest visualizations for a dataset.
    
    Returns list of visualization suggestions with type and columns.
    """
    import json
    import re
    
    columns = df.columns.tolist()
    dtypes = df.dtypes.to_dict()
    dtype_str = ", ".join(f"{col}: {str(dtype)}" for col, dtype in dtypes.items())

    try:
        sample_data = df.head(3).to_string(index=False)
        # Limit description to save tokens
        stats = df.describe().to_string() 
    except Exception:
        sample_data = "N/A"
        stats = "N/A"
    
    prompt = f"""Suggest 3 visualizations for this dataset based on its actual content patterns.

COLUMNS: {columns}
TYPES: {dtype_str}

DATA SAMPLES:
{sample_data}

STATISTICS:
{stats}

Return ONLY a JSON array, no other text. Format:
[{{"type": "bar", "x": "column_name", "y": "column_name", "title": "Chart title", "reason": "Why this visualization based on the data values"}}]

Valid types: bar, line, scatter, histogram, pie

JSON:"""

    response = get_llm_completion(prompt, max_tokens=600, temperature=0.3)
    
    if not response or len(response) < 5:
        logger.warning("suggest_visualizations: Empty or very short LLM response")
        return []
    
    logger.debug(f"suggest_visualizations raw response: {response[:500]}")
    
    try:
        # Strategy 1: Direct JSON parse (if response is pure JSON)
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract JSON array from response
        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            return json.loads(json_str)
        
        # Strategy 3: Look for individual JSON objects
        objects = re.findall(r'\{[^{}]+\}', response)
        if objects:
            parsed = []
            for obj_str in objects[:3]:  # Max 3
                try:
                    obj = json.loads(obj_str)
                    if 'type' in obj and 'x' in obj:
                        parsed.append(obj)
                except json.JSONDecodeError:
                    continue
            if parsed:
                return parsed
        
        logger.warning(f"suggest_visualizations: Could not parse JSON from response")
        return []
        
    except Exception as e:
        logger.error(f"suggest_visualizations error: {e}")
        return []


def generate_insight(df, chart_type: str, x_col: str, y_col: str = None) -> str:
    """
    Generate a natural language insight for a visualization.
    
    Args:
        df: DataFrame
        chart_type: Type of chart (bar, line, scatter, etc.)
        x_col: X-axis column
        y_col: Y-axis column (optional)
        
    Returns:
        Natural language insight about the visualization
    """
    # Get relevant statistics
    if y_col:
        if x_col in df.columns and y_col in df.columns:
            correlation = df[x_col].corr(df[y_col]) if df[x_col].dtype.kind in 'ifc' and df[y_col].dtype.kind in 'ifc' else None
            x_stats = df[x_col].describe().to_dict() if df[x_col].dtype.kind in 'ifc' else {}
            y_stats = df[y_col].describe().to_dict() if df[y_col].dtype.kind in 'ifc' else {}
        else:
            correlation = None
            x_stats = {}
            y_stats = {}
    else:
        correlation = None
        x_stats = df[x_col].describe().to_dict() if x_col in df.columns and df[x_col].dtype.kind in 'ifc' else {}
        y_stats = {}
    
    prompt = f"""Generate a brief insight for this visualization:

Chart Type: {chart_type}
X-axis: {x_col}
Y-axis: {y_col or 'N/A'}
Correlation: {correlation if correlation else 'N/A'}
X Stats: {x_stats}
Y Stats: {y_stats}

Write 1-2 sentences highlighting the key insight a user should notice:"""

    return get_llm_completion(prompt, max_tokens=150, temperature=0.5)

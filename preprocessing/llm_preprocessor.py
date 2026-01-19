"""Utilities for preprocessing data with a local LLM backend."""

import json
import logging
import time
import threading
from collections import OrderedDict
from pathlib import Path
import pandas as pd

# from sklearn.metrics.pairwise import cosine_similarity - moved to local import

from preprocessing.prompt_templates import generate_prompt
from preprocessing.metadata_parser import parse_metadata
from preprocessing.sanitize import redact
from config import (
    LLM_CACHE_TTL,
    LLM_MAX_CONCURRENCY,
    LLM_TOKEN_BUDGET,
    LLM_TEMPERATURE,
    LLM_MAX_INPUT_CHARS,
    LLM_NON_PRINTABLE_THRESHOLD,
    AUTO_DOWNLOAD_LLM,
    LLM_REPO_ID,
    LLM_FILENAME,
    LLM_PROVIDER,
    MISTRAL_MODEL_PATH,
    LLM_HF_REPO_ID,
)

# Lazy load placeholder
Llama = None

LLM_INFO_PATH = Path(__file__).resolve().parents[1] / "adm" / "llm_backends" / "local_model" / "model_info.json"

_logger = logging.getLogger(__name__)
_warned_deprecation = False


_CACHE_MAXSIZE = 128
_llm_cache = OrderedDict()
_cache_lock = threading.Lock()
_semaphore = threading.BoundedSemaphore(LLM_MAX_CONCURRENCY)

# Singleton pattern for LLM model instance - prevents loading model for every call
_llm_instance = None
_llm_instance_lock = threading.Lock()
_llm_load_attempted = False


def _cached_llm_call(key: tuple, call):
    now = time.time()
    with _cache_lock:
        if key in _llm_cache:
            ts, value = _llm_cache[key]
            if now - ts < LLM_CACHE_TTL:
                return value
            del _llm_cache[key]
    with _semaphore:
        result = call()
    with _cache_lock:
        _llm_cache[key] = (time.time(), result)
        # purge expired
        for k in list(_llm_cache.keys()):
            ts, _ = _llm_cache[k]
            if time.time() - ts >= LLM_CACHE_TTL:
                del _llm_cache[k]
        while len(_llm_cache) > _CACHE_MAXSIZE:
            _llm_cache.popitem(last=False)
    return result


def _load_llm_once(info_path: Path = LLM_INFO_PATH):
    """Load the LLM model once (internal helper for singleton pattern)."""
    # Lazy import
    global Llama
    if Llama is None:
        try:
            from llama_cpp import Llama as _Llama
            Llama = _Llama
        except ImportError:
            _logger.warning("llama_cpp not installed")
            return None
        except Exception as e:
            _logger.warning(f"llama_cpp import failed: {e}")
            return None

    # Safer loading parameters
    load_kwargs = {
        "n_ctx": 2048,       # Limited context for stability
        "n_gpu_layers": 20,  # Hybrid offload (safe for RTX 3070 8GB + Desktop)
        "verbose": False,
    }
    
    # Strategy 1: Use MISTRAL_MODEL_PATH if set and exists
    if MISTRAL_MODEL_PATH:
        model_path = Path(MISTRAL_MODEL_PATH)
        if model_path.exists():
            _logger.info(f"Loading LLM from MISTRAL_MODEL_PATH: {model_path}")
            try:
                return Llama(model_path=str(model_path), **load_kwargs)
            except Exception as e:
                _logger.error(f"Failed to load model: {e}")
                return None
    
    # Strategy 2: Look for any .gguf file in the local_model directory
    try:
        model_dir = info_path.parent
        gguf_files = list(model_dir.glob("*.gguf"))
        if gguf_files:
            model_file = gguf_files[0]  # Use first found
            _logger.info(f"Loading LLM from: {model_file}")
            try:
                return Llama(model_path=str(model_file), **load_kwargs)
            except Exception as e:
                _logger.error(f"Failed to load model: {e}")
                return None
    except Exception:
        pass
    
    # Strategy 3: Try model_info.json naming convention
    try:
        info = json.loads(info_path.read_text())
        model_file = info_path.parent / f"{info['model_name']}.{info['quantization']}.{info['format']}"
        if model_file.exists():
            try:
                return Llama(model_path=str(model_file), **load_kwargs)
            except Exception as e:
                _logger.error(f"Failed to load model: {e}")
                return None
    except Exception:
        pass
    
    # Strategy 4: Auto-download if enabled (skip if we already have a model)
    if AUTO_DOWNLOAD_LLM:
        try:
            from huggingface_hub import hf_hub_download
            dl_path = hf_hub_download(repo_id=LLM_REPO_ID, filename=LLM_FILENAME)
            return Llama(model_path=str(dl_path), **load_kwargs)
        except Exception:
            return None
    return None


def load_local_llm(info_path: Path = LLM_INFO_PATH):
    """
    Load the local LLM if available (singleton pattern).
    
    The model is loaded once and cached for all subsequent calls.
    This prevents the massive memory overhead of loading the model
    multiple times during testing or normal operation.
    """
    global _llm_instance, _llm_load_attempted
    
    # Fast path: already loaded
    if _llm_instance is not None:
        return _llm_instance
    
    # Fast path: already tried and failed
    if _llm_load_attempted:
        return None
    
    with _llm_instance_lock:
        # Double-check after acquiring lock
        if _llm_instance is not None:
            return _llm_instance
        if _llm_load_attempted:
            return None
        
        _llm_load_attempted = True
        _llm_instance = _load_llm_once(info_path)
        
        if _llm_instance is not None:
            _logger.info("LLM model loaded successfully (singleton)")
        else:
            _logger.debug("No LLM model available")
        
        return _llm_instance


def unload_llm():
    """
    Unload the cached LLM instance to free memory.
    
    Call this when you're done with LLM operations and want to reclaim RAM.
    Useful in tests or long-running processes that need to free resources.
    """
    global _llm_instance, _llm_load_attempted
    with _llm_instance_lock:
        if _llm_instance is not None:
            _logger.info("Unloading LLM model to free memory")
            del _llm_instance
            _llm_instance = None
        _llm_load_attempted = False


def _guard_input(text: str) -> str:
    red = redact(text)
    if not red:
        return red
    total = len(red)
    non_printables = sum(
        1 for c in red if not c.isprintable() and c not in "\n\r\t"
    )
    if total and (non_printables / total) > LLM_NON_PRINTABLE_THRESHOLD:
        _logger.warning(
            "LLM input rejected due to non-printable ratio %.2f",
            non_printables / total,
        )
        raise ValueError("Input appears to be binary or high-entropy data")
    if total > LLM_MAX_INPUT_CHARS:
        truncated = red[:LLM_MAX_INPUT_CHARS] + "[TRUNCATED]"
        _logger.warning("LLM input truncated: %s", truncated)
        return truncated
    return red


def llm_completion(prompt: str, max_tokens: int = LLM_TOKEN_BUDGET) -> str:
    try:
        prompt = _guard_input(prompt)
    except ValueError:
        return "LLM input rejected"

    def _call():
        llm = load_local_llm()
        if llm is None:
            return "LLM unavailable"
        try:
            output = llm.create_completion(prompt=prompt, max_tokens=max_tokens)
            return output["choices"][0]["text"].strip()
        except Exception as exc:  # pragma: no cover - runtime issues
            return f"LLM error: {exc}"

    key = ("llm_completion", prompt, max_tokens)
    return _cached_llm_call(key, _call)


class _HFWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def create_completion(self, prompt: str, max_tokens: int, temperature: float):
        from torch import no_grad
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        with no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True if temperature and temperature > 0 else False,
                temperature=float(temperature) if temperature is not None else 0.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True)
        return {"choices": [{"text": text}]}


def load_transformers_model(repo_id: str | None = None):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception:
        return None
    try:
        rid = repo_id or LLM_HF_REPO_ID
        tok = AutoTokenizer.from_pretrained(rid, use_fast=True)
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(rid)
        mdl.eval()
        return _HFWrapper(mdl, tok)
    except Exception:
        return None


def preprocess_data_with_llm(data, task: str = "text-classification"):
    """Deprecated LLM-based preprocessing.

    The function now acts as a pass-through to preserve backward compatibility
    while avoiding any LLM calls. A one-time info log is emitted when used with
    a DataFrame.
    """

    global _warned_deprecation

    if isinstance(data, pd.DataFrame):
        if not _warned_deprecation:
            _logger.info(
                "llm_preprocessor.preprocess_data_with_llm is deprecated; returning input unchanged."
            )
            _warned_deprecation = True
        return data

    # For non-DataFrame inputs, retain previous behavior of wrapping into a DataFrame
    # without performing any LLM-based transformations.
    return pd.DataFrame(data)


def analyze_dataset_with_llm(df: pd.DataFrame) -> str:
    metadata = parse_metadata(df)
    prompt = generate_prompt(metadata)
    return llm_completion(prompt, max_tokens=LLM_TOKEN_BUDGET)


def preprocess_multiple_tasks(data, tasks=None):
    if tasks is None:
        tasks = ["text-classification", "sentiment-analysis"]
    results = {}
    for task in tasks:
        results[task] = llm_completion(f"Perform {task} on: {data}", max_tokens=LLM_TOKEN_BUDGET)
    return results


def handle_missing_values(data: pd.DataFrame, strategy: str = "mean"):
    if strategy == "mean":
        return data.fillna(data.mean())
    if strategy == "median":
        return data.fillna(data.median())
    if strategy == "mode":
        return data.fillna(data.mode().iloc[0])
    raise ValueError("Unsupported strategy")


def handle_outliers(data: pd.DataFrame, method: str = "IQR"):
    if method == "IQR":
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        return data[~((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr)))]
    if method == "Z-score":
        from scipy import stats
        return data[(abs(stats.zscore(data)) < 3)]
    raise ValueError("Unsupported method")


def encode_categorical_data(data: pd.Series) -> pd.Series:
    return data.astype("category").cat.codes


def preprocess_data_with_agents(data: pd.DataFrame) -> pd.DataFrame:
    data = handle_missing_values(data)
    data = handle_outliers(data)
    data = encode_categorical_data(data)
    return data


def score_similarity(uploaded_df: pd.DataFrame, datalake_dfs: dict) -> list:
    """Return datalake files sorted by similarity to the uploaded DataFrame."""
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        raise RuntimeError("scikit-learn is required for similarity scoring")
        
    uploaded_metadata = parse_metadata(uploaded_df)
    uploaded_vector = pd.DataFrame(uploaded_metadata["summary"]).values.flatten()
    uploaded_tags = set(str(col).lower() for col in uploaded_metadata["columns"])

    similarity_scores = {}
    for file_name, datalake_df in datalake_dfs.items():
        datalake_metadata = parse_metadata(datalake_df)
        datalake_vector = pd.DataFrame(datalake_metadata["summary"]).values.flatten()
        datalake_tags = set(str(col).lower() for col in datalake_metadata["columns"])

        # Vector similarity based on dataset summaries
        vec_score = cosine_similarity([uploaded_vector], [datalake_vector])[0][0]

        # Tag similarity based on overlapping column names
        union = uploaded_tags | datalake_tags
        tag_score = len(uploaded_tags & datalake_tags) / len(union) if union else 0.0

        # Combine the scores with equal weight
        similarity_scores[file_name] = (vec_score + tag_score) / 2

    sorted_files = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_files


def tag_dataset_with_llm(df: pd.DataFrame) -> str:
    metadata = parse_metadata(df)
    prompt = (
        "Tag the dataset columns and infer structure.\n"
        f"Columns: {metadata['columns']}\n"
        f"Types: {metadata['dtypes']}\n"
    )
    return llm_completion(prompt, max_tokens=LLM_TOKEN_BUDGET)


def recommend_models_with_llm(df: pd.DataFrame) -> str:
    metadata = parse_metadata(df)
    prompt = (
        "Recommend modeling approaches for the dataset.\n"
        f"Summary: {metadata['summary']}\n"
    )
    return llm_completion(prompt, max_tokens=LLM_TOKEN_BUDGET)


def load_mistral_model(model_path: str):
    """Load the Mistral model from ``model_path`` using ``llama_cpp``."""
    """Load the Mistral model from ``model_path`` using ``llama_cpp``."""
    # Lazy import
    global Llama
    if Llama is None:
        try:
            from llama_cpp import Llama as _Llama
            Llama = _Llama
        except Exception:
            pass
    
    if Llama is None:
        return None
    if model_path is not None:
        try:
            return Llama(model_path=model_path)
        except Exception:
            pass
    if AUTO_DOWNLOAD_LLM:
        try:
            from huggingface_hub import hf_hub_download
            dl_path = hf_hub_download(repo_id=LLM_REPO_ID, filename=LLM_FILENAME)
            return Llama(model_path=str(dl_path))
        except Exception:
            return None
    return None


def run_mistral_inference(
    model,
    input: str,
    max_tokens: int = LLM_TOKEN_BUDGET,
    temperature: float = LLM_TEMPERATURE,
) -> str:
    """Run inference with a loaded Mistral ``model``."""
    if model is None:
        return "LLM unavailable"

    try:
        input = _guard_input(input)
    except ValueError:
        return "LLM input rejected"

    def _call():
        try:
            output = model.create_completion(
                prompt=input,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return output["choices"][0]["text"].strip()
        except Exception as exc:  # pragma: no cover - runtime issues
            return f"LLM error: {exc}"

    key = ("run_mistral_inference", input, max_tokens, temperature)
    return _cached_llm_call(key, _call)


class LLMClient:
    """Unified LLM interface backing local GGUF and explicit model paths.

    Provider selection order:
    - provider='mistral_path': use MISTRAL_MODEL_PATH (fallback to local_llama)
    - provider='local_llama': use local GGUF via llama.cpp (fallback to mistral_path)
    - provider='auto' (default): try mistral_path, then local_llama
    """

    def __init__(self, provider: str | None = None, model_path: str | None = None):
        self.provider = (provider or LLM_PROVIDER or "auto").strip().lower()
        self.model_path = model_path if model_path is not None else MISTRAL_MODEL_PATH
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        if self.provider == "mistral_path":
            self._model = load_mistral_model(self.model_path)
            if self._model is None:
                self._model = load_local_llm()
            if self._model is None:
                self._model = load_transformers_model()
        elif self.provider == "local_llama":
            self._model = load_local_llm()
            if self._model is None:
                self._model = load_mistral_model(self.model_path)
            if self._model is None:
                self._model = load_transformers_model()
        elif self.provider == "transformers":
            self._model = load_transformers_model()
            if self._model is None:
                self._model = load_mistral_model(self.model_path)
            if self._model is None:
                self._model = load_local_llm()
        else:  # auto
            self._model = load_mistral_model(self.model_path)
            if self._model is None:
                self._model = load_local_llm()
            if self._model is None:
                self._model = load_transformers_model()
        return self._model

    def complete(self, prompt: str, max_tokens: int | None = None, temperature: float | None = None) -> str:
        if max_tokens is None:
            max_tokens = LLM_TOKEN_BUDGET
        if temperature is None:
            temperature = LLM_TEMPERATURE

        try:
            prompt = _guard_input(prompt)
        except ValueError:
            return "LLM input rejected"

        def _call():
            model = self._load()
            if model is None:
                return "LLM unavailable"
            try:
                output = model.create_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return output["choices"][0]["text"].strip()
            except Exception as exc:
                return f"LLM error: {exc}"

        key = ("LLMClient.complete", prompt, max_tokens, temperature, self.provider, bool(self.model_path))
        return _cached_llm_call(key, _call)

from . import get_bool, get_int
import logging
import sys


ENABLE_LOCAL_LLM: bool = get_bool("ENABLE_LOCAL_LLM", True)
AUTO_DOWNLOAD_LLM: bool = get_bool("AUTO_DOWNLOAD_LLM", True)
ENABLE_PROMETHEUS: bool = get_bool("ENABLE_PROMETHEUS", True)
MAX_ROWS_FIRST_PASS: int = get_int("MAX_ROWS_FIRST_PASS", 10_000_000)  # 10M rows
MAX_FEATURES_FIRST_PASS: int = get_int("MAX_FEATURES_FIRST_PASS", 500)
MODEL_TIME_BUDGET_SECONDS: int = get_int("MODEL_TIME_BUDGET_SECONDS", 300)  # 5 minutes
ALLOW_FULL_COMPARE_MODELS: bool = get_bool("ALLOW_FULL_COMPARE_MODELS", True)

# Model explanations - SHAP values enabled by default for interpretability
ENABLE_HEAVY_EXPLANATIONS: bool = get_bool("ENABLE_HEAVY_EXPLANATIONS", True)
ENABLE_SHAP_EXPLANATIONS: bool = get_bool("ENABLE_SHAP_EXPLANATIONS", True)

# Profiling limits
PROFILE_MAX_COLS: int = get_int("PROFILE_MAX_COLS", 200)
PROFILE_SAMPLE_ROWS: int = get_int("PROFILE_SAMPLE_ROWS", 5000)

# =============================================================================
# Developer/Diagnostic Mode
# =============================================================================

# DEV_MODE - enables all diagnostic features
# Set via environment: DEV_MODE=1 or DEV_MODE=true
DEV_MODE: bool = get_bool("DEV_MODE", False)

# Individual diagnostic flags (auto-enabled by DEV_MODE, but can be set independently)
# LOUD_FAILURES: Raise exceptions instead of catching silently (for debugging)
LOUD_FAILURES: bool = get_bool("LOUD_FAILURES", DEV_MODE)

# VERBOSE_LOGGING: Enable DEBUG-level logging for all Assay modules
VERBOSE_LOGGING: bool = get_bool("VERBOSE_LOGGING", DEV_MODE)

# TRACE_EXECUTIONS: Log detailed execution traces for cascade planner
TRACE_EXECUTIONS: bool = get_bool("TRACE_EXECUTIONS", DEV_MODE)

# DISABLE_ERROR_RECOVERY: Skip fallback/retry logic for clearer error diagnosis
DISABLE_ERROR_RECOVERY: bool = get_bool("DISABLE_ERROR_RECOVERY", False)

if DEV_MODE:
    ENABLE_LOCAL_LLM = False
    ENABLE_PROMETHEUS = False
    MAX_ROWS_FIRST_PASS = 1000


def configure_diagnostic_logging():
    """Configure logging based on diagnostic flags."""
    if VERBOSE_LOGGING:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[logging.StreamHandler(sys.stderr)],
        )
        # Suppress noisy third-party loggers
        for noisy in ["urllib3", "httpx", "httpcore", "openai", "anthropic"]:
            logging.getLogger(noisy).setLevel(logging.WARNING)


def diagnostic_raise(exception: Exception) -> None:
    """
    Raise exception in DEV_MODE, otherwise log warning.
    
    Use this instead of silent exception swallowing:
    
        try:
            risky_operation()
        except Exception as e:
            diagnostic_raise(e)  # Will raise in DEV_MODE, warn otherwise
    """
    if LOUD_FAILURES:
        raise exception
    else:
        logging.warning(f"Suppressed exception (set LOUD_FAILURES=1 for full trace): {exception}")


def diagnostic_context() -> dict:
    """Return current diagnostic configuration for debugging."""
    return {
        "DEV_MODE": DEV_MODE,
        "LOUD_FAILURES": LOUD_FAILURES,
        "VERBOSE_LOGGING": VERBOSE_LOGGING,
        "TRACE_EXECUTIONS": TRACE_EXECUTIONS,
        "DISABLE_ERROR_RECOVERY": DISABLE_ERROR_RECOVERY,
    }



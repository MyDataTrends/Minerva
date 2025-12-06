from . import get_bool, get_int


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

# Developer mode - simplifies setup for development/testing
DEV_MODE: bool = get_bool("DEV_MODE", False)
if DEV_MODE:
    ENABLE_LOCAL_LLM = False
    ENABLE_PROMETHEUS = False
    MAX_ROWS_FIRST_PASS = 1000


import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def load_dotenv(*_args, **_kwargs):
        return False

load_dotenv()

# Auto-create required directories to avoid setup friction
_REQUIRED_DIRS = ["logs", "User_Data", "local_data", "models", "output_files", "metadata"]
for _dir in _REQUIRED_DIRS:
    Path(_dir).mkdir(exist_ok=True)


def get_env(name: str, default: str | None = None, required: bool = False) -> str:
    """Return environment variable ``name`` with optional default.

    Raises a ``RuntimeError`` if the variable is required but not present.
    """
    value = os.getenv(name, default)
    if required and value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_bool(name: str, default: bool = False) -> bool:
    """Return environment variable ``name`` parsed as a boolean."""
    val = get_env(name, None)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "t", "yes", "y"}


def get_int(name: str, default: int = 0) -> int:
    """Return environment variable ``name`` parsed as an integer."""
    val = get_env(name, None)
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


USE_CLOUD = get_bool("USE_CLOUD", False)
LOCAL_DATA_DIR = get_env("LOCAL_DATA_DIR", "local_data")
BUCKET_NAME = get_env("BUCKET_NAME", "mydatatrendsbucket")
TABLE_NAME = get_env("TABLE_NAME", "UserJobs")
SNS_TOPIC_ARN = get_env("SNS_TOPIC_ARN", "arn:aws:sns:us-east-2:971422695246:JobCompletion")
LOG_LEVEL = get_env("LOG_LEVEL", "INFO")
LOG_DIR = get_env("LOG_DIR", "logs")
LOG_FILE = get_env("LOG_FILE", "app.log")
REDACTION_ENABLED = get_bool("REDACTION_ENABLED", True)
MODEL_SAVE_PATH = get_env("MODEL_SAVE_PATH", "model.pkl")
MIN_R2 = float(get_env("MIN_R2", "0.25"))
MAX_MAPE = float(get_env("MAX_MAPE", "30"))
MIN_ROWS = get_int("MIN_ROWS", 500)
SEMANTIC_INDEX_KEY = get_env("SEMANTIC_INDEX_KEY", "semantic_index.db")
CSV_READ_CHUNKSIZE = get_int("CSV_READ_CHUNKSIZE", 50000)
DESCRIBE_SAMPLE_ROWS = get_int("DESCRIBE_SAMPLE_ROWS", 10000)
LARGE_FILE_BYTES = get_int("LARGE_FILE_BYTES", 100 * 1024 * 1024)

LLM_CACHE_TTL = get_int("LLM_CACHE_TTL", 900)
LLM_MAX_CONCURRENCY = get_int("LLM_MAX_CONCURRENCY", 2)
# Path to local Mistral model - auto-detected from adm/llm_backends/local_model/
_LOCAL_MODEL_DIR = Path(__file__).parent.parent / "adm" / "llm_backends" / "local_model"
_MISTRAL_GGUF = _LOCAL_MODEL_DIR / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MISTRAL_MODEL_PATH = get_env("MISTRAL_MODEL_PATH", str(_MISTRAL_GGUF) if _MISTRAL_GGUF.exists() else None)
LLM_TOKEN_BUDGET = get_int("LLM_TOKEN_BUDGET", 2048)  # Increased for code generation
LLM_SUMMARY_TOKEN_BUDGET = get_int("LLM_SUMMARY_TOKEN_BUDGET", 512)
LLM_INTENT_TOKEN_BUDGET = get_int("LLM_INTENT_TOKEN_BUDGET", 256)
LLM_VISUALIZATION_TOKEN_BUDGET = get_int("LLM_VISUALIZATION_TOKEN_BUDGET", 128)
LLM_TEMPERATURE = float(get_env("LLM_TEMPERATURE", "0.7"))
LLM_INTENT_TEMPERATURE = float(get_env("LLM_INTENT_TEMPERATURE", "0"))
LLM_MAX_INPUT_CHARS = get_int("LLM_MAX_INPUT_CHARS", 16000)  # Increased for data profiles
LLM_NON_PRINTABLE_THRESHOLD = float(get_env("LLM_NON_PRINTABLE_THRESHOLD", "0.1"))
# Use Mistral 7B as default - much more capable than Qwen 0.5B
LLM_REPO_ID = get_env("LLM_REPO_ID", "TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
LLM_FILENAME = get_env("LLM_FILENAME", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
LLM_PROVIDER = get_env("LLM_PROVIDER", "auto")
LLM_HF_REPO_ID = get_env("LLM_HF_REPO_ID", "mistralai/Mistral-7B-Instruct-v0.2")

# Free-tier limits
MAX_REQUESTS_FREE = get_int("MAX_REQUESTS_FREE", 20)
MAX_GB_FREE = float(get_env("MAX_GB_FREE", "1"))

# Optional DynamoDB table for usage tracking
USAGE_TRACKING_TABLE = get_env("USAGE_TRACKING_TABLE", None)

# Optional endpoints for agent recipes
ENABLE_RECIPES = get_bool("ENABLE_RECIPES", False)
REORDER_API = get_env("REORDER_API", "")
ANOMALY_API = get_env("ANOMALY_API", "")
ROLE_REVIEW_API = get_env("ROLE_REVIEW_API", "")


if USE_CLOUD:
    import boto3
    s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
    dynamodb_client = boto3.resource("dynamodb", region_name=os.getenv("AWS_REGION", "us-east-1"))
    sns_client = boto3.client("sns", region_name=os.getenv("AWS_REGION", "us-east-1"))
else:
    s3_client = None
    dynamodb_client = None
    sns_client = None


def get_s3_bucket():
    if s3_client is None:
        raise RuntimeError("Cloud services are disabled")
    return s3_client.Bucket(BUCKET_NAME)


def get_dynamodb_table():
    if dynamodb_client is None:
        raise RuntimeError("Cloud services are disabled")
    return dynamodb_client.Table(TABLE_NAME)


def get_usage_table():
    """Return the DynamoDB table used for usage tracking."""
    if dynamodb_client is None or USAGE_TRACKING_TABLE is None:
        raise RuntimeError("Usage tracking is disabled")
    return dynamodb_client.Table(USAGE_TRACKING_TABLE)


import yaml
from pathlib import Path
from utils.security import secure_join

def load_yaml(path: str | Path):
    safe_path = secure_join(Path.cwd(), str(path))
    with open(safe_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

# Feature flags
from .feature_flags import (
    ENABLE_LOCAL_LLM,
    AUTO_DOWNLOAD_LLM,
    ENABLE_PROMETHEUS,
    MAX_ROWS_FIRST_PASS,
    MAX_FEATURES_FIRST_PASS,
    MODEL_TIME_BUDGET_SECONDS,
    ALLOW_FULL_COMPARE_MODELS,
    PROFILE_MAX_COLS,
    PROFILE_SAMPLE_ROWS,
    ENABLE_HEAVY_EXPLANATIONS,
    ENABLE_SHAP_EXPLANATIONS,
    DEV_MODE,
)



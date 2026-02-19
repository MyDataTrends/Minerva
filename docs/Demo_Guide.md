# Assay Demo Guide

This guide walks you through demonstrating all features of the Assay platform.

## Quick Start (5 minutes)

```powershell
# 1. Setup environment
cd C:\Projects\Minerva\Minerva
.\.venv\Scripts\Activate.ps1

# 2. Setup public data sources and semantic index
python -m catalog.public_data_sources setup
python -m catalog.public_data_sources rebuild

# 3. Run the full demo
python -m examples.full_demo
```

## Feature Demonstrations

### 1. Public Data Sources Registry

Assay includes a curated database of public datasets for semantic enrichment:

```powershell
# List all registered data sources
python -m catalog.public_data_sources list

# Setup/download all datasets
python -m catalog.public_data_sources setup

# Rebuild the semantic index
python -m catalog.public_data_sources rebuild
```

**Available Categories:**
- **Demographics**: Census income data, population statistics
- **Holidays**: US Federal holidays, Ecuador events
- **Economic**: Oil prices, economic indicators
- **Retail**: Store master data, transaction samples
- **Weather**: Temperature, precipitation data
- **Geographic**: ZIP to FIPS mappings, state codes

### 2. Semantic Enrichment Demo

```python
from Integration.semantic_merge import rank_and_merge
from preprocessing.metadata_parser import infer_column_meta
import pandas as pd

# Your data
user_data = pd.DataFrame({
    "date": pd.date_range("2023-01-01", periods=30),
    "store_nbr": [1, 2, 3] * 10,
    "sales": [100, 150, 200] * 10,
})

# Infer column roles
meta = infer_column_meta(user_data)

# Automatically enrich with best matching public dataset
enriched_df, report = rank_and_merge(user_data, meta)

print(f"Added {enriched_df.shape[1] - user_data.shape[1]} columns")
print(f"Merged with: {report['chosen_table']}")
```

### 3. Data Quality Scoring

```python
from orchestration.data_quality_scorer import compute_safety_metrics, summarize_for_display
import pandas as pd

df = pd.read_csv("datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv")
metrics = compute_safety_metrics(df)
summary = summarize_for_display(metrics)

print(f"Quality Score: {summary['score']}%")
print(f"Status: {summary['status_text']}")
print(f"Warnings: {summary['warnings']}")
```

### 4. Full Workflow Execution

```python
from orchestrate_workflow import orchestrate_workflow
from storage.local_backend import load_datalake_dfs

result = orchestrate_workflow(
    user_id="demo_user",
    file_name="WA_Fn-UseC_-Telco-Customer-Churn.csv",
    datalake_dfs=load_datalake_dfs(),
    target_column="Churn",
    diagnostics_config={
        "check_misalignment": True,
        "score_imputations": True,
        "monitor_drift": True,
    }
)

print(f"Analysis type: {result['analysis_type']}")
print(f"Metrics: {result['metrics']}")
```

### 5. Dashboard Demo

```powershell
# Start the API server
uvicorn main:app --reload

# In another terminal, start the dashboard
streamlit run ui/dashboard.py
```

Then:
1. Open http://localhost:8501
2. Upload `datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Watch automatic analysis complete
4. View the Data Quality Report
5. Explore Model Explanations (SHAP values)
6. Try the chatbot interface

## LLM Features

### Enable Local LLM

```powershell
# Option 1: Auto-download (recommended)
$env:ENABLE_LOCAL_LLM = "true"
$env:AUTO_DOWNLOAD_LLM = "true"
python main.py

# Option 2: Use existing model
$env:ENABLE_LOCAL_LLM = "true"
$env:MISTRAL_MODEL_PATH = "path/to/your/model.gguf"
python main.py
```

### LLM-Powered Features

When LLM is enabled:
- **Enhanced column role inference**: Better semantic understanding
- **Model recommendations**: Intelligent suggestions based on data
- **Business summaries**: Plain-English explanations of results
- **Smart visualization**: LLM fallback when heuristics uncertain
- **Chatbot responses**: Natural language interaction

### Test LLM Availability

```python
from preprocessing.llm_preprocessor import recommend_models_with_llm
import pandas as pd

df = pd.DataFrame({"sales": [100, 200, 300], "date": ["2023-01-01", "2023-01-02", "2023-01-03"]})
result = recommend_models_with_llm(df)
print(result)  # Will show recommendations or "LLM unavailable"
```

## Development Mode

For quick testing without LLM:

```powershell
$env:DEV_MODE = "true"
python main.py
```

DEV_MODE automatically:
- Disables LLM features
- Disables Prometheus metrics
- Reduces row limits for faster processing

## API Endpoints

Start the API server:
```powershell
uvicorn main:app --reload
```

**Available Endpoints:**
- `GET /healthz` - Health check
- `GET /sessions` - List past analysis sessions
- `POST /sessions/{run_id}/rerun` - Rerun a previous analysis
- `GET /docs` - Interactive API documentation

## Test Commands

```powershell
# Health check
pytest tests/test_health.py -v

# Full test suite
pytest tests/ -v

# Specific component tests
pytest tests/test_selector.py -v
pytest tests/test_explanations_flag.py -v
```

## Showcase Script

For a complete demonstration of all features:

```powershell
python -m examples.full_demo
```

This interactive script demonstrates:
1. Public data registry
2. Semantic index
3. Data preprocessing
4. Column role inference
5. Semantic enrichment
6. Analyzer selection
7. SHAP explanations
8. LLM features
9. Visualization selection
10. Full workflow execution

## Troubleshooting

### Common Issues

**"LLM unavailable" messages:**
- This is normal if LLM is disabled or model not downloaded
- System falls back to heuristic methods

**Missing datasets:**
```powershell
python -m catalog.public_data_sources setup
```

**Semantic index errors:**
```powershell
python -m catalog.public_data_sources rebuild
```

**Import errors:**
```powershell
pip install -r config/requirements.txt
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_LOCAL_LLM` | true | Enable local LLM features |
| `AUTO_DOWNLOAD_LLM` | true | Auto-download LLM model |
| `DEV_MODE` | false | Simplified development mode |
| `ENABLE_SHAP_EXPLANATIONS` | true | Enable SHAP values |
| `LOCAL_DATA_DIR` | local_data | Data storage directory |
| `LOG_DIR` | logs | Log file directory |
| `MIN_R2` | 0.25 | Minimum R² threshold |
| `MAX_MAPE` | 30 | Maximum MAPE threshold |

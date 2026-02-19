# Onboarding (Windows + VS Code)

## Prerequisites
- Python 3.10 or 3.11 (3.12+ may have compatibility issues)
- Git
- Optional: Conda (Miniconda/Anaconda)

## Quick Start (5 minutes)

### 1. Clone Repository
```powershell
git clone [repository-url]
cd Assay
```

### 2. Create Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r config\requirements.txt
```

### 4. Quick Configuration
```powershell
# Create required directories
mkdir logs, User_Data, local_data -Force

# Create minimal .env file
echo "LOCAL_DATA_DIR=local_data" > config\.env
echo "LOG_DIR=logs" >> config\.env
```

### 5. Verify Installation
```powershell
python -c "from orchestration.workflow_manager import WorkflowManager; print('Core imports work')"
```

### 6. Run Health Check
```powershell
pytest tests\test_health.py -v
```

### 7. Start Services (No LLM Mode)
```powershell
# Terminal 1: Start API
python main.py --no-llm

# Terminal 2: Start Dashboard
streamlit run ui\dashboard.py
```

### 8. Test with Sample Data
- Open http://localhost:8501
- Upload `datasets\WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Analysis should complete in < 60 seconds

## Full Configuration

For advanced setup, copy and edit the example environment file:
```powershell
copy config\.env.example config\.env
# Edit config\.env to customize LOG_DIR, LOCAL_DATA_DIR, LLM settings, etc.
```

## Running Tests
```powershell
# Quick sanity check
pytest tests\test_health.py -v

# Full test suite
pytest -q
```

## Running Demos
```powershell
python -m examples.test_workflow_demo
python -m examples.imputation_confidence_demo
```

## Logs
JSON logs are written to `%PROJECT%\logs\app.log` (configurable via LOG_DIR/LOG_FILE)

## Rerun Previous Analyses
Use FastAPI endpoint `POST /sessions/{run_id}/rerun` or via the Dashboard UI sidebar

## Troubleshooting

### Common Issues
- **ModuleNotFoundError**: Ensure virtual environment is activated
- **LLM unavailable**: This is normal in `--no-llm` mode; system uses heuristic fallbacks
- **Missing directories**: Run `mkdir logs, User_Data, local_data -Force`
- **Python 3.12+ issues**: Downgrade to Python 3.10 or 3.11

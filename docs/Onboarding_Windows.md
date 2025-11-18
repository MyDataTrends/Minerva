# Onboarding (Windows + VS Code)

- Prereqs
  - Install Python 3.10+
  - Install Git
  - Optional: Conda (Miniconda/Anaconda)

- Clone
  - git clone [repository-url] && cd Minerva

- Create a virtualenv (PowerShell)
  - python -m venv .venv
  - .\.venv\Scripts\Activate.ps1

- Install dependencies (pip)
  - pip install -r infra\requirements\requirements.txt
  - Optional: conda users can use infra\environments\environment.yml

- Configure environment
  - copy config/.env.example config/.env
  - edit config/.env (set LOG_DIR, LOCAL_DATA_DIR, etc.)

- Run tests
  - pytest -q

- Run demos
  - python -m examples.test_workflow_demo
  - python -m examples.imputation_confidence_demo

- Start services
  - API: uvicorn main:app --reload
  - Dashboard: streamlit run ui\dashboard.py

- Logs
  - JSON logs written to %PROJECT%\logs\app.log (set LOG_DIR/LOG_FILE)

- Rerun flows
  - Use FastAPI endpoint POST /sessions/{run_id}/rerun or via UI

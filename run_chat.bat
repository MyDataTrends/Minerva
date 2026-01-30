@echo off
REM Launch Minerva Chat Mode
echo Starting Minerva Chat Mode...
cd /d "%~dp0"
call .venv\Scripts\activate 2>nul || (
    echo Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate
    pip install -r config\requirements.txt
)
streamlit run ui/chat_mode.py --server.port 8502 --browser.gatherUsageStats false

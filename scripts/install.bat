@echo off
setlocal

echo ==================================================
echo   Minerva - AI Data Analyst Installer (Windows)
echo ==================================================
echo.

:: 1. Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.10+ and add it to PATH.
    echo Get it here: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: 2. Create Virtual Environment if not exists
if not exist ".venv" (
    echo [INFO] Creating virtual environment (.venv)...
    python -m venv .venv
) else (
    echo [INFO] Virtual environment already exists.
)

:: 3. Activate Venv and Install Dependencies
echo [INFO] Installing dependencies...
call .venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

:: 4. Done
echo.
echo [SUCCESS] Minerva is installed!
echo.
echo To run the dashboard:
echo    scripts\run_dashboard.bat
echo.
echo To run the agent:
echo    python -m agents run conductor
echo.
pause

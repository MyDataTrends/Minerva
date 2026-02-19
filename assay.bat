@echo off
REM Assay CLI launcher for Windows
REM Usage: assay <command> [options]

setlocal

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Activate virtual environment if it exists
if exist "%SCRIPT_DIR%.venv\Scripts\activate.bat" (
    call "%SCRIPT_DIR%.venv\Scripts\activate.bat"
)

REM Run the CLI
python "%SCRIPT_DIR%cli.py" %*

endlocal

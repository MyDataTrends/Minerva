#!/bin/bash

echo "=================================================="
echo "  Assay - AI Data Analyst Installer (Mac/Linux)"
echo "=================================================="
echo ""

# 1. Check for Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found! Please install it."
    exit 1
fi

# 2. Create Virtual Environment if not exists
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment (.venv)..."
    python3 -m venv .venv
else
    echo "[INFO] Virtual environment already exists."
fi

# 3. Activate Venv and Install Dependencies
echo "[INFO] Installing dependencies..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Done
echo ""
echo "[SUCCESS] Assay is installed!"
echo ""
echo "To run the dashboard:"
echo "   source .venv/bin/activate && streamlit run ui/dashboard.py"
echo ""
echo "To run the agent:"
echo "   source .venv/bin/activate && python -m agents run conductor"
echo ""

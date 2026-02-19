#!/usr/bin/env bash
# Assay one-liner installer for Linux/Mac
# Usage: curl -sSL https://raw.githubusercontent.com/YOUR_GITHUB/assay/main/install.sh | bash

set -euo pipefail

REPO_URL="https://github.com/MyDataTrends/assay.git"
INSTALL_DIR="$HOME/.assay"
BIN_DIR="$HOME/.local/bin"
PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=10

# ── colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[assay]${NC} $*"; }
warn()    { echo -e "${YELLOW}[assay]${NC} $*"; }
error()   { echo -e "${RED}[assay] ERROR:${NC} $*" >&2; exit 1; }

# ── check Python ─────────────────────────────────────────────────────────────
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=${ver%%.*}; minor=${ver##*.}
        if [[ $major -gt $PYTHON_MIN_MAJOR ]] || \
           [[ $major -eq $PYTHON_MIN_MAJOR && $minor -ge $PYTHON_MIN_MINOR ]]; then
            PYTHON="$cmd"
            info "Found Python $ver at $(command -v $cmd)"
            break
        fi
    fi
done
[[ -z "$PYTHON" ]] && error "Python 3.10+ is required. Install it from https://python.org and re-run."

# ── clone or update repo ──────────────────────────────────────────────────────
if [[ -d "$INSTALL_DIR/.git" ]]; then
    info "Updating existing install at $INSTALL_DIR ..."
    git -C "$INSTALL_DIR" pull --ff-only
else
    info "Cloning Assay into $INSTALL_DIR ..."
    git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"
fi

# ── virtual environment ───────────────────────────────────────────────────────
VENV="$INSTALL_DIR/.venv"
if [[ ! -d "$VENV" ]]; then
    info "Creating virtual environment ..."
    "$PYTHON" -m venv "$VENV"
fi
PIP="$VENV/bin/pip"
PYTHON_VENV="$VENV/bin/python"

info "Upgrading pip ..."
"$PIP" install --quiet --upgrade pip

# Install CPU-only torch first to avoid pulling in CUDA (~2 GB)
info "Installing PyTorch (CPU-only) ..."
"$PIP" install --quiet torch --index-url https://download.pytorch.org/whl/cpu

info "Installing Assay dependencies ..."
"$PIP" install --quiet -r "$INSTALL_DIR/requirements.txt"

# ── .env ─────────────────────────────────────────────────────────────────────
if [[ ! -f "$INSTALL_DIR/.env" && -f "$INSTALL_DIR/.env.example" ]]; then
    cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
    warn ".env created from .env.example — edit it to add your API keys."
fi

# ── launcher script ───────────────────────────────────────────────────────────
mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/assay" <<EOF
#!/usr/bin/env bash
exec "$PYTHON_VENV" "$INSTALL_DIR/cli.py" "\$@"
EOF
chmod +x "$BIN_DIR/assay"
info "Launcher written to $BIN_DIR/assay"

# ── PATH hint ────────────────────────────────────────────────────────────────
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    warn "$BIN_DIR is not in your PATH."
    warn "Add this line to your ~/.bashrc or ~/.zshrc:"
    warn '  export PATH="$HOME/.local/bin:$PATH"'
    warn "Then run: source ~/.bashrc  (or open a new terminal)"
fi

info "Done! Run: assay info"

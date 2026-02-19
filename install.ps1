# Assay one-liner installer for Windows (PowerShell)
# Usage: irm https://raw.githubusercontent.com/YOUR_GITHUB/assay/main/install.ps1 | iex

$ErrorActionPreference = "Stop"

$REPO_URL = "https://github.com/MyDataTrends/assay.git"
$INSTALL_DIR = Join-Path $env:USERPROFILE ".assay"

function Info { param($msg) Write-Host "[assay] $msg" -ForegroundColor Green }
function Warn { param($msg) Write-Host "[assay] $msg" -ForegroundColor Yellow }
function Fail { param($msg) Write-Host "[assay] ERROR: $msg" -ForegroundColor Red; exit 1 }

# ── check Python ──────────────────────────────────────────────────────────────
$PYTHON = $null
foreach ($cmd in @("python", "python3")) {
    try {
        $ver = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($ver) {
            $parts = $ver.Split(".")
            $major = [int]$parts[0]; $minor = [int]$parts[1]
            if ($major -gt 3 -or ($major -eq 3 -and $minor -ge 10)) {
                $PYTHON = $cmd
                Info "Found Python $ver"
                break
            }
        }
    }
    catch {}
}
if (-not $PYTHON) { Fail "Python 3.10+ is required. Download from https://python.org and re-run." }

# ── clone or update repo ───────────────────────────────────────────────────────
if (Test-Path (Join-Path $INSTALL_DIR ".git")) {
    Info "Updating existing install at $INSTALL_DIR ..."
    git -C $INSTALL_DIR pull --ff-only
}
else {
    Info "Cloning Assay into $INSTALL_DIR ..."
    git clone --depth 1 $REPO_URL $INSTALL_DIR
}

# ── virtual environment ────────────────────────────────────────────────────────
$VENV = Join-Path $INSTALL_DIR ".venv"
if (-not (Test-Path $VENV)) {
    Info "Creating virtual environment ..."
    & $PYTHON -m venv $VENV
}
$PIP = Join-Path $VENV "Scripts\pip.exe"
$PYTHON_VENV = Join-Path $VENV "Scripts\python.exe"

Info "Upgrading pip ..."
& $PIP install --quiet --upgrade pip

# Install CPU-only torch first to avoid CUDA download (~2 GB)
Info "Installing PyTorch (CPU-only) ..."
& $PIP install --quiet torch --index-url https://download.pytorch.org/whl/cpu

Info "Installing Assay dependencies ..."
& $PIP install --quiet -r (Join-Path $INSTALL_DIR "requirements.txt")

# ── .env ──────────────────────────────────────────────────────────────────────
$ENV_FILE = Join-Path $INSTALL_DIR ".env"
$ENV_EXAMPLE = Join-Path $INSTALL_DIR ".env.example"
if (-not (Test-Path $ENV_FILE) -and (Test-Path $ENV_EXAMPLE)) {
    Copy-Item $ENV_EXAMPLE $ENV_FILE
    Warn ".env created from .env.example — edit it to add your API keys."
}

# ── add assay function to PowerShell profile ───────────────────────────────────
$FUNC = @"

# Assay CLI (added by installer)
function assay {
    & "$PYTHON_VENV" "$INSTALL_DIR\cli.py" @args
}
"@

if (-not (Test-Path $PROFILE)) {
    New-Item -ItemType File -Path $PROFILE -Force | Out-Null
}
$profileContent = Get-Content $PROFILE -Raw -ErrorAction SilentlyContinue
if ($profileContent -notmatch "Assay CLI") {
    Add-Content -Path $PROFILE -Value $FUNC
    Info "Added 'assay' function to PowerShell profile: $PROFILE"
    Warn "Restart your terminal (or run: . `$PROFILE) for the 'assay' command to be available."
}
else {
    Info "'assay' function already present in PowerShell profile."
}

Info "Done! Open a new terminal and run: assay info"

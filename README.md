# Assay 🦉

[![CI](https://github.com/MyDataTrends/Assay/actions/workflows/ci.yml/badge.svg)](https://github.com/MyDataTrends/Assay/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/assay)](https://pypi.org/project/assay/)
[![Docker](https://img.shields.io/docker/v/mydatatrends/assay?label=docker)](https://hub.docker.com/r/mydatatrends/assay)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Upload. Analyze. Done.**

Assay is an **AI-powered data analyst** that runs locally on your machine. It turns raw CSVs into insights, charts, and reports without you writing a single line of code.

## Why Assay?

Traditional analytics tools (Excel, Tableau, Power BI) require you to manually clean data, build charts, and configure dashboards.
Chat-based AI tools (ChatGPT, Julius) require you to upload your sensitive data to the cloud.

Assay is different:

- **Zero Configuration**: Just upload a file. The system auto-detects types, cleans data, selects models, and generates insights.
- **Local & Private**: Your data never leaves your machine. Processing happens locally or via your personal API keys.
- **Self-Improving**: It remembers your domain. If you teach it that "Fiscal Year starts in Feb", it remembers that for every future analysis.
- **Transparent**: Unlike black-box AI, Assay generates actual Python code that you can inspect, audit, and reuse.

## Features

- **Autonomous Analysis**: Upload a dataset and get a full analysis report in seconds.
- **Smart Charts**: Auto-selects the best visualization for your data (e.g., time-series lines, categorical bars).
- **Natural Language Query**: Ask "Show me sales by region" and get a chart + answer.
- **Data Fabric**: Visualizes how your datasets connect and where they came from.
- **Agentic Infrastructure**: Built as an MCP Server, so other agents (like Claude Desktop) can use Assay as a tool.
- **Active Learning**: Learns from your feedback to improve future analyses.

## Quick Start (Win/Mac/Linux)

### 1. Install

Assay requires **Python 3.10+** and **Git**.

#### One-liner (Linux / Mac)

```bash
curl -sSL https://raw.githubusercontent.com/MyDataTrends/Assay/main/install.sh | bash
```

Clones the repo to `~/.assay`, creates a virtualenv, installs all dependencies, and adds `assay` to your PATH.

#### One-liner (Windows PowerShell)

```powershell
irm https://raw.githubusercontent.com/MyDataTrends/Assay/main/install.ps1 | iex
```

Same as above — adds an `assay` function to your PowerShell profile.

#### pip (Python package)

```bash
pip install assay                   # core deps + CLI
pip install "assay[full]"           # + ML/cloud extras (torch, spaCy, boto3, etc.)
assay info
```

#### Docker (zero-dependency)

```bash
docker-compose up --build
```

Access the dashboard at `http://localhost:8501`.

#### Manual install

```bash
git clone https://github.com/MyDataTrends/Assay.git
cd Assay
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run ui/dashboard.py
```

### 2. Configure (Optional)

Copy `.env.example` to `.env` and add your API keys:

- `ANTHROPIC_API_KEY`: For faster, smarter analysis (highly recommended).
- `FRED_API_KEY`: To fetch economic data from the Federal Reserve.

### 3. Run

```bash
assay dashboard        # launch Streamlit UI
assay serve            # launch FastAPI backend only
assay analyze FILE     # analyze a CSV from the command line
assay info             # show version and environment info
```

### 4. First Analysis

1. Upon launch, you'll see the **Welcome Wizard**.
2. Click **"Load Sample Data"** to analyze the included sales dataset.
3. Watch as Assay:
   - Cleans the data
   - Profiles column types
   - Selects a forecasting model
   - Generates a sales trend chart

## Advanced Usage

### Running the Agent Stack

Assay isn't just a dashboard — it's a system of autonomous agents that can run your data operations.

To run the **Conductor** (which orchestrates the entire system):

```bash
python -m agents run conductor
```

To run the **Engineer** (which autonomously fixes code gaps):

```bash
python -m agents run engineer
```

### MCP Server (For Claude Desktop)

Assay exposes its tools via the Model Context Protocol (MCP). To let Claude use Assay:

1. Add this to your Claude Desktop config:

```json
{
  "mcpServers": {
    "assay": {
      "command": "python",
      "args": ["-m", "mcp_server", "run"]
    }
  }
}
```

1. Restart Claude Desktop. You can now ask Claude: *"Analyze the data in my Assay workspace"* or *"Find GDP data using Assay"*.

## Privacy & Security

- **Local-First**: All data storage (SQLite + Parquet) is local to your machine.
- **Sandboxed Execution**: AI-generated code runs in a restricted environment.
- **Key Management**: API keys are stored in a local `.env` file or encrypted OS credential store.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to set up your dev environment and submit PRs.

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

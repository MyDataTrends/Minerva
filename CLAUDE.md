# CLAUDE.md — Assay Project Context

## What This Project Is

Assay is a **local-first, AI-powered autonomous data analyst** (~47K lines of Python, ~220 source files). Users upload CSVs and get automated cleaning, profiling, modeling, visualization, and insights — zero configuration. It runs as a Streamlit dashboard, FastAPI backend, CLI, and MCP server.

**Business context**: The owner is a solo founder (former Lowe's business analytics consultant) building toward either an open-source SaaS (free local tier + paid cloud tier) or a technology asset sale. The priority is making the system as autonomous as possible — the agent layer should handle routine operations so the founder only makes strategic decisions.

## Architecture (Three Layers)

### Layer 1: Core Engine (Deterministic)
- **Entry points**: `main.py` (FastAPI), `ui/dashboard.py` (Streamlit), `cli.py`
- **Data pipeline**: `storage/local_backend.py` → `preprocessing/` → `orchestration/data_preprocessor.py`
- **Modeling**: `modeling/` — pluggable analyzer registry (regression, classification, clustering, anomaly, descriptive). Each reports `suitability_score`. Cross-validated sweep over LightGBM/XGBoost/Ridge/LogReg from `config/model_allowlist.py`. SHAP explanations.
- **Storage**: Dual-backend (local filesystem + S3), SQLite sessions, Parquet support, chunked CSV for large files.

### Layer 2: Cognitive Core (Planning)
- **CascadePlanner** (`orchestration/cascade_planner.py`): Intent classification (regex first, LLM fallback) → JSON plan → tool execution with retry/fallback.
- **ToolRegistry** (`orchestration/tool_registry.py`): Typed, schema-validated tool handlers (profiler, filter, group_by, chart_generator, etc.).
- **PlanLearner** (`orchestration/plan_learner.py`): Active learning — stores successful patterns, tool weights (Bayesian), and intent patterns in SQLite.

### Layer 3: Agent Layer (Autonomous Workforce)
All agents inherit from `BaseAgent` (`agents/base.py`). Communication is via artifacts (files), not direct calls. Conductor orchestrates.

| Agent | File | Role |
|-------|------|------|
| Conductor | `agents/conductor.py` | Daily orchestrator, compiles digest (🔴/🟡/🟢/📊) |
| Engineer | `agents/engineer.py` | Scores subsystem maturity (1-4), writes gap analysis |
| Sentinel | `agents/sentinel.py` | Runs pytest + ruff, gates PRs with confidence score (1-10) |
| Scheduler | `agents/scheduler.py` | Recurring analysis via CascadePlanner |
| Advocate | `agents/advocate.py` | GitHub issue triage, auto-label, draft responses |
| Productizer | `agents/productizer.py` | Matches datasets to vertical templates, generates MVP plans |

**Agents that need to be built** (high priority):
- **Marketing Agent**: Monitor HN/Reddit/Twitter for relevant threads, draft responses, generate changelog posts from git diffs, draft blog posts from Productizer templates.
- **Support Agent**: Extend Advocate — answer common questions from knowledge base, escalate hard ones only.
- **Telemetry Agent**: Track anonymous usage patterns (with consent), identify popular features, surface churn signals, feed back into Productizer.
- **Billing/Onboarding Module**: Stripe integration, license key system, usage-based gating (extends existing `utils/usage_tracker.py`).

## Key Subsystems

- **MCP Server** (`mcp_server/`): JSON-RPC 2.0, stdio + HTTP transport. Tools for analysis, visualization, semantic enrichment, API discovery, feedback, workflow. Session management.
- **API Discovery** (`mcp_server/discovery_agent.py`): NL → API registry search with Kaggle-derived vertical weighting → LLM parameter mapping → web search fallback → dynamic connector generation.
- **Semantic Merge** (`Integration/semantic_merge.py`): Auto-joins user data with public datasets by column *roles* (not names). Supports zip→FIPS, city+state hashing, multi-strategy type coercion.
- **Vector Store** (`learning/vector_store.py`): SQLite + cosine similarity. Teachable memory — users store domain rules retrieved via RAG.
- **Smart Charts** (`visualization/smart_charts.py`): Column profiling → chart recommendation with confidence scores.
- **LLM Manager** (`llm_manager/`): Pluggable providers (local/Anthropic/OpenAI/Ollama). Registry with auto-scan for GGUF files. Interface cascades: registry → auto-select → subprocess fallback.
- **Credential Manager** (`mcp_server/credential_manager.py`): Fernet AES-128 encryption, PBKDF2 key derivation.

## Coding Conventions

- **Agent pattern**: Inherit `BaseAgent`, implement `run(**kwargs) -> AgentResult`. Use `OperationalMemory` for state, `KnowledgeBase` for shared artifacts. Config via `agents/agents_config.yaml` + env vars. Default `dry_run=True`.
- **Escalation levels**: `Priority.URGENT` (🔴), `Priority.REVIEW` (🟡), `Priority.FYI` (🟢), `Priority.METRIC` (📊).
- **LLM calls**: Use `llm_manager.llm_interface.get_llm_completion()` or `get_llm_chat()`. Never assume LLM is available — always handle empty string returns.
- **Security**: Use `utils.security.secure_join()` for all file paths. Restricted globals for `exec()`. Never send row-level PII to LLM — schema/metadata only.
- **Config**: All settings in `config/__init__.py` via env vars with sensible defaults. Feature flags in `config/feature_flags.py`. Use `get_bool()`, `get_int()`, `get_env()`.
- **Logging**: Use `utils.logging.get_logger(__name__)`. JSON structured logging. Redaction enabled by default.
- **Testing**: pytest. Tests in `tests/` and some inline `test_*.py` files. Run: `pytest tests/ -v`. Conftest in `tests/conftest.py`.
- **Error handling**: Three-strategy fallback pattern (LLM → deterministic → safe default). Use `config.feature_flags.diagnostic_raise()` instead of silent swallowing.

## File Layout Cheat Sheet

```
agents/          — Agent implementations + memory + tools
catalog/         — Public data source registry + semantic index
chatbot/         — Intent parsing, decision routing
config/          — All configuration, feature flags, model allowlist
Data_Intake/     — Ingestion from S3/API/local
Integration/     — Semantic merge (auto-join logic)
learning/        — Vector store, embeddings, cache
llm_learning/    — Interaction logger, example store
llm_manager/     — LLM providers, registry, scanner, downloader
mcp_server/      — MCP server, tools, discovery agent, credential manager
modeling/        — ML analyzers, model selector, training, suitability checks
orchestration/   — Cascade planner, tool registry, plan learner, workflow manager
output/          — Output formatting
preprocessing/   — Data cleaning, metadata, column meta, LLM preprocessor
public_data/     — Connectors (FRED, Census, Alpha Vantage, World Bank, etc.)
reports/         — Report generation
scripts/         — Utility scripts (drift monitor, imputation, viz selector)
storage/         — Local + S3 backends, session DB
tagging/         — Category management
tests/           — Test suite (~60 test files)
tools/           — Alignment drift monitor, dependency checker
ui/              — Streamlit frontend (dashboard, chat, NL query, data fabric, etc.)
utils/           — Logging, security, metrics, safe pickle, usage tracker
visualization/   — Smart chart selection + rendering
```

## Commands

```bash
# Run dashboard
streamlit run ui/dashboard.py

# Run API server
uvicorn main:app --reload

# Run agent system
python -m agents run conductor
python -m agents run engineer

# Run tests
pytest tests/ -v
pytest tests/test_health.py -v  # Quick sanity check

# Setup public data
python -m catalog.public_data_sources setup
python -m catalog.public_data_sources rebuild

# Full demo
python -m examples.full_demo
```

## Current Priorities

1. Build Marketing Agent, Support Agent, Telemetry Agent (follow BaseAgent pattern)
2. Clean repo for public GitHub launch (remove hardcoded paths, verify Docker works, write tutorials)
3. Add Stripe billing module (extend `utils/usage_tracker.py`)
4. Landing page / docs site
5. Launch on Hacker News + Reddit

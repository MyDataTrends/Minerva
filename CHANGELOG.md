# Changelog

All notable changes to Assay will be documented here.
Format: [Keep a Changelog](https://keepachangelog.com) · [Semantic Versioning](https://semver.org)

## [Unreleased]

## [0.1.0] — 2026-02-19

### Added
- Multi-stage Docker build; `docker-compose up` zero-config launch
- FastAPI backend (port 8000) + Streamlit dashboard (port 8501) + embedded MCP server (port 8766)
- CascadePlanner: intent classification → JSON plan → tool execution with retry/fallback
- Agent workforce: Conductor, Engineer, Sentinel, Scheduler, Advocate, Productizer
- LightGBM/XGBoost/Ridge/LogReg modeling sweep with SHAP explanations
- Semantic merge: auto-join user data with public datasets by column role
- Vector store teachable memory (SQLite + cosine similarity)
- Smart chart recommendation engine
- Public data connectors: FRED, Census, Alpha Vantage, World Bank
- `assay` CLI: analyze, serve, dashboard, test, info, admin, ingest
- One-liner install script for Linux/Mac (`install.sh`) and Windows (`install.ps1`)
- MCP server with JSON-RPC 2.0, stdio + HTTP transport
- API discovery: NL → public API registry search with dynamic connector generation
- Pluggable LLM manager: local GGUF, Anthropic, OpenAI, Ollama
- Fernet-encrypted credential manager with PBKDF2 key derivation
- PlanLearner: active learning — stores successful patterns and tool weights in SQLite
- `pyproject.toml` packaging; installable via `pip install assay`
- GitHub Actions CI (Python 3.10–3.12 matrix) and release pipeline (Docker Hub + PyPI)

[Unreleased]: https://github.com/MyDataTrends/assay/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/MyDataTrends/assay/releases/tag/v0.1.0

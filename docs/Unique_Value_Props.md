# Unique Value Propositions

## Core Differentiators

**Privacy-First Architecture** – Assay runs entirely on the user's local machine. Data never leaves the host. The local LLM (Llama 3 via `llama_cpp`) runs in a detached subprocess with optional CUDA GPU acceleration, providing cloud-level AI capability with zero data exposure.

**Self-Improving Execution Engine** – The Cascade Planner (`orchestration/cascade_planner.py`) classifies intents, generates multi-step execution plans, and routes to registered tools with retry, fallback, and dependency-aware execution. The Plan Learner (`orchestration/plan_learner.py`) records every successful execution pattern in SQLite, refining tool weights and intent classification over time – the system literally gets smarter with use.

**Teachable Memory** – The Vector Store (`learning/vector_store.py`) uses SQLite + cosine similarity to store user-taught formulas, queries, and domain-specific logic. Users can "teach" Assay their business rules and KPIs, which are recalled via embedding-based retrieval (RAG) during future queries.

**Autonomous API Discovery & Dynamic Connectors** – The API Discovery Agent (`mcp_server/discovery_agent.py`) takes natural-language descriptions of data needs, searches a curated registry of 20+ public APIs, applies Kaggle-derived vertical weighting, and ranks results. When no known API matches, the Dynamic Connector Generator (`mcp_server/dynamic_connector.py`) fetches API documentation, uses the LLM to generate Python connector code, validates it via AST analysis, and executes it in a sandbox.

**Model Context Protocol (MCP) Server** – A full JSON-RPC 2.0 MCP server (`mcp_server/server.py`) with stdio + HTTP transport, tool registration, session management, and resource discovery. Enables direct integration with Claude Desktop, MCP Inspector, or custom AI agents.

**Semantic Data Enrichment** – Automatic merging of user data with public datasets (FRED, World Bank, Census Bureau, Alpha Vantage, Google Sheets, OneDrive, local files) using semantic join key detection (ZIP/FIPS/ISO). The Semantic Router (`mcp_server/semantic_router.py`) uses embedding-based matching with LLM fallback to find the best data source.

**Automated Model Selection** – The Model Selector (`modeling/model_selector.py`) runs cross-validated sweeps across regression, classification, clustering, anomaly detection, and forecasting algorithms using an allowlisted set of models. The LLM Dynamic Analyzer (`modeling/llm_dynamic_analyzer.py`) generates custom preprocessing and analysis code on the fly, validated via AST before sandboxed execution.

**Smart Visualization** – `visualization/smart_charts.py` profiles each column (temporal, categorical, numeric, geographic, etc.), recommends optimal chart types with confidence scores and reasoning, and renders via Plotly.

---

## Technical Moats

| Capability | Implementation |
|:---|:---|
| **Dynamic Analyzer Registry** | Regression, classification, clustering, anomaly, and descriptive analyzers each report a `suitability_score` for automatic selection |
| **LLM Code Sandboxing** | All LLM-generated code is AST-validated, import-whitelisted, and executed in restricted globals |
| **Active Learning Loop** | `PlanLearner` stores learned intent patterns, tool success weights, and input mappings in SQLite for progressive optimization |
| **Encrypted Credential Store** | `CredentialManager` uses Fernet AES-128 encryption (PBKDF2 key derivation) for secure local API key storage |
| **3-Strategy Workflow Fallback** | `run_workflow` tries LLM dynamic analysis → standard orchestration → descriptive stats, never failing silently |
| **Interaction Telemetry** | `InteractionLogger` tracks all LLM interactions with frustration detection, session summarization, gamification milestones, and two-tier retention |
| **Run Artifact Replay** | `ArtifactStore` saves complete execution context (data snapshots, code, outputs) enabling failed run replay for debugging |
| **Tier-Based Access Gating** | Usage tracker enforces request/size limits with `check_quota`, providing built-in upgrade path |

---

## Competitive Advantage

This combination of local LLM execution, self-improving orchestration, teachable memory, and autonomous data sourcing is architecturally difficult to replicate. Cloud competitors cannot offer the same privacy guarantees. Traditional BI tools cannot offer the AI-powered adaptability. Open-source notebooks cannot offer the zero-configuration automation.

---

## Opportunities to Strengthen

- Apple Silicon (Metal) and AMD ROCm GPU support for broader hardware reach
- Multi-user collaboration with shared vector store synchronization
- Plugin marketplace for community-contributed connectors and agent recipes
- Multi-language LLM support for international users
- Streaming/incremental analysis for real-time data sources

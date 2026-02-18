# Architecture & Design Guide (Phase 2)

*Status: **Active***
*Version: 2.0 (Phase 2: Autonomous Operations)*
*Date: February 2026*

## 1. Executive Summary

Minerva Phase 2 evolves the platform from a reactive data analysis tool to a **proactive, autonomous agentic workforce**. While the core engine (Phase 1) remains responsible for the heavy lifting of data processing and modeling, the new **Agent Layer** sits atop it, orchestrating workflows, maintaining the codebase, and proactively finding insights.

The system follows a **"Glass Box"** philosophy: strictly typed, observable, and debuggable execution, replacing "black box" AI magic with transparent, loggable planning steps.

---

## 2. High-Level Architecture

The system is composed of three primary layers:

### Layer 1: The Core Engine (Deterministic & Robust)

*The muscle.* Handles data ingestion, cleaning, training, and deterministic logic.

* **Entry Points**: `main.py` (API), `ui/dashboard.py` (Frontend), `cli.py` (Command Line).
* **Data Layer**: `DataPreprocessor`, `SemanticEnricher`, `Integration.semantic_merge`.
* **Modeling**: `AnalyzerSelector`, `ModelSelector`, `modeling/analyzers/*.py`.
* **State**: `local_data/` (Files), `storage/session_db.py` (SQLite).

### Layer 2: The Cognitive Core (Planning)

*The brain.* Decides *how* to solve a user request.

* **Cascade Planner** (`orchestration/cascade_planner.py`): The retrieval-augmented planner that breaks goals into steps (`FILTER`, `TRANSFORM`, `VISUALIZE`).
* **Plan Learner** (`orchestration/plan_learner.py`): Records successful execution paths to speed up future queries.
* **Tool Registry** (`orchestration/tool_registry.py`): A strictly typed library of atomic actions open to the planner.

### Layer 3: The Agent Layer (Autonomous Workforce)

*The workers.* Operates asynchronously to maintain the system and generate value.

* **Conductor** (`agents/conductor.py`): The boss. Wakes up daily, coordinates other agents.
* **Engineer** (`agents/engineer.py`): The builder. Scans code, identifies technical debt, proposes fixes.
* **Scheduler** (`agents/scheduler.py`): The analyst. Runs recurring data reports.

#### D. Productizer Agent (`agents/productizer.py`)

* **Role**: Vertical Use Case Generator.
* **Trigger**: Manual / On-Demand.
* **Workflow**:
    1. Analyzes Dataset Profile + Run Outputs.
    2. Matches to Vertical Template (Retail, Finance, Healthcare).
    3. Generates "MVP Plan" (Workflow, Outputs, Next Questions).
    4. Saves to Knowledge Base for user customization.

* **Sentinel** (`agents/sentinel.py`): The guard. Runs tests and linting on changes.
* **Advocate** (`agents/advocate.py`): The voice. Manages user issues and feedback.

---

## 3. The Agent Layer Detail

The Agent Layer is designed as a **Multi-Agent System (MAS)** where agents communicate via a shared memory and artifact store.

### 3.1. Infrastructure (`agents/base.py`)

All agents inherit from `BaseAgent` and share:

* **Memory**: `AgentMemory` (SQLite) for storing state, key/value pairs, and history.
* **Workspace**: Sandbox filesystem access.
* **LLM Access**: Shared `LLMClient` with rate limiting and cost tracking.

### 3.2. Agent Roles

| Agent | Responsibility | Triggers | Outputs |
| :--- | :--- | :--- | :--- |
| **Conductor** | Orchestration & Reporting. | Daily Cron | `daily_briefing.md` |
| **Engineer** | Code Quality & Refactoring. | Manual / Conductor | `gap_analysis.md`, PRs |
| **Scheduler** | Report Automation. | Scheduled Interval | `Analysis Result`, Email (Planned) |
| **Sentinel** | CI/CD, Testing, Security. | Pre-Commit, Pre-Run | `qa_report.md`, Blocked Exec |
| **Advocate** | User Feedback Loop. | Issue Creation | GitHub Issues, Feature reqs |
| **Productizer** | Vertical Solution Planning. | Manual | `mvp_plan.md` |

### 3.3. Inter-Agent Communication

Agents do not call each other directly (to avoid coupling). They communicate via:

1. **Artifacts**: One agent writes a file (e.g., `gap_analysis.md`), another reads it.
2. **Conductor Loop**: The Conductor invokes agents sequentially based on their readiness state.

---

## 4. Orchestration & Planning (The "Brain")

Minerva uses **Hierarchical Planning** rather than a single monolithic prompt.

### 4.1. Intent Detection

Logic in `ui/chat_logic.py` and `orchestration/cascade_planner.py`:

1. **Keyword Heuristics**: "Show me sales" -> `VISUALIZATION`.
2. **LLM Classification**: Fallback for complex queries ("Why did revenue drop?").
3. **Ambiguity Check**: Ask clarifying questions if confidence is low.

### 4.2. The Cascade Planner

Instead of writing Python code directly, the Planner generates a **JSON Plan**:

```json
[
  {"tool": "load_data", "args": {"file": "sales.csv"}},
  {"tool": "filter_rows", "args": {"column": "Region", "value": "West"}},
  {"tool": "visualize", "args": {"type": "bar_chart"}}
]
```

Crucially, this plan is **validated** against the `ToolRegistry` *before* execution, preventing hallucinated function calls.

### 4.3. The "Glass Box" Execution

Execution is transparent. Every step is logged to the UI (`st.status`):

* ✅ **Plan**: "Filtering data by Region..."
* ✅ **Code**: `df = df[df['Region'] == 'West']`
* ✅ **Verify**: "Rows remaining: 450"

---

## 5. Memory & Knowledge Systems

Minerva possesses "Teachable Memory" allowing it to learn user preferences.

* **Vector Store** (`learning/vector_store.py`): Stores user feedback, business heuristics (e.g., "Fiscal year starts in Oct"), and successful past plans.
* **Plan Cache** (`orchestration/plan_learner.py`): Maps `Query Hash` -> `Successful Plan JSON`. If a query is repeated, it skips the LLM and runs the cached plan.
* **Semantic Index** (`catalog/semantic_index.db`): Stores metadata about public datasets (files in `datasets/`), managing column roles (e.g., this column is a "Zip Code") for join logic.

---

## 6. Data Management

### 6.1. Semantic Merge 2.0 (`Integration/semantic_merge.py`)

Minerva automatically joins user data with public context (Census, Weather, Economics).

* **Discovery**: Scans `datasets/` and builds an index of column *meanings* (Roles).
* **Matching**: When a user uploads data, it infers roles (e.g., "Postcode" -> `zip_code`).
* **Join**: Automatically joins on roles (Start Date + Zip Code) rather than identical column names.

### 6.2. Data Intake (`Data_Intake/`)

* **S3/API**: CLI tools to fetch data from remote sources.
* **Local**: Simple drag-and-drop in UI.

---

## 7. Security & Sandboxing

* **Code Execution**: User-generated code runs in `exec()` but with a **restricted globals** dictionary. Dangerous imports (`os`, `sys`, `subprocess`) are blocked unless explicitly allowlisted for specific "System Agents" (like Engineer).
* **Data Privacy**: All data remains local. LLM calls send only schema/metadata, never row-level PII (unless explicitly opted-in for specific analysis tasks).
* **Dependency Management**: Strict `requirements.txt` and `environment.yml` lockfiles.

---

## 8. Directory Structure

```text
Minerva/
├── agents/                 # [PHASE 2] Autonomous Agents
│   ├── base.py
│   ├── conductor.py
│   └── ...
├── catalog/                # Semantic Data Registry
├── config/                 # Configuration & Feature Flags
├── Data_Intake/            # Ingestion Scripts
├── docs/                   # Documentation
├── Integration/            # Data Joining Logic (Semantic Merge)
├── modeling/               # [PHASE 1] ML Analyzers
├── orchestration/          # [PHASE 2] Planning & Tools
│   ├── cascade_planner.py
│   └── tool_registry.py
├── preprocessing/          # Data Cleaning & Metadata
├── ui/                     # Streamlit Frontend
├── utils/                  # Shared Helpers
├── main.py                 # FastAPI Entry Point
├── cli.py                  # CLI Entry Point
└── tests/                  # Pytest Suite
```

## 9. Future Roadmap

1. **Distributed Agents**: Move agents to separate containers/processes.
2. **Plugin System**: Allow users to write custom Agents.
3. **Active Learning**: Agents that ask *proactive* questions to fill knowledge gaps.

# Minerva: The Local, Trainable AI Analyst

Minerva is a privacy-first data analysis platform that runs **entirely on your local machine**.
It combines the power of **Large Language Models (LLMs)** with **Deterministic Python Code Execution** to give you an AI analyst that you can actually trust.

![Dashboard](https://via.placeholder.com/800x400?text=Minerva+Dashboard+Screenshot)

## 🚀 Why Minerva?

### 1. 🔒 Privacy First (Local)

Unlike web-based tools (ChatGPT, Julius AI), Minerva runs on your hardware.

- **No data leaves your machine.**
- Great for sensitive financial, healthcare, or proprietary data.
- Supports **GPU Acceleration** (CUDA) for real-time performance.

### 2. 🧠 Trainable Memory (The "Teach" Button)

Minerva doesn't just chat; it learns.

- Use **Teaching Mode** to save specific business logic (e.g., "This is how we calculate Churn").
- These skills are stored in a local **Vector Database** (`.vector_store`).
- Next time you ask, it retrieves *your* specific formula, not a generic one.

### 3. 🔌 Auto-Enrichment

Stop downloading CSVs manually. Minerva automatically connects your data to public sources:

- **World Bank** (GDP, Population)
- **FRED** (Interest Rates, Inflation)
- **Census Bureau** (Demographics)
- **Alpha Vantage** (Stock Prices)

It uses **Semantic Matching** to find the right join keys (Zip/FIPS/ISO) automatically.

---

## 🛠️ Installation

### Prerequisites

- **OS**: Windows, Linux, or Mac
- **Python**: 3.10+
- **GPU (Recommended)**: NVIDIA GPU with CUDA 12+ (for fast inference)

### Quick Start

1. **Clone & Install**

    ```powershell
    git clone https://github.com/your-repo/minerva.git
    cd minerva
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r config/requirements.txt
    ```

2. **Enable GPU (Optional but Recommended)**
    If you have an NVIDIA GPU, install the CUDA-enabled engine:

    ```powershell
    $env:CMAKE_ARGS="-DGGML_CUDA=on"
    pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
    ```

3. **Run the Dashboard**

    ```powershell
    python -m streamlit run ui/dashboard.py
    ```

---

## 🏗️ Architecture

Minerva follows a modular "Model Context Protocol" (MCP) design:

- **`ui/`**: User Interfaces (Streamlit)
  - `dashboard.py`: Main entry point.
  - `chat_logic.py`: "Smart Routing" intent classifier (Code vs. Text).
  - `teach_logic.py`: Interface for saving new skills to memory.
- **`llm_manager/`**: The Brain.
  - `subprocess_manager.py`: Runs Llama-3 in a detached process (prevents UI freezing).
  - `providers/local.py`: Wraps `llama_cpp` execution.
- **`learning/`**: Long-Term Memory.
  - `vector_store.py`: SQLite + FastEmbed for storing user-taught skills.
  - `interaction_logger.py`: Implicitly learns from your usage.
- **`public_data/`**: External Connectors.
  - `connectors/`: Clients for World Bank, FRED, etc.
- **`orchestration/`**: The "Do-er".
  - `analysis_router.py`: Decides which statistical model to run.
  - `llm_dynamic_analyzer.py`: Generates safe Pandas code on the fly.

---

## 💡 Key Features

| Feature | Description | Status |
| :--- | :--- | :--- |
| **Smart Routing** | Detects if you want a Chart (`viz`), a Calculation (`analysis`), or Text (`info`). | ✅ Active |
| **Code Sandboxing** | All LLM-generated code is scanned for dangerous imports (`os`, `sys`) before running. | ✅ Active |
| **Context Window** | Automatically injects dataframe schema + sample rows so the LLM knows your data. | ✅ Active |
| **Project Memory** | "Teach" specific SQL queries or formulas once, use forever. | ✅ Active |

---

## ⚠️ Commercial Licensing

Minerva is **Open Core** software.

- **Personal License**: Free for individuals and students.
- **Commercial License**: Required for business use.
- **Enterprise Edition**: Includes Team Skill Sync (share your vector store) and SSO.

*Built with ❤️ (and Llama 3) by the Minerva Team.*

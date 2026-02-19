# Use Cases

## Data Ingestion & Auto-Analysis

- Upload a CSV via the Streamlit dashboard or connect via REST API / public data connector
- Assay auto-detects column types, runs data quality scoring, selects the best analyzer (regression, classification, clustering, anomaly, descriptive), and returns metrics + insights
- LLM-powered dynamic analysis generates custom preprocessing and modeling code when available, with deterministic fallback

## Natural Language Querying

- Ask questions in plain English via the NL Query panel (e.g., "What is the average sales by region?")
- The system generates Pandas code via RAG-augmented LLM, executes it in a sandbox, and returns results with natural language explanations
- Previously taught examples are retrieved via vector similarity search to improve accuracy

## Smart Visualization

- Request any chart type via the chatbot (e.g., "Show me a scatter plot of price vs volume")
- Smart Charts recommends optimal chart types based on column profiling (temporal, categorical, numeric, geographic)
- Charts rendered via Plotly with auto-detected axes and color dimensions

## Autonomous Data Discovery & Enrichment

- Describe data needs in natural language (e.g., "I need US unemployment data")
- The API Discovery Agent searches 20+ curated APIs, applies Kaggle-derived vertical weighting, and fetches data
- When no known API matches, a dynamic connector is auto-generated from API documentation
- Semantic join key detection (ZIP/FIPS/ISO) automatically merges external data with user datasets

## Public Data Enrichment

- One-click enrichment from FRED (economic indicators), World Bank (global development), Census Bureau (demographics), Alpha Vantage (stock prices), Google Sheets, and OneDrive
- Data Fabric tab shows a NetworkX lineage graph visualizing how user data connects to public sources

## Teachable Memory

- Use "Teach Mode" to save domain-specific formulas, queries, and business logic
- Stored in a local vector store (SQLite + cosine similarity) and recalled during future queries
- Teaching a churn formula once means every future "what's the churn?" query uses your specific definition

## MCP-Powered Agent Integration

- Start the MCP server (stdio or HTTP) for direct integration with Claude Desktop, MCP Inspector, or custom agents
- All Assay capabilities are exposed as structured MCP tools with JSON-RPC 2.0 protocol
- Session-based state management enables multi-step agent workflows

## Workflow Reruns & Session Replay

- Use `POST /sessions/{run_id}/rerun` to re-execute a saved workflow with modified parameters
- Run Artifact Store enables replay of failed executions for debugging
- Session history accessible via sidebar and API

## Diagnostics & Data Quality

- Automatic data quality scoring: missing data, outlier detection (z-score), type consistency, value range analysis
- Distribution drift detection between original and processed data
- Imputation confidence scoring with misaligned row detection
- High-risk column identification for model reliability warnings

## Post-Analysis Actions

- Action agent sends notifications (email/SNS) and logs results to SQLite
- Optional agent recipes trigger anomaly alerts, reorder suggestions, or role review requests
- Extensible via custom recipes without altering core logic

## Secure Credential Management

- API keys stored with Fernet AES-128 encryption (PBKDF2 key derivation)
- Master password unlocking with environment variable fallback
- Kaggle credential integration for dataset-driven source weighting

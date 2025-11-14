# Use Cases

- Upload + Auto Analysis
  - Upload a CSV via the UI, auto-select analyzer, get metrics + insights.

- Public Data Enrichment
  - Merge with public datasets using integration.semantic_merge and view enrichment report.

- Rerun with Persisted Model
  - Use POST /sessions/{run_id}/rerun to load saved model and recompute metrics.

- Drift & Imputation Diagnostics
  - Run tools/alignment_drift_monitor.py for drift check between batches.
  - Run examples/imputation_confidence_demo.py to score imputed values.

- Chatbot-driven Visualization
  - Use the dashboard chatbot to request a specific visualization; falls back to heuristic when LLM disabled.

# UVP Summary

**Self-configuring analytics** – The system automatically selects the most suitable analysis strategy based on dataset characteristics. `analysis_selector.select_analyzer` evaluates each registered `Analyzer` and chooses the best fit using heuristics.

**Local LLM powered enrichment** – Preprocessing utilities load a local Llama‑based model when available. Functions such as `preprocess_data_with_llm` and `analyze_dataset_with_llm` augment the data without external calls.

**Semantic data integration** – `rank_and_merge` locates candidate tables via a semantic index, synthesizes join keys (e.g. zip to FIPS mapping or hashed city/state) and merges the best table.

**Automated action agents** – After modeling, `agents.action_agent.execute_actions` records outcomes, emails notifications and can trigger optional recipes (reorder suggestions, anomaly alerts or role review requests).

**Tier‑based gating and usage tracking** – The workflow prevents free‑tier reruns and enforces request/size limits using `get_user_tier` and usage checks.

# Differentiators

- **Dynamic Analyzer Registry** – Regression, classification, clustering, anomaly and descriptive analyzers each report a `suitability_score` so the workflow can adapt to any dataset.
- **Offline LLM integration** – Local inference with llama_cpp allows dataset tagging, similarity scoring and modeling suggestions even without cloud connectivity.
- **Semantic join heuristics** – Merging public data uses role-driven matches, hashed join keys and ranking by new column gain for the best enrichment.
- **Extensible agent recipes** – Post‑processing hooks execute custom API calls when specific conditions are met, enabled only when `ENABLE_RECIPES` is set.
- **User tier enforcement** – Free users face limits on reruns and data size, providing a built‑in upgrade path.

# Competitive Advantage

This combination of automatic model selection, offline LLM enrichment and semantic merges drastically reduces manual preprocessing. Competitors would need robust local LLM support and a curated semantic index to replicate the same self‑configuring workflow. The modular agent recipes enable integration with external operational systems without altering core logic, providing extensibility that is hard to retrofit.

# Opportunities to Strengthen

- Expand the local LLM modules with more advanced prompt templates and multilingual support.
- Broaden the semantic index to cover additional industries and file types.
- Enhance the free‑tier gating with customizable plans or credit systems.
- Provide UI tooling for creating new agent recipes and monitoring their execution.


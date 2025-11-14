# Semantic Merge 2.0

This document outlines the improved data enrichment workflow.

## Design Rationale

The original merging strategy relied only on column name overlap. When a user
upload lacked matching names or contained only a handful of rows, the system
failed to join reference tables and produced weak models.

Semantic Merge 2.0 indexes the meaning—or *role*—of each column. Reference
tables in the `datasets/` directory are scanned and catalogued. Each column is
assigned a role such as `transaction date` or `store id` along with uniqueness
statistics. During ingestion, the uploaded file's columns are inferred in the
same way. Matching now occurs on these roles rather than raw names, allowing
joins even when the headers differ.

If no common key exists, synthetic join keys are created using small hash
functions. The merge process logs the entire provenance, including which
reference tables joined and validation metrics before and after enrichment. The
report is saved as `merge_report.json` next to the trained model.

## Refreshing the Index

The semantic index is a lightweight SQLite database. Refresh it whenever new
reference datasets are added:

```bash
python -m Data_Intake.datalake_ingestion s3 --bucket my-bucket --prefix retail/ --dest datasets
```

Both `s3` and `api` subcommands call `catalog.semantic_index.build_index` after
downloading files. Rebuilding the index is safe and quick—existing entries are
replaced automatically.

## Environment Variables

Three environment variables govern when the optional role review flow is
triggered:

- `MIN_R2` – Minimum acceptable R² value (default `0.25`).
- `MAX_MAPE` – Maximum acceptable mean absolute percentage error (default `30`).
- `MIN_ROWS` – Minimum validation rows before thresholds apply (default `500`).

When a model falls outside these bounds, the dashboard shows a banner inviting
the user to clarify column roles. If they opt in, the workflow reruns the merge
and training using the updated roles.

## Fallback Behaviour

Semantic Merge is backwards compatible. If the semantic index does not exist or
fails to load, the system reverts to the original name-based merging logic. This
ensures existing workflows continue to function while the new index is being
populated.

# Semantic Index

The semantic index catalogues the roles of every column found in the `datasets/` directory. It is stored as a lightweight SQLite database (`catalog/semantic_index.db`) and is used by the merge workflow to locate reference tables with matching semantics.

## How the Index Is Built

`catalog.semantic_index.build_index` scans each file in the datasets directory and uses `preprocessing.metadata_parser.infer_column_meta` to infer a role for every column. The role and its uniqueness statistic are stored under the `column_stats` table of the SQLite database.

The indexing step runs automatically whenever `Data_Intake.datalake_ingestion` downloads new data, but it can also be invoked manually:

```bash
python -m catalog.semantic_index build_index
```

(see the command below for the typical rebuild workflow)

## Adding Curated Datasets

Simply place CSV, JSON or Excel files in the `datasets/` folder or point the ingestion CLI to an external source. When the index is rebuilt, any new files will have their columns analysed and recorded in the database.

## Roles and Column Matching

Roles are defined in `config/semantic_roles.yaml`. During ingestion each column name and optional description is mapped to one of these roles. Later, `Integration.semantic_integration` queries the index by role to find candidate tables for enrichment. Matching is therefore independent of the original column names and relies purely on the assigned roles.

## Rebuilding the Index

To refresh the semantic index after adding or updating datasets, run:

```bash
python -m Data_Intake.datalake_ingestion s3 --bucket <my-bucket> --prefix <path> --dest datasets
```

or use the `api` subcommand with an endpoint. Both subcommands will call `build_index` once the data is downloaded.

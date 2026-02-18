# Minerva Architecture Overview

## Modules

### main.py
#### Imports
- argparse
- config
- fastapi
- json
- orchestrate_workflow
- os
- prometheus_client
- pydantic
- storage
- utils
- uvicorn

### adm\__init__.py
#### Imports
- orchestrate_workflow
- orchestration

### agents\action_agent.py
#### Imports
- __future__
- agent_recipes
- config
- datetime
- email
- json
- os
- re
- smtplib
- sqlite3
- typing
- utils

### agents\__init__.py

### agent_recipes\anomaly_alert.py
#### Imports
- __future__
- config
- logging
- typing
- utils

### agent_recipes\reorder.py
#### Imports
- __future__
- config
- logging
- typing
- utils

### agent_recipes\role_review.py
#### Imports
- __future__
- config
- feedback
- logging
- typing
- utils

### agent_recipes\__init__.py
#### Imports
- anomaly_alert
- reorder
- role_review

### catalog\semantic_index.py
#### Imports
- boto3
- config
- os
- pandas
- pathlib
- preprocessing
- sqlite3
- typing

### catalog\__init__.py

### categorization\categorize.py
#### Imports
- pandas
- typing

### categorization\__init__.py

### chatbot\chatbot.py
#### Imports
- decision
- dotenv
- orchestration
- os
- streamlit

### chatbot\decision.py
#### Imports
- __future__
- intent_parser
- typing

### chatbot\intent_parser.py
#### Imports
- 
- logging
- spacy
- typing
- utils

### chatbot\llm_intent_classifier.py
#### Imports
- config
- functools
- json
- logging
- preprocessing
- typing

### chatbot\__init__.py

### config\feature_flags.py
#### Imports
- 

### config\model_allowlist.py

### config\__init__.py
#### Imports
- boto3
- dotenv
- feature_flags
- os
- pathlib
- utils
- yaml

### Data_Intake\datalake_ingestion.py
#### Imports
- __future__
- argparse
- boto3
- catalog
- os
- pathlib
- utils

### Data_Intake\test_data_intake.py
#### Imports
- io
- os
- pandas
- pytest
- storage

### Data_Intake\__init__.py
#### Imports
- datalake_ingestion
- storage

### debug\check_path.py
#### Imports
- sys

### descriptive\stats.py
#### Imports
- __future__
- pandas

### descriptive\__init__.py

### examples\imputation_confidence_demo.py
#### Imports
- numpy
- pandas
- scripts

### examples\selenium_decode_demo.py
#### Imports
- PIL
- selenium

### examples\test_workflow_demo.py
#### Imports
- orchestration
- pandas

### examples\__init__.py

### feedback\ratings.py
#### Imports
- datetime
- json
- os
- pathlib
- utils

### feedback\role_corrections.py
#### Imports
- json
- os
- pandas
- pathlib
- preprocessing
- typing
- utils

### feedback\__init__.py
#### Imports
- ratings
- role_corrections

### feedback\tests\test_ratings.py
#### Imports
- feedback
- json

### feedback\tests\test_role_corrections.py
#### Imports
- feedback
- pandas
- preprocessing

### Integration\data_integration.py
#### Imports
- pandas

### Integration\semantic_integration.py
#### Imports
- Integration
- __future__
- pandas
- pathlib
- preprocessing
- typing
- warnings

### Integration\semantic_merge.py
#### Imports
- __future__
- catalog
- config
- pandas
- pathlib
- preprocessing
- typing
- utils

### Integration\test_data_integration.py
#### Imports
- Integration
- storage

### Integration\__init__.py

### modeling\anomaly.py
#### Imports
- __future__
- pandas
- sklearn

### modeling\anomaly_analyzer.py
#### Imports
- 
- __future__
- analyzers
- anomaly
- interfaces
- logging
- pandas
- typing
- typing_extensions
- utils

### modeling\baseline_runner.py
#### Imports
- __future__
- config
- numpy
- pandas
- sklearn
- time
- typing

### modeling\classification.py
#### Imports
- __future__
- pandas
- sklearn

### modeling\classification_analyzer.py
#### Imports
- analyzers

### modeling\clustering.py
#### Imports
- __future__
- pandas
- sklearn

### modeling\cluster_analyzer.py
#### Imports
- analyzers

### modeling\descriptive_analyzer.py
#### Imports
- 
- __future__
- descriptive
- interfaces
- logging
- pandas
- typing
- typing_extensions
- utils

### modeling\forecasting.py
#### Imports
- __future__
- pandas
- statsmodels

### modeling\interfaces.py
#### Imports
- __future__
- abc
- pandas

### modeling\model_selector.py
#### Imports
- config
- lightgbm
- numpy
- os
- pandas
- pycaret
- sklearn
- xgboost

### modeling\model_training.py
#### Imports
- config
- modeling
- pandas
- pathlib
- pickle
- preprocessing
- shap
- utils

### modeling\regression_analyzer.py
#### Imports
- analyzers

### modeling\suitability_check.py
#### Imports
- __future__
- config
- pandas
- typing

### modeling\test_model_training.py
#### Imports
- modeling
- pandas
- pytest

### modeling\__init__.py
#### Imports
- analyzers

### modeling\analyzers\anomaly.py
#### Imports
- 
- __future__
- anomaly
- interfaces
- logging
- pandas
- typing
- typing_extensions
- utils

### modeling\analyzers\classification.py
#### Imports
- 
- __future__
- classification
- interfaces
- logging
- pandas
- typing
- typing_extensions
- utils

### modeling\analyzers\cluster.py
#### Imports
- 
- __future__
- clustering
- interfaces
- logging
- pandas
- typing
- typing_extensions
- utils

### modeling\analyzers\descriptive.py
#### Imports
- 
- __future__
- descriptive
- interfaces
- logging
- pandas
- typing
- typing_extensions
- utils

### modeling\analyzers\regression.py
#### Imports
- 
- __future__
- interfaces
- logging
- model_selector
- pandas
- typing
- typing_extensions
- utils

### modeling\analyzers\__init__.py
#### Imports
- anomaly
- classification
- cluster
- descriptive
- regression

### orchestration\agent_trigger.py
#### Imports
- __future__
- agents
- preprocessing
- typing

### orchestration\analysis_router.py
#### Imports
- __future__
- config
- dataclasses
- preprocessing
- typing
- utils

### orchestration\analysis_selector.py
#### Imports
- modeling
- pandas
- typing
- utils

### orchestration\analyzer_selector_helper.py
#### Imports
- Integration
- __future__
- config
- logging
- modeling
- orchestrate_workflow
- orchestration
- os
- output
- pandas
- pathlib
- preprocessing
- sklearn
- typing
- uuid

### orchestration\data_preprocessor.py
#### Imports
- __future__
- logging
- orchestrate_workflow
- output
- pandas
- pathlib
- pickle
- preprocessing
- scripts
- sklearn
- storage
- typing
- utils

### orchestration\metadata_cache.py
#### Imports
- __future__
- pandas
- preprocessing
- typing

### orchestration\orchestrate_workflow.py
#### Imports
- __future__
- config
- json
- logging
- modeling
- orchestration
- pandas
- pathlib
- preprocessing
- storage
- utils

### orchestration\orchestrator.py
#### Imports
- __future__
- chatbot
- feedback
- orchestration
- scripts
- typing

### orchestration\output_generator.py
#### Imports
- __future__
- orchestrate_workflow
- output
- pandas
- pathlib
- preprocessing
- storage
- sys
- typing

### orchestration\semantic_enricher.py
#### Imports
- __future__
- logging
- pandas
- preprocessing
- tagging
- typing

### orchestration\workflow_manager.py
#### Imports
- __future__
- logging
- orchestration
- pandas
- pathlib
- scripts
- typing

### orchestration\__init__.py

### output\output_formatter.py

### output\__init__.py

### preprocessing\advanced_schema_validator.py
#### Imports
- __future__
- pandas
- re
- typing

### preprocessing\column_meta.py
#### Imports
- dataclasses
- json
- logging
- pandas
- pathlib
- save_meta
- typing
- utils

### preprocessing\context_missing_finder.py
#### Imports
- __future__
- pandas
- typing

### preprocessing\data_categorization.py
#### Imports
- __future__
- json
- llm_preprocessor
- metadata_parser
- pandas
- pathlib
- typing

### preprocessing\data_cleaning.py
#### Imports
- pandas
- rapidfuzz
- scipy

### preprocessing\llm_analyzer.py
#### Imports
- preprocessing

### preprocessing\llm_preprocessor.py
#### Imports
- collections
- config
- huggingface_hub
- json
- llama_cpp
- logging
- pandas
- pathlib
- preprocessing
- scipy
- sklearn
- threading
- time
- torch
- transformers

### preprocessing\llm_summarizer.py
#### Imports
- config
- json
- preprocessing

### preprocessing\metadata_parser.py
#### Imports
- column_meta
- config
- pandas
- pathlib
- re
- typing
- utils
- yaml

### preprocessing\misaligned_row_detector.py
#### Imports
- __future__
- pandas
- re
- typing

### preprocessing\prompt_templates.py

### preprocessing\sanitize.py
#### Imports
- __future__
- config
- pandas
- re
- typing

### preprocessing\save_meta.py
#### Imports
- __future__
- column_meta
- hashlib
- json
- os
- pandas
- pathlib
- typing
- utils

### preprocessing\test_advanced_schema_validator.py
#### Imports
- pandas
- preprocessing

### preprocessing\test_context_missing_finder.py
#### Imports
- pandas
- preprocessing

### preprocessing\test_data_categorization.py
#### Imports
- pandas
- pathlib
- preprocessing
- pytest

### preprocessing\test_data_cleaning.py
#### Imports
- pandas
- preprocessing

### preprocessing\test_llm_cache.py
#### Imports
- config
- preprocessing
- threading
- time
- unittest

### preprocessing\test_llm_guardrails.py
#### Imports
- logging
- preprocessing

### preprocessing\test_llm_preprocessor.py
#### Imports
- pandas
- preprocessing
- pytest

### preprocessing\test_misaligned_row_detector.py
#### Imports
- pandas
- preprocessing

### preprocessing\__init__.py
#### Imports
- advanced_schema_validator
- context_missing_finder
- misaligned_row_detector

### preprocessing\tests\test_sanitize.py
#### Imports
- config
- importlib
- preprocessing

### Purchase_handler\purchase_handler.py
#### Imports
- boto3
- config
- flask
- orchestrate_workflow
- os
- pandas
- pathlib
- storage
- sys
- utils

### Purchase_handler\__init__.py

### Purchase_handler\tests\test_rerun_endpoint.py
#### Imports
- Purchase_handler
- modeling
- orchestrate_workflow
- pandas
- pathlib
- preprocessing
- pytest
- sklearn
- storage

### scripts\alignment_drift_monitor.py
#### Imports
- argparse
- collections
- json
- pandas
- re
- typing

### scripts\analyze_docs.py
#### Imports
- argparse
- pathlib
- re
- typing

### scripts\generate_architecture.py
#### Imports
- argparse
- ast
- json
- os
- pathlib
- typing

### scripts\imputation_confidence.py
#### Imports
- numpy
- pandas
- sklearn
- typing

### scripts\visualization_selector.py
#### Imports
- config
- json
- logging
- pandas
- preprocessing
- typing
- utils

### storage\base.py
#### Imports
- __future__
- abc
- pathlib

### storage\get_backend.py
#### Imports
- base
- local_backend
- os
- s3_backend

### storage\local_backend.py
#### Imports
- 
- __future__
- base
- categorization
- config
- datetime
- io
- json
- os
- pandas
- pathlib
- sqlite3
- typing
- utils

### storage\s3_backend.py
#### Imports
- __future__
- base
- boto3
- config
- os
- pathlib
- tempfile
- typing

### storage\session_db.py
#### Imports
- boto3
- datetime
- json
- os
- sqlite3
- typing

### storage\__init__.py
#### Imports
- get_backend
- local_backend
- pathlib

### tagging\category_manager.py
#### Imports
- __future__
- json
- pathlib
- typing

### tagging\test_category_manager.py
#### Imports
- pathlib
- tagging

### tagging\__init__.py

### tests\conftest.py
#### Imports
- boto3
- moto
- pathlib
- pytest
- storage
- sys

### tests\test case.py
#### Imports
- examples

### tests\test_action_agent_db.py
#### Imports
- agents
- json
- pathlib
- sqlite3

### tests\test_allowlist_models.py
#### Imports
- config
- modeling
- pandas
- pytest
- sys
- types

### tests\test_anomaly_stats.py
#### Imports
- descriptive
- modeling
- pandas
- pathlib
- sys

### tests\test_baseline_runner.py
#### Imports
- modeling
- numpy
- pandas

### tests\test_classification_clustering.py
#### Imports
- modeling
- pandas
- pathlib
- sklearn
- sys

### tests\test_cli_flags.py
#### Imports
- main
- os
- sys

### tests\test_config.py
#### Imports
- config
- pytest

### tests\test_csv_chunked_read.py
#### Imports
- config
- pandas
- pathlib
- storage
- types

### tests\test_dashboard_sections.py
#### Imports
- json
- pathlib
- streamlit
- sys

### tests\test_dashboard_summary.py
#### Imports
- contextlib
- importlib
- pathlib
- sys
- types

### tests\test_data_preprocessor_extended.py
#### Imports
- orchestration
- pandas
- scripts

### tests\test_decision.py
#### Imports
- chatbot
- pathlib
- pytest
- sys

### tests\test_dynamo_sessions.py
#### Imports
- boto3
- importlib
- json
- moto
- storage

### tests\test_end_to_end_workflow.py
#### Imports
- orchestrate_workflow
- os
- pathlib
- pytest
- storage
- sys

### tests\test_explanations_flag.py
#### Imports
- config
- importlib
- modeling
- numpy
- pandas
- sys
- types

### tests\test_health.py
#### Imports
- fastapi
- main
- pathlib
- sys

### tests\test_helper_classes.py
#### Imports
- orchestrate_workflow
- orchestration
- pandas

### tests\test_imputation_confidence.py
#### Imports
- numpy
- pandas
- scripts

### tests\test_integration_baseline.py
#### Imports
- config
- numpy
- orchestration
- pandas

### tests\test_integration_nomodel.py
#### Imports
- orchestration
- pandas

### tests\test_intent_parser.py
#### Imports
- chatbot
- pathlib
- pytest
- sys

### tests\test_llm_outputs.py
#### Imports
- pathlib
- preprocessing
- sys

### tests\test_logging_redaction.py
#### Imports
- importlib
- json
- logging
- utils

### tests\test_metadata_parser.py
#### Imports
- config
- pandas
- pathlib
- preprocessing
- sys

### tests\test_metrics_flag.py
#### Imports
- os
- pytest
- socket
- subprocess
- sys
- time

### tests\test_net.py
#### Imports
- pytest
- requests
- responses
- time
- utils

### tests\test_orchestration.py
#### Imports
- config
- orchestrate_workflow
- orchestration
- pandas
- pathlib
- storage

### tests\test_orchestration_purity.py
#### Imports
- json
- orchestration
- pandas
- sys

### tests\test_preprocess_no_llm.py
#### Imports
- pandas
- preprocessing

### tests\test_rerun_validation.py
#### Imports
- fastapi
- importlib
- main
- pytest
- storage

### tests\test_role_feedback.py
#### Imports
- feedback
- importlib
- pandas
- preprocessing

### tests\test_role_review_flow.py
#### Imports
- json
- numpy
- pandas
- pathlib
- preprocessing
- sys
- utils

### tests\test_router.py
#### Imports
- modeling
- orchestration
- pandas
- pytest

### tests\test_run_id_workflow.py
#### Imports
- json
- modeling
- orchestrate_workflow
- pandas
- pathlib
- preprocessing
- sklearn
- storage

### tests\test_run_metadata.py
#### Imports
- json
- storage

### tests\test_save_meta_versions.py
#### Imports
- pandas
- preprocessing
- time

### tests\test_security.py
#### Imports
- modeling
- pathlib
- pytest
- storage

### tests\test_selector.py
#### Imports
- modeling
- orchestration
- pandas
- pathlib
- pytest
- sys
- utils

### tests\test_selector_run.py
#### Imports
- orchestration
- pandas

### tests\test_semantic_index_cloud.py
#### Imports
- boto3
- catalog
- moto
- pandas

### tests\test_semantic_integration.py
#### Imports
- Integration
- catalog
- pandas
- pathlib
- preprocessing
- sys

### tests\test_sessions.py
#### Imports
- fastapi
- importlib
- json
- main
- pathlib
- pytest
- storage
- sys

### tests\test_session_history.py
#### Imports
- os
- pathlib
- sqlite3
- storage
- sys

### tests\test_storage_backends.py
#### Imports
- pathlib
- pytest
- storage

### tests\test_suitability_check.py
#### Imports
- modeling
- orchestration
- pandas

### tests\test_tier_access.py
#### Imports
- Purchase_handler
- importlib
- json
- orchestrate_workflow
- pandas
- pathlib
- storage

### tests\test_usage_limits.py
#### Imports
- orchestrate_workflow
- pandas
- pathlib
- storage
- utils

### tests\test_visualization_selector.py
#### Imports
- config
- pandas
- pathlib
- scripts
- sys

### tests\test_workflow.py
#### Imports
- examples

### tools\alignment_drift_monitor.py
#### Imports
- argparse
- json
- pandas
- scripts

### tools\check_dependency_drift.py
#### Imports
- __future__
- config
- packaging
- pathlib
- re
- sys
- yaml

### tools\__init__.py

### ui\column_review.py
#### Imports
- pandas
- preprocessing
- streamlit
- typing
- ui
- utils

### ui\dashboard.py
#### Imports
- chatbot
- faulthandler
- feedback
- hashlib
- json
- logging
- matplotlib
- orchestrate_workflow
- orchestration
- os
- pandas
- pathlib
- preprocessing
- storage
- streamlit
- ui

### ui\session_history.py
#### Imports
- json
- os
- pandas
- sqlite3
- streamlit
- ui

### ui\visualizations.py
#### Imports
- matplotlib
- pandas
- streamlit

### ui\__init__.py
#### Imports
- __future__
- config
- streamlit

### utils\key_mappers.py
#### Imports
- hashlib
- pandas
- typing

### utils\logging.py
#### Imports
- config
- logging
- pandas
- pathlib
- preprocessing
- re
- structlog
- typing

### utils\logging_config.py
#### Imports
- utils
- warnings

### utils\metrics.py
#### Imports
- config
- pandas
- prometheus_client

### utils\net.py
#### Imports
- random
- requests
- time
- typing

### utils\role_mapper.py
#### Imports
- __future__
- functools
- json
- numpy
- pathlib
- sentence_transformers
- sklearn
- typing

### utils\security.py
#### Imports
- __future__
- pathlib

### utils\usage_tracker.py
#### Imports
- __future__
- boto3
- config
- json
- os
- pathlib
- threading
- utils

### utils\user_profile.py
#### Imports
- json
- os
- pathlib
- utils

### utils\__init__.py

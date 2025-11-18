# Architecture & Design Guide

## Overview

This repository implements a data processing and automated modeling orchestration platform.  
Core features include dataset ingestion, cleaning and enrichment, automated model selection, result summarization, and dashboard generation. FastAPI endpoints expose past sessions and allow rerunning workflows.

## Application Entry Points

- **main.py** exposes REST endpoints using FastAPI. Endpoints list and rerun workflow sessions. Example endpoints:

  ```python
  @app.get("/sessions")
  def list_sessions(limit: int = 20):
      return session_db.list_sessions(limit)
  ```

  【F:main.py†L18-L21】

- **orchestrate_workflow.orchestrate_workflow** is the main workflow entry point. It uses a `WorkflowManager` to run each step:

  ```python
  def orchestrate_workflow(..., diagnostics_config=None):
      manager = WorkflowManager(...)
      result = manager.preprocess_and_cache(diagnostics_config)
      if result:
          return result
      manager.enrich_with_public_data()
      result = manager.select_and_run_analysis()
      if result:
          return result
      manager.generate_outputs()
      manager.trigger_agent_actions()
      return manager.result or {}
  ```

  【F:orchestrate_workflow.py†L22-L50】

## Orchestration Modules

- **WorkflowManager** coordinates preprocessing, enrichment, analysis, output generation and agent triggering. Key methods:

  ```python
  class WorkflowManager:
      ...
      def preprocess_and_cache(self, diagnostics_config=None) -> Optional[dict]:
          pre = DataPreprocessor()
          data, target, err, path = pre.run(...)
  ```

  【F:workflow_manager.py†L16-L63】

- **DataPreprocessor** loads and cleans the uploaded file, guesses the target column and validates the dataset.

  ```python
  class DataPreprocessor:
      def clean(self, data: pd.DataFrame) -> pd.DataFrame:
          ...
      def run(self, user_id: str, file_name: str, target_column: Optional[str], run_id: Optional[str]) -> tuple[Optional[pd.DataFrame], Optional[str], Optional[dict], Path]:
          ...
  ```

  【F:orchestration/data_preprocessor.py†L26-L85】

- **SemanticEnricher** performs dataset tagging and merges public data.

  ```python
  class SemanticEnricher:
      def enrich(self, df: pd.DataFrame, datalake_dfs: Optional[dict] = None, category: Optional[str] = None, ...):
          ...
  ```

  【F:orchestration/semantic_enricher.py†L21-L55】

- **AnalyzerSelector** selects the appropriate modeling strategy and orchestrates training and evaluation.

  ```python
  class AnalyzerSelector:
      def analyze(self, df: pd.DataFrame, target_column: str, datalake_dfs: Optional[dict] = None, ... ) -> Dict[str, Any]:
          ...
  ```

  【F:orchestration/analyzer_selector_helper.py†L22-L146】

- **OutputGenerator** writes predictions to disk, summarizes results with the local LLM and invokes the dashboard orchestrator.

  ```python
  class OutputGenerator:
      def generate(self, model, predictions, metrics: Dict[str, Any], model_info: Dict[str, Any], run_id: str, data: pd.DataFrame, target_column: str, needs_role_review: bool, file_name: str) -> Dict[str, Any]:
          ...
  ```

  【F:orchestration/output_generator.py†L14-L75】

- **AgentTrigger** invokes the action agent to send notifications or execute additional recipes.

  ```python
  class AgentTrigger:
      def trigger(self, result: Dict[str, Any], predictions, best_score: float, file_name: str, model, best_df, roles_dict) -> None:
          result["actions"] = execute_actions({...})
  ```

  【F:orchestration/agent_trigger.py†L9-L33】

## Preprocessing Layer

- **metadata_parser** extracts metadata and infers column roles.

  ```python
  def parse_metadata(df: pd.DataFrame) -> dict:
      return {
          "columns": df.columns.tolist(),
          "dtypes": df.dtypes.apply(lambda x: x.name).to_dict(),
          "summary": df.describe(include="all").to_dict(),
      }
  ```

  【F:preprocessing/metadata_parser.py†L1-L7】

- **data_cleaning** contains reusable cleaning utilities such as `convert_to_datetime`, `remove_outliers` and `encode_categorical_columns`.
- **llm_preprocessor** interfaces with a local LLM to preprocess text fields and compute dataset similarity scores.
- **data_categorization** generates tags for datasets and stores them in a lightweight catalog.
- **llm_summarizer.generate_summary** produces plain‑English summaries using an embedded model.

  ```python
  def generate_summary(data_stats: dict, model_results: dict, prompt: str):
      ...
      model = load_mistral_model(model_path)
      response = run_mistral_inference(model, input=json.dumps(payload), max_tokens=512, temperature=0.7)
      ...
  ```

  【F:preprocessing/llm_summarizer.py†L6-L37】

## Modeling Layer

- All modeling strategies implement the abstract `Analyzer` interface:

  ```python
  class Analyzer(ABC):
      @abstractmethod
      def suitability_score(self, df: pd.DataFrame) -> float: ...
      @abstractmethod
      def run(self, df: pd.DataFrame, **kwargs) -> dict: ...
  ```

  【F:modeling/interfaces.py†L6-L19】

- Concrete analyzer subclasses include:
  - `RegressionAnalyzer`
  - `ClassificationAnalyzer`
  - `ClusterAnalyzer`
  - `AnomalyAnalyzer`
  - `DescriptiveAnalyzer`
  The registry is defined in `modeling/__init__.py` mapping names to classes.
- Model selection uses `modeling.model_selector.select_best_model` for regression and related functions for classification/cluster tasks.
- The `analysis_selector.select_analyzer` helper scores analyzers and picks the best one.

## Integration Utilities

- **Data_Intake.datalake_ingestion** provides CLI helpers `sync_from_s3` and `fetch_from_api` to populate the local datasets folder. The functions also rebuild the semantic index.
- **Integration.semantic_integration.rank_and_merge** merges user data with public datasets based on inferred column roles.

  ```python
  def rank_and_merge(df_user: pd.DataFrame, column_meta: List[ColumnMeta], datasets_dir: Path | None = None) -> Tuple[pd.DataFrame, Dict]:
      ...
      return best_df, report
  ```

  【F:Integration/semantic_integration.py†L81-L117】
- **Integration.data_integration** has simple helpers for resolving field conflicts and merging additional data.

## Agents and Actions

- **agents.action_agent.execute_actions** logs results, sends optional email/SNS notifications and calls additional recipes when enabled.

  ```python
  def execute_actions(result: Dict[str, Any]) -> List[str]:
      actions: List[str] = []
      ...
      actions.append("notification_sent")
      ...
  ```

  【F:agents/action_agent.py†L31-L50】
- `AgentTrigger` in the orchestration layer packages workflow results and invokes this agent. Actions such as anomaly alerts can be plugged in via the optional `agent_recipes` module.

## Chatbot and Dashboarding

- The Streamlit chatbot interface interprets user questions and calls `orchestrate_dashboard`.

  ```python
  def chatbot_interface(data, visualizations, models, target=None, prefill=None):
      ...
      result = orchestrate_dashboard(...)
  ```

  【F:chatbot/chatbot.py†L6-L38】
- `orchestrator.orchestrate_dashboard` decides which visualization or model action to perform based on parsed chatbot intent.

  ```python
  def orchestrate_dashboard(data, model_output, model_type, target_variable, kernel_analysis, chatbot_input, ...):
      action, params = decide_action(chatbot_input)
      ...
      return {
          "dashboard": layout,
          "insights": insights,
          "confidence_score": f"The confidence in these results is {confidence_score}%.",
          ...
      }
  ```

  【F:orchestrator.py†L8-L112】

## Output Module

- `output.output_formatter` provides small helpers to convert predictions or analysis text into serializable structures.

  ```python
  def format_output(predictions):
      formatted_output = [{"prediction": pred} for pred in predictions]
      return formatted_output
  ```

  【F:output/output_formatter.py†L1-L3】

## Workflow Paths

### 1. From User Upload to Metadata Extraction

1. A user uploads a dataset. `DataPreprocessor.run` retrieves the file via `storage.get_backend` and validates size and user tier.
2. The dataset is cleaned using `DataPreprocessor.clean`, which normalizes text columns, converts date fields and removes outliers.
3. Column metadata is extracted via `parse_metadata` and roles are inferred with `infer_column_meta`.
4. `SemanticEnricher.enrich` scores the dataset against public tables, applies further cleaning and generates tags.

### 2. From Model Selection to Output Rendering

1. `AnalyzerSelector.analyze` chooses the appropriate analyzer using `analysis_selector.select_analyzer`. It may merge public data with the user dataset (`rank_and_merge`).
2. Training is delegated to `modeling.model_training.train_model`; evaluation uses `evaluate_model`.
3. Best predictions and metrics are passed to `OutputGenerator.generate`. Predictions are saved to disk and summarization occurs via `generate_summary` and `ask_follow_up_question`.
4. `orchestrate_dashboard` assembles the final dashboard layout and insights which are shown via the Streamlit chatbot or API response.
5. `AgentTrigger.trigger` finally executes agent actions like notifications.

## System Diagram

```text
User Upload
    |
    v
DataPreprocessor --clean--> SemanticEnricher
    |                           |
    |                           v
    |                     AnalyzerSelector
    |                           |
    v                           v
WorkflowManager ----------> Model Training
    |                           |
    v                           v
OutputGenerator ---------> orchestrate_dashboard
    |                           |
    v                           v
AgentTrigger <-------------- Chatbot UI
```

## Reusable Interfaces and Patterns

- The `Analyzer` abstract base class defines a standard interface for any analysis or modeling approach. Subclasses implement `suitability_score` and `run` and are registered in a central registry for selection.
- The orchestration layer (WorkflowManager + helpers) is composed of small classes with single responsibilities, making the workflow extensible.
- Agents use a simple contract (`execute_actions`) allowing custom automation recipes to hook into results.


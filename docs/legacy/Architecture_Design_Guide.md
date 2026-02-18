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

## Orchestration Layer

The orchestration layer coordinates the entire workflow through specialized helper classes.

### WorkflowManager (`orchestration/workflow_manager.py`)

Central coordinator that manages the five-stage workflow:

```python
class WorkflowManager:
    def __init__(
        self,
        user_id: str,
        file_name: str,
        target_column: Optional[str] = None,
        datalake_dfs: Optional[dict] = None,
        category: Optional[str] = None,
        user_labels: Optional[dict[str, str]] = None,
        run_id: Optional[str] = None,
    ):
        self.user_id = user_id
        self.file_name = file_name
        self.target_column = target_column
        self.run_id = run_id or str(uuid.uuid4())
        # ... initialization
    
    def preprocess_and_cache(self, diagnostics_config: Optional[Dict[str, Any]] = None) -> Optional[dict]:
        """Stage 1: Load, clean, validate dataset"""
        preprocessor = DataPreprocessor()
        data, target, error, path = preprocessor.run(
            self.user_id, self.file_name, self.target_column, self.run_id
        )
        if error:
            return error
        self.data = data
        self.target_column = target
        self.data_path = path
        return None
    
    def enrich_with_public_data(self) -> None:
        """Stage 2: Semantic enrichment with public datasets"""
        enricher = SemanticEnricher()
        self.data, self.enrichment_report = enricher.enrich(
            self.data, self.datalake_dfs, self.category, self.user_labels
        )
    
    def select_and_run_analysis(self, intent: Optional[dict] = None) -> Optional[dict]:
        """Stage 3: Select analyzer and run modeling"""
        selector = AnalyzerSelector()
        result = selector.analyze(
            self.data, self.target_column, self.datalake_dfs, self.user_labels, self.run_id
        )
        if "error" in result:
            return result
        self.analysis_result = result
        return None
    
    def generate_outputs(self) -> None:
        """Stage 4: Save models, predictions, generate summaries"""
        generator = OutputGenerator()
        self.result = generator.generate(
            model=self.analysis_result["model"],
            predictions=self.analysis_result["predictions"],
            metrics=self.analysis_result["metrics"],
            model_info=self.analysis_result["model_info"],
            run_id=self.run_id,
            data=self.data,
            target_column=self.target_column,
            needs_role_review=self.analysis_result.get("needs_role_review", False),
            file_name=self.file_name,
        )
    
    def trigger_agent_actions(self) -> None:
        """Stage 5: Execute automated actions (notifications, alerts)"""
        trigger = AgentTrigger()
        trigger.trigger(
            result=self.result,
            predictions=self.analysis_result["predictions"],
            best_score=self.analysis_result["metrics"].get("score", 0.0),
            file_name=self.file_name,
            model=self.analysis_result["model"],
            best_df=self.data,
            roles_dict=self.analysis_result.get("roles", {}),
        )
```

### DataPreprocessor (`orchestration/data_preprocessor.py`)

Handles data loading, cleaning, and validation:

- **Storage backend abstraction**: Supports local filesystem and S3
- **Tier-based quotas**: Enforces `MAX_REQUESTS_FREE`, `MAX_GB_FREE` limits
- **Data cleaning**: Removes outliers, normalizes text, converts dates
- **Target detection**: Auto-guesses target column if not provided
- **Metadata extraction**: Infers column roles (categorical, numeric, date, etc.)

```python
class DataPreprocessor:
    def run(
        self,
        user_id: str,
        file_name: str,
        target_column: Optional[str],
        run_id: Optional[str],
    ) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[dict], Path]:
        # Load from storage backend
        backend = get_backend()
        data = backend.load_data(user_id, file_name)
        
        # Clean and validate
        data = self.clean(data)
        
        # Infer target if not provided
        if not target_column:
            target_column = self._guess_target(data)
        
        # Extract metadata
        metadata = parse_metadata(data)
        
        return data, target_column, None, data_path
```

### SemanticEnricher (`orchestration/semantic_enricher.py`)

Enriches user data with relevant public datasets:

- **Column role inference**: Uses `infer_column_meta` to detect semantic roles
- **Dataset tagging**: Categorizes datasets for better discovery
- **Semantic merging**: Calls `rank_and_merge` from `integration/semantic_merge.py`

```python
class SemanticEnricher:
    def enrich(
        self,
        df: pd.DataFrame,
        datalake_dfs: Optional[dict] = None,
        category: Optional[str] = None,
        user_labels: Optional[dict[str, str]] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        # Infer column roles
        column_meta = infer_column_meta(df, user_labels)
        
        # Find and merge best matching dataset
        enriched_df, report = rank_and_merge(
            df, column_meta, datasets_dir=Path("datasets")
        )
        
        return enriched_df, report
```

### AnalyzerSelector (`orchestration/analyzer_selector_helper.py`)

Selects and runs the appropriate ML analyzer:

- **Modelability assessment**: Uses `assess_modelability` to check if modeling is viable
- **Analyzer selection**: Calls `select_analyzer` to pick best analyzer from registry
- **Iterative refinement**: Reruns with user-labeled metadata if initial results are poor
- **Fallback strategies**: Falls back to baseline models or descriptive stats if needed

```python
class AnalyzerSelector:
    def analyze(
        self,
        df: pd.DataFrame,
        target_column: str,
        datalake_dfs: Optional[dict] = None,
        user_labels: Optional[dict[str, str]] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Check if modeling is viable
        suitability = assess_modelability(df, target_column)
        if not suitability["is_modelable"]:
            # Fall back to descriptive analysis
            desc_analyzer = select_analyzer(df, preferred="DescriptiveAnalyzer")
            return desc_analyzer.run(df)
        
        # Select best analyzer
        analyzer = select_analyzer(df)
        
        # Run analysis
        result = analyzer.run(df, target_column=target_column)
        
        # Iterative refinement if needed
        if result["metrics"].get("r2", 0) < MIN_R2 and user_labels:
            # Rerun with user feedback
            result = analyzer.run(df, target_column=target_column, user_labels=user_labels)
        
        return result
```

### OutputGenerator (`orchestration/output_generator.py`)

Generates final outputs and summaries:

- **Model persistence**: Saves models to `local_data/models/<run_id>/best_model`
- **Prediction export**: Writes predictions to `output_files/<run_id>_predictions.csv`
- **LLM summarization**: Generates plain-English summaries using local LLM
- **Metadata logging**: Records run metadata to session database

```python
class OutputGenerator:
    def generate(
        self,
        model,
        predictions,
        metrics: Dict[str, Any],
        model_info: Dict[str, Any],
        run_id: str,
        data: pd.DataFrame,
        target_column: str,
        needs_role_review: bool,
        file_name: str,
    ) -> Dict[str, Any]:
        # Save model
        save_model(model, "best_model", run_id)
        
        # Save predictions
        output_path = Path("output_files") / f"{run_id}_predictions.csv"
        pd.DataFrame({"prediction": predictions}).to_csv(output_path, index=False)
        
        # Generate LLM summary
        summary = generate_summary(metrics, model_info)
        
        # Log metadata
        log_run_metadata(run_id, True, needs_role_review, file_name, model_info)
        
        return {
            "predictions": predictions,
            "metrics": metrics,
            "summary": summary,
            "model_path": str(Path("models") / run_id / "best_model"),
        }
```

### AgentTrigger (`orchestration/agent_trigger.py`)

Executes automated actions based on analysis results:

- **Action execution**: Calls `execute_actions` from `agents/action_agent.py`
- **Agent recipes**: Pluggable automation modules (anomaly alerts, role review, reorder)

```python
class AgentTrigger:
    def trigger(
        self,
        result: Dict[str, Any],
        predictions,
        best_score: float,
        file_name: str,
        model,
        best_df,
        roles_dict,
    ) -> None:
        actions = execute_actions({
            "result": result,
            "predictions": predictions,
            "score": best_score,
            "file_name": file_name,
        })
        result["actions"] = actions
```

## Preprocessing Layer

The preprocessing layer provides data cleaning, validation, and metadata extraction capabilities.

### Metadata Extraction (`preprocessing/metadata_parser.py`)

Extracts column metadata and infers semantic roles:

```python
def parse_metadata(df: pd.DataFrame) -> dict:
    """Extract basic metadata from dataframe"""
    return {
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.apply(lambda x: x.name).to_dict(),
        "summary": df.describe(include="all").to_dict(),
        "row_count": len(df),
        "memory_usage": df.memory_usage(deep=True).sum(),
    }

def infer_column_meta(
    df: pd.DataFrame,
    user_labels: Optional[dict[str, str]] = None,
) -> List[ColumnMeta]:
    """Infer semantic roles for each column"""
    # Uses heuristics and optional LLM to detect:
    # - Categorical vs numeric
    # - Date/time columns
    # - Geographic identifiers (zip, fips, city, state)
    # - ID columns
    # - Target candidates
    ...
```

### Data Cleaning (`preprocessing/data_cleaning.py`)

Reusable cleaning utilities:

- **`convert_to_datetime(df, columns)`**: Converts string columns to datetime
- **`remove_outliers(df, columns, method='iqr')`**: Removes statistical outliers
- **`encode_categorical_columns(df, columns)`**: Label encodes categorical features
- **`normalize_text(df, columns)`**: Lowercases and strips whitespace
- **`handle_missing_values(df, strategy='mean')`**: Imputes missing values

### Column Metadata (`preprocessing/column_meta.py`)

Defines the `ColumnMeta` dataclass for storing column information:

```python
@dataclass
class ColumnMeta:
    name: str
    role: str  # e.g., 'categorical', 'numeric', 'date', 'zip_code', 'fips_code'
    dtype: str
    missing_count: int
    unique_count: int
    sample_values: List[Any]
```

### LLM Preprocessor (`preprocessing/llm_preprocessor.py`)

Interfaces with local LLM for advanced preprocessing:

- **Text field preprocessing**: Extracts entities, sentiment, topics
- **Dataset similarity**: Computes semantic similarity between datasets
- **Column role inference**: Uses LLM to infer semantic roles when heuristics fail
- **Model caching**: Thread-safe model loading and caching

```python
def compute_dataset_similarity(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    method: str = "embedding",
) -> float:
    """Compute semantic similarity between two datasets"""
    # Uses sentence transformers to embed column names and sample values
    # Returns cosine similarity score
    ...
```

### LLM Summarizer (`preprocessing/llm_summarizer.py`)

Generates plain-English summaries of analysis results:

```python
def generate_summary(
    data_stats: dict,
    model_results: dict,
    prompt: str = None,
) -> str:
    """Generate natural language summary using local LLM"""
    # Loads Mistral model and generates summary
    # Uses structured prompt with data stats and model metrics
    ...
```

### Data Categorization (`preprocessing/data_categorization.py`)

Generates semantic tags for datasets:

```python
def categorize_dataset(df: pd.DataFrame) -> List[str]:
    """Generate category tags for dataset"""
    # Returns tags like: ['finance', 'time_series', 'geographic']
    # Stored in catalog for semantic search
    ...
```

### Schema Validation (`preprocessing/advanced_schema_validator.py`)

Validates data quality and schema consistency:

- **Type validation**: Ensures columns match expected types
- **Range validation**: Checks numeric values are within expected ranges
- **Pattern validation**: Validates strings match expected patterns (e.g., email, phone)
- **Referential integrity**: Checks foreign key relationships

### Sanitization (`preprocessing/sanitize.py`)

Sanitizes sensitive data before logging or display:

- **PII redaction**: Removes personally identifiable information
- **SQL injection prevention**: Escapes dangerous characters
- **Path traversal prevention**: Validates file paths

## Modeling Layer

The modeling layer provides a unified interface for different analysis strategies.

### Analyzer Interface (`modeling/interfaces.py`)

All analyzers implement the abstract `Analyzer` base class:

```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class Analyzer(ABC):
    """Abstract base class for all analysis strategies"""
    
    @abstractmethod
    def suitability_score(self, df: pd.DataFrame) -> float:
        """Return 0-1 score indicating how suitable this analyzer is for the dataset"""
        pass
    
    @abstractmethod
    def run(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Execute the analysis and return results"""
        pass
```

### Analyzer Registry (`modeling/__init__.py`)

Central registry mapping analyzer names to classes:

```python
from modeling.analyzers import (
    RegressionAnalyzer,
    ClassificationAnalyzer,
    ClusterAnalyzer,
    AnomalyAnalyzer,
    DescriptiveAnalyzer,
)

REGISTRY = {
    "RegressionAnalyzer": RegressionAnalyzer,
    "ClassificationAnalyzer": ClassificationAnalyzer,
    "ClusterAnalyzer": ClusterAnalyzer,
    "AnomalyAnalyzer": AnomalyAnalyzer,
    "DescriptiveAnalyzer": DescriptiveAnalyzer,
}
```

### Concrete Analyzers

#### RegressionAnalyzer (`modeling/analyzers/regression.py`)

Handles continuous target prediction:

- **Model selection**: Uses `select_best_model` to try multiple algorithms (LightGBM, XGBoost, RandomForest, LinearRegression)
- **Quality thresholds**: Requires `r2 >= MIN_R2` and `mape <= MAX_MAPE`
- **Feature engineering**: Automatic feature selection and encoding
- **Explainability**: SHAP values for model interpretation

#### ClassificationAnalyzer (`modeling/analyzers/classification.py`)

Handles categorical target prediction:

- **Binary and multiclass**: Supports both classification types
- **Model selection**: Tries LogisticRegression, RandomForest, XGBoost, LightGBM
- **Metrics**: Accuracy, precision, recall, F1, ROC-AUC
- **Class imbalance**: Handles imbalanced datasets with SMOTE

#### ClusterAnalyzer (`modeling/analyzers/cluster.py`)

Unsupervised grouping of similar records:

- **Algorithms**: KMeans, DBSCAN, hierarchical clustering
- **Optimal clusters**: Elbow method and silhouette analysis
- **Visualization**: 2D projection using PCA or t-SNE

#### AnomalyAnalyzer (`modeling/analyzers/anomaly.py`)

Detects outliers and anomalies:

- **Algorithms**: IsolationForest, LocalOutlierFactor, OneClassSVM
- **Anomaly scoring**: Assigns anomaly scores to each record
- **Threshold tuning**: Auto-tunes contamination parameter

#### DescriptiveAnalyzer (`modeling/analyzers/descriptive.py`)

Fallback for non-modelable datasets:

- **Summary statistics**: Mean, median, std, quartiles
- **Distribution analysis**: Histograms, density plots
- **Correlation analysis**: Pearson and Spearman correlations
- **Missing value analysis**: Patterns and recommendations

### Model Selection (`modeling/model_selector.py`)

Selects the best model for a given task:

```python
def select_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str = "regression",
) -> Tuple[Any, Dict[str, float]]:
    """Try multiple models and return the best one"""
    # Uses PyCaret for automated model selection
    # Returns best model and metrics
    ...
```

### Analyzer Selection (`orchestration/analysis_selector.py`)

Selects the best analyzer for a dataset:

```python
def select_analyzer(
    df: pd.DataFrame,
    preferred: Optional[str] = None,
) -> Analyzer:
    """Select best analyzer based on dataset characteristics"""
    if preferred and preferred in REGISTRY:
        return REGISTRY[preferred]()
    
    # Score each analyzer and return the best
    best_analyzer = None
    best_score = -1
    for analyzer_class in REGISTRY.values():
        analyzer = analyzer_class()
        score = analyzer.suitability_score(df)
        if score > best_score:
            best_score = score
            best_analyzer = analyzer
    
    return best_analyzer
```

### Suitability Check (`modeling/suitability_check.py`)

Assesses whether a dataset is suitable for modeling:

```python
def assess_modelability(
    df: pd.DataFrame,
    target_column: str,
) -> Dict[str, Any]:
    """Check if dataset is suitable for modeling"""
    # Checks:
    # - Minimum row count (> 50)
    # - Target variance (not all same value)
    # - Feature count (at least 1 feature)
    # - Missing value ratio (< 90%)
    return {
        "is_modelable": True/False,
        "reason": "...",
        "confidence": 0.0-1.0,
    }
```

## Integration Layer

The integration layer handles semantic merging of datasets and data lake management.

### Semantic Merge (`integration/semantic_merge.py`)

Automatically discovers and merges relevant public datasets:

```python
def rank_and_merge(
    df_user: pd.DataFrame,
    column_meta: List[ColumnMeta],
    datasets_dir: Path | None = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Find and merge the best matching enrichment dataset"""
    # 1. Load all available datasets from datasets_dir
    # 2. Score each dataset based on column role overlap
    # 3. Synthesize join keys (direct, zip_to_fips, city_state_hash)
    # 4. Perform left join with best dataset
    # 5. Return enriched dataframe and merge report
    ...

def synthesise_join_keys(
    df_user: pd.DataFrame,
    df_table: pd.DataFrame,
    user_meta: List[ColumnMeta],
    table_meta: List[Tuple[str, str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[str, str]], str]:
    """Automatically synthesize join keys between datasets"""
    # Strategies:
    # - Direct role match (e.g., both have 'fips_code')
    # - Zip to FIPS conversion
    # - City + State hashing
    # - Multi-role composite keys
    ...
```

### Data Lake Ingestion (`Data_Intake/datalake_ingestion.py`)

CLI tools for populating the local datasets folder:

```python
def sync_from_s3(bucket: str, prefix: str) -> None:
    """Download datasets from S3 to local datasets folder"""
    # Downloads CSV files and rebuilds semantic index
    ...

def fetch_from_api(api_url: str, dataset_name: str) -> None:
    """Fetch dataset from API and save locally"""
    # Fetches data, saves as CSV, updates semantic index
    ...
```

### Semantic Index (`catalog/semantic_index.py`)

Maintains a searchable index of available datasets:

- **SQLite database**: Stores dataset metadata and column roles
- **Embedding-based search**: Uses sentence transformers for semantic similarity
- **Automatic indexing**: Rebuilds index when new datasets are added

```python
def build_semantic_index(datasets_dir: Path) -> None:
    """Build searchable index of all datasets"""
    # Scans datasets_dir, extracts metadata, stores in _INDEX_DB
    ...

def search_datasets(
    query_meta: List[ColumnMeta],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Search for datasets matching query metadata"""
    # Returns list of (dataset_path, similarity_score)
    ...
```

## Agents and Actions

The agents layer provides automated actions triggered by analysis results.

### Action Agent (`agents/action_agent.py`)

Executes automated actions based on workflow results:

```python
def execute_actions(result: Dict[str, Any]) -> List[str]:
    """Execute automated actions based on analysis results"""
    actions: List[str] = []
    
    # Log results to database
    log_to_action_db(result)
    actions.append("logged_to_db")
    
    # Send notifications if configured
    if should_notify(result):
        send_notification(result)
        actions.append("notification_sent")
    
    # Execute agent recipes
    if result.get("needs_role_review"):
        trigger_role_review(result)
        actions.append("role_review_triggered")
    
    if result.get("anomalies_detected"):
        trigger_anomaly_alert(result)
        actions.append("anomaly_alert_sent")
    
    return actions
```

### Agent Recipes (`agent_recipes/`)

Pluggable automation modules:

#### Role Review Recipe (`agent_recipes/role_review.py`)

Triggers human review when column roles are uncertain:

- Creates review task in feedback system
- Notifies data stewards
- Tracks review status

#### Anomaly Alert Recipe (`agent_recipes/anomaly_alert.py`)

Sends alerts when anomalies are detected:

- Filters anomalies by severity threshold
- Formats alert message with context
- Sends via email/SNS/webhook

#### Reorder Recipe (`agent_recipes/reorder.py`)

Triggers reorder workflows for inventory predictions:

- Checks prediction against reorder threshold
- Creates purchase order
- Notifies procurement team

## Chatbot and Dashboarding

The UI layer provides interactive data exploration and visualization.

### Streamlit Dashboard (`ui/dashboard.py`)

Main dashboard interface:

- **File upload**: Drag-and-drop CSV upload
- **Session history**: View past analysis runs
- **Column review**: Interactive column role correction
- **Visualization gallery**: Auto-generated charts
- **Model insights**: Metrics, predictions, explanations

### Chatbot Interface (`chatbot/chatbot.py`)

Natural language interface for data exploration:

```python
def chatbot_interface(
    data: pd.DataFrame,
    visualizations: Mapping[str, Callable],
    models: Mapping[str, Callable],
    target: str = None,
    prefill: str = None,
) -> None:
    """Interactive chatbot for data exploration"""
    user_input = st.chat_input("Ask about your data...")
    
    if user_input:
        # Parse intent
        intent = parse_intent(user_input)
        
        # Orchestrate dashboard action
        result = orchestrate_dashboard(
            data=data,
            chatbot_input=user_input,
            visualizations=visualizations,
            models=models,
            target_variable=target,
        )
        
        # Display results
        st.write(result["insights"])
        st.plotly_chart(result["dashboard"])
```

### Dashboard Orchestrator (`orchestration/orchestrator.py`)

Decides which visualization or model action to perform:

```python
def orchestrate_dashboard(
    data: pd.DataFrame,
    model_output: Dict,
    model_type: str,
    target_variable: str,
    kernel_analysis: Dict,
    chatbot_input: str,
    visualizations: Mapping[str, Callable] = None,
    models: Mapping[str, Callable] = None,
) -> Dict[str, Any]:
    """Orchestrate dashboard based on user intent"""
    # Parse intent
    action, params = decide_action(chatbot_input)
    
    # Select visualization
    if action == "visualize":
        viz_type = select_visualization(data, params)
        layout = visualizations[viz_type](data, **params)
    
    # Generate insights
    insights = generate_insights(data, model_output, action)
    
    return {
        "dashboard": layout,
        "insights": insights,
        "confidence_score": f"Confidence: {model_output.get('confidence', 0):.1%}",
        "action": action,
    }
```

### Intent Parser (`chatbot/intent_parser.py`)

Parses natural language queries:

- **Entity extraction**: Identifies column names, values, operations
- **Intent classification**: Categorizes query type (visualize, predict, explain)
- **Parameter extraction**: Extracts query parameters

### Visualization Selector (`scripts/visualization_selector.py`)

Selects appropriate chart type:

- **Heuristic-first**: Rule-based selection for common patterns
- **LLM fallback**: Uses LLM when heuristics are uncertain
- **Chart types**: Scatter, line, bar, histogram, box, heatmap, etc.

## Storage Layer

The storage layer abstracts data persistence across multiple backends.

### Storage Backends

#### Local Backend (`storage/local_backend.py`)

Default filesystem-based storage:

- **User data**: Stored in `local_data/users/<user_id>/`
- **Models**: Stored in `local_data/models/<run_id>/`
- **Outputs**: Stored in `output_files/`
- **Tier enforcement**: Checks quotas before operations

```python
class LocalBackend(StorageBackend):
    def save_data(self, user_id: str, file_name: str, data: pd.DataFrame) -> Path:
        """Save dataframe to local filesystem"""
        path = Path("local_data") / "users" / user_id / file_name
        data.to_csv(path, index=False)
        return path
    
    def load_data(self, user_id: str, file_name: str) -> pd.DataFrame:
        """Load dataframe from local filesystem"""
        path = Path("local_data") / "users" / user_id / file_name
        return pd.read_csv(path)
```

#### S3 Backend (`storage/s3_backend.py`)

Cloud storage for production deployments:

- **Bucket structure**: `s3://bucket/users/<user_id>/`
- **Temporary caching**: Downloads to temp dir for processing
- **Presigned URLs**: For secure file access

### Session Database (`storage/session_db.py`)

Tracks workflow execution history:

```python
def log_run_metadata(
    run_id: str,
    success: bool,
    needs_role_review: bool,
    file_name: str,
    model_type: str,
    model_path: str,
    metadata_path: str,
    output_path: str,
) -> None:
    """Log workflow metadata to session database"""
    # Stores in SQLite or DynamoDB
    ...

def list_sessions(user_id: str, limit: int = 20) -> List[Dict]:
    """List recent workflow sessions"""
    # Returns list of session metadata
    ...
```

## Output Module

The output module formats and serializes analysis results.

### Output Formatter (`output/output_formatter.py`)

Converts predictions and analysis into serializable structures:

```python
def format_output(predictions: np.ndarray) -> List[Dict[str, Any]]:
    """Format predictions for API response"""
    return [{"prediction": float(pred)} for pred in predictions]

def format_analysis(analysis: str) -> Dict[str, Any]:
    """Format LLM analysis into structured output"""
    return {
        "summary": analysis,
        "timestamp": datetime.now().isoformat(),
    }
```

## Workflow Paths

### Complete Analysis Workflow

#### Stage 1: Data Ingestion and Preprocessing

1. **User uploads dataset** via Streamlit UI or API endpoint
2. **Storage backend** (`get_backend()`) retrieves file from local filesystem or S3
3. **Tier validation** checks user quotas (`MAX_REQUESTS_FREE`, `MAX_GB_FREE`)
4. **DataPreprocessor.run** loads and validates the dataset:
   - Checks file size and format
   - Loads CSV into pandas DataFrame
   - Applies `clean()` method: removes outliers, normalizes text, converts dates
   - Guesses target column if not provided
5. **Metadata extraction** via `parse_metadata()` and `infer_column_meta()`:
   - Extracts column types, statistics, sample values
   - Infers semantic roles (categorical, numeric, date, zip_code, fips_code, etc.)
   - Creates `ColumnMeta` objects for each column

#### Stage 2: Semantic Enrichment

1. **SemanticEnricher.enrich** searches for relevant public datasets:
   - Queries semantic index with user dataset metadata
   - Scores candidate datasets based on column role overlap
2. **rank_and_merge** selects and merges best dataset:
   - Synthesizes join keys (direct match, zip→fips, city+state hash)
   - Performs left join to enrich user data
   - Returns enriched DataFrame and merge report
3. **Dataset tagging** via `categorize_dataset()` for catalog indexing

#### Stage 3: Analysis Selection and Execution

1. **Modelability assessment** via `assess_modelability()`:
   - Checks minimum row count, target variance, feature availability
   - Returns `is_modelable` flag and reason
2. **Analyzer selection** via `select_analyzer()`:
   - Scores each analyzer in REGISTRY based on dataset characteristics
   - Returns best-fit analyzer (Regression, Classification, Cluster, Anomaly, or Descriptive)
3. **Model training and evaluation**:
   - Analyzer calls `select_best_model()` to try multiple algorithms
   - Evaluates models using appropriate metrics (R², MAPE, accuracy, F1, silhouette, etc.)
   - Applies quality thresholds (`MIN_R2`, `MAX_MAPE`)
4. **Iterative refinement** (if needed):
   - If initial results are poor and user labels are available, rerun with corrected metadata
   - If still poor, fall back to baseline model or descriptive analysis
5. **Feature importance** via SHAP values for explainability

#### Stage 4: Output Generation

1. **Model persistence** via `save_model()`:
   - Saves trained model to `local_data/models/<run_id>/best_model`
   - Enables reliable reruns without retraining
2. **Prediction export**:
   - Writes predictions to `output_files/<run_id>_predictions.csv`
3. **LLM summarization**:
   - Generates plain-English summary via `generate_summary()`
   - Creates business recommendations via `ask_follow_up_question()`
4. **Metadata logging** via `log_run_metadata()`:
   - Records run details to session database
   - Tracks success, model type, file paths, timestamps
5. **Dashboard assembly** via `orchestrate_dashboard()`:
   - Selects appropriate visualizations
   - Formats insights and confidence scores

#### Stage 5: Agent Actions

1. **AgentTrigger.trigger** executes automated actions:
   - Logs results to action database
   - Sends notifications if configured
2. **Agent recipes** execute based on results:
   - **Role review**: Triggers human review for uncertain column roles
   - **Anomaly alert**: Sends alerts for detected anomalies
   - **Reorder**: Triggers procurement workflows for inventory predictions

### Rerun Workflow

1. **User requests rerun** via `/rerun/{run_id}` endpoint
2. **Load saved model** from `local_data/models/<run_id>/best_model`
3. **Load original data** and metadata from session database
4. **Skip training**: Use saved model for predictions
5. **Generate new outputs** with updated data or parameters

## System Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────────┐
│                         Entry Points                             │
├─────────────────────────────────────────────────────────────────┤
│  FastAPI (main.py)           Streamlit UI (ui/dashboard.py)     │
│  - GET /sessions             - File upload                       │
│  - POST /rerun/{run_id}      - Session history                  │
│  - GET /health               - Chatbot interface                │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────────┐
│              orchestrate_workflow (orchestration/)               │
│                    WorkflowManager                               │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        v            v            v
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Stage 1:     │ │ Stage 2:     │ │ Stage 3:     │
│ Preprocess   │→│ Enrich       │→│ Analyze      │
├──────────────┤ ├──────────────┤ ├──────────────┤
│DataPrepro-   │ │Semantic-     │ │Analyzer-     │
│cessor        │ │Enricher      │ │Selector      │
│              │ │              │ │              │
│- Load data   │ │- Infer roles │ │- Assess      │
│- Clean       │ │- Search index│ │  modelability│
│- Validate    │ │- rank_and_   │ │- Select      │
│- Extract     │ │  merge       │ │  analyzer    │
│  metadata    │ │- Categorize  │ │- Train model │
└──────────────┘ └──────────────┘ │- Evaluate    │
                                   │- Refine      │
                                   └──────┬───────┘
                                          │
                     ┌────────────────────┼────────────────────┐
                     │                    │                    │
                     v                    v                    v
              ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
              │ Stage 4:     │     │ Stage 5:     │     │ Storage      │
              │ Generate     │     │ Agent        │     │ Layer        │
              │ Outputs      │     │ Actions      │     ├──────────────┤
              ├──────────────┤     ├──────────────┤     │- Local       │
              │Output-       │     │AgentTrigger  │     │  Backend     │
              │Generator     │     │              │     │- S3 Backend  │
              │              │     │- execute_    │     │- Session DB  │
              │- Save model  │     │  actions     │     │- Semantic    │
              │- Export      │     │- Role review │     │  Index       │
              │  predictions │     │- Anomaly     │     └──────────────┘
              │- LLM summary │     │  alert       │
              │- Log metadata│     │- Reorder     │
              │- Dashboard   │     └──────────────┘
              └──────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Supporting Layers                           │
├─────────────────────────────────────────────────────────────────┤
│ Preprocessing: metadata_parser, data_cleaning, llm_preprocessor │
│ Modeling: RegressionAnalyzer, ClassificationAnalyzer, Cluster,  │
│           AnomalyAnalyzer, DescriptiveAnalyzer                   │
│ Integration: semantic_merge, datalake_ingestion, semantic_index │
│ Agents: action_agent, agent_recipes (role_review, anomaly_alert)│
│ UI: chatbot, orchestrator, visualization_selector               │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Patterns

### Strategy Pattern (Analyzers)

The `Analyzer` abstract base class defines a standard interface for any analysis or modeling approach:

- **Polymorphism**: All analyzers implement `suitability_score()` and `run()`
- **Registry pattern**: Analyzers registered in `REGISTRY` for dynamic selection
- **Scoring mechanism**: Each analyzer scores its fit for a dataset
- **Extensibility**: New analyzers can be added without modifying core logic

### Composition Pattern (Orchestration)

The orchestration layer uses composition over inheritance:

- **WorkflowManager**: Coordinates helper classes (DataPreprocessor, SemanticEnricher, etc.)
- **Single responsibility**: Each helper class has one clear purpose
- **Loose coupling**: Helpers communicate via well-defined interfaces
- **Testability**: Each component can be tested independently

### Plugin Pattern (Agent Recipes)

Agents use a plugin architecture for extensibility:

- **Simple contract**: All recipes implement `execute()` method
- **Dynamic loading**: Recipes discovered and loaded at runtime
- **Configuration-driven**: Recipes enabled/disabled via config
- **Custom automation**: Users can add custom recipes without modifying core code

### Backend Abstraction (Storage)

Storage layer abstracts persistence details:

- **Abstract base class**: `StorageBackend` defines interface
- **Multiple implementations**: LocalBackend, S3Backend, etc.
- **Factory pattern**: `get_backend()` returns appropriate backend
- **Transparent switching**: Application code unaware of storage details

## Configuration and Feature Flags

### Key Configuration Values

**Quality Thresholds:**

- `MIN_R2 = 0.5`: Minimum R² for regression models
- `MAX_MAPE = 0.3`: Maximum MAPE for regression models

**Scalability Limits:**

- `MAX_ROWS_FIRST_PASS = 25000`: Rows for initial assessment
- `MAX_FEATURES_FIRST_PASS = 100`: Features for initial assessment
- `MAX_ROWS_FULL = 5000`: Rows for full analysis
- `PROFILE_SAMPLE_ROWS = 1000`: Rows for profiling large datasets

**Quota System:**

- `MAX_REQUESTS_FREE = 20`: Free tier request limit
- `MAX_GB_FREE = 1`: Free tier storage limit (GB)

**Feature Flags:**

- `ENABLE_LLM_PREPROCESSING`: Use LLM for advanced preprocessing
- `ENABLE_SHAP_EXPLANATIONS`: Generate SHAP explanations
- `ENABLE_SEMANTIC_MERGE`: Enable automatic dataset enrichment
- `ENABLE_AGENT_ACTIONS`: Enable automated agent actions

### File Paths

**Data Storage:**

- User data: `local_data/users/<user_id>/`
- Models: `local_data/models/<run_id>/best_model`
- Outputs: `output_files/<run_id>_predictions.csv`
- Datasets: `datasets/`

**Databases:**

- Semantic index: `local_data/_INDEX_DB`
- Session history: `local_data/sessions.db`
- Action logs: `local_data/action_logs.db`

## Testing Strategy

**Unit Tests:**

- Individual analyzer implementations
- Preprocessing utilities
- Storage backend operations

**Integration Tests:**

- End-to-end workflow execution
- Semantic merge functionality
- Model training and evaluation

**Test Data:**

- Sample datasets in `datasets/` directory
- Synthetic data generation for edge cases
- Real-world datasets for validation

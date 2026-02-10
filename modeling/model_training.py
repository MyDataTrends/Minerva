from config.feature_flags import ENABLE_HEAVY_EXPLANATIONS, ENABLE_SHAP_EXPLANATIONS
from preprocessing.llm_preprocessor import preprocess_data_with_llm
from preprocessing.llm_analyzer import analyze_dataset, score_dataset_similarity
from preprocessing.data_cleaning import (
    clean_missing_values,
    normalize_text_columns,
    convert_to_datetime,
    remove_outliers,
    encode_categorical_columns,
    fuzzy_match_columns,
)
from modeling.model_selector import select_best_model
import pandas as pd


def get_model_explanations(model, X):
    """Return basic interpretability information for a trained model."""
    explanations: dict[str, list] = {}
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        try:
            explanations["feature_importances"] = fi.tolist()
        except Exception:  # pragma: no cover - numeric conversion fallback
            explanations["feature_importances"] = list(fi)
    if hasattr(model, "coef_"):
        coef = model.coef_
        try:
            explanations["coefficients"] = coef.tolist()
        except Exception:  # pragma: no cover
            explanations["coefficients"] = list(coef)

    if not ENABLE_HEAVY_EXPLANATIONS:
        explanations["explanations_disabled"] = True
        return explanations

    # SHAP explanations - controlled by separate flag for granular control
    if not ENABLE_SHAP_EXPLANATIONS:
        return explanations

    try:  # optional dependency
        import shap  # type: ignore
    except Exception:  # pragma: no cover - SHAP may be unavailable
        explanations["shap_unavailable"] = True
        return explanations

    if hasattr(model, "feature_importances_"):
        try:  # tree-based models
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)
            try:
                explanations["shap_values"] = shap_vals.tolist()
            except Exception:  # pragma: no cover - when shap returns list
                explanations["shap_values"] = [sv.tolist() for sv in shap_vals]
            # Add mean absolute SHAP values for feature importance ranking
            import numpy as np
            if isinstance(shap_vals, list):
                # Multi-class case
                mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
            else:
                mean_shap = np.abs(shap_vals).mean(axis=0)
            explanations["shap_importance"] = mean_shap.tolist()
        except Exception:  # pragma: no cover - shap may fail
            pass

    return explanations


def train_model(X, y, datalake_dfs):
    """Train a model with resilient preprocessing - tries multiple strategies on failure."""
    import numpy as np
    import logging
    
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame({"text_column": list(X)})
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    # Resilient analysis - don't fail if LLM unavailable
    try:
        analysis = analyze_dataset(X)
        print("Dataset Analysis and Recommendations:", analysis.get("summary", "N/A"))
    except Exception as e:
        logging.warning(f"Dataset analysis failed: {e}")
        analysis = {"summary": "Analysis unavailable"}

    try:
        similarity_scores = score_dataset_similarity(X, datalake_dfs)
        print("Similarity Scores with Datalake:", similarity_scores)
    except Exception as e:
        logging.warning(f"Similarity scoring failed: {e}")

    # Resilient preprocessing with multiple fallback strategies
    X_processed = X.copy()
    
    # Strategy 1: Try LLM preprocessing
    try:
        X_processed = preprocess_data_with_llm(X_processed)
    except Exception as e:
        logging.warning(f"LLM preprocessing failed: {e}")
    
    # Strategy 2: Clean missing values
    try:
        X_processed = clean_missing_values(X_processed, strategy="fill", fill_value=0)
    except Exception as e:
        logging.warning(f"clean_missing_values failed: {e}")
        # Fallback: simple fillna
        for col in X_processed.columns:
            try:
                if X_processed[col].dtype in ['int64', 'float64']:
                    X_processed[col] = X_processed[col].fillna(0)
                else:
                    X_processed[col] = X_processed[col].fillna('')
            except:
                pass
    
    # Ensure all columns are numeric for modeling
    cols_to_drop = []
    for col in X_processed.columns:
        # Handle datetime columns
        if X_processed[col].dtype == 'datetime64[ns]':
            try:
                X_processed[col] = X_processed[col].astype('int64') // 10**9
            except:
                cols_to_drop.append(col)
        # Handle object columns
        elif X_processed[col].dtype == 'object':
            try:
                # Try numeric conversion first
                X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
                X_processed[col] = X_processed[col].fillna(0)
            except:
                try:
                    # Try label encoding
                    X_processed[col] = pd.factorize(X_processed[col])[0]
                except:
                    cols_to_drop.append(col)
        # Handle any remaining non-numeric types
        elif not np.issubdtype(X_processed[col].dtype, np.number):
            try:
                X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce').fillna(0)
            except:
                cols_to_drop.append(col)
    
    # Drop columns that couldn't be converted
    if cols_to_drop:
        logging.warning(f"Dropping non-numeric columns: {cols_to_drop}")
        X_processed = X_processed.drop(columns=cols_to_drop, errors='ignore')
    
    # Final cleanup - ensure no NaN/inf values
    X_processed = X_processed.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Ensure we have at least one feature
    if X_processed.shape[1] == 0:
        raise ValueError("No valid features remaining after preprocessing")
    
    # Align y with X_processed
    y_aligned = y.loc[X_processed.index] if hasattr(y, 'loc') else y[:len(X_processed)]
    
    # Try multiple model training strategies
    model = None
    training_errors = []
    
    # Strategy 1: Best model selection
    try:
        model = select_best_model(X_processed, y_aligned)
        model.fit(X_processed, y_aligned)
    except Exception as e:
        training_errors.append(f"select_best_model: {e}")
        model = None
    
    # Strategy 2: Fallback to simple models
    if model is None:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
        
        is_classification = len(y_aligned.unique()) <= 20
        
        fallback_models = [
            RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42) if is_classification 
            else RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
            LogisticRegression(max_iter=1000, random_state=42) if is_classification 
            else Ridge(random_state=42),
        ]
        
        for fallback in fallback_models:
            try:
                fallback.fit(X_processed, y_aligned)
                model = fallback
                logging.info(f"Fallback model succeeded: {type(fallback).__name__}")
                break
            except Exception as e:
                training_errors.append(f"{type(fallback).__name__}: {e}")
    
    if model is None:
        raise ValueError(f"All model training strategies failed: {training_errors}")

    class TextModel:
        def __init__(self, model):
            self.model = model
            self.training_warnings = training_errors if training_errors else None

        def _transform(self, texts):
            if not isinstance(texts, list):
                texts = [texts]
            df = pd.DataFrame({"text_column": texts})
            try:
                df["text_column"] = (
                    df["text_column"].astype(str).str.extract(r"(\d+)").astype(float).fillna(0).astype(int)
                )
            except:
                df["text_column"] = 0
            return df

        def predict(self, X):
            # Pass-through for DataFrame inputs (standard pipeline usage)
            if isinstance(X, pd.DataFrame):
                return self.model.predict(X)
            # Otherwise, support simple text inputs for convenience
            df = self._transform(X)
            return self.model.predict(df)

    text_model = TextModel(model)
    
    # Get explanations with error handling
    try:
        text_model.explanations = get_model_explanations(model, X_processed)
    except Exception as e:
        logging.warning(f"Model explanations failed: {e}")
        text_model.explanations = {"error": str(e)}
    
    return text_model


from pathlib import Path
from utils.security import secure_join


def save_model(model, filename: str = "model.pkl", run_id: str | None = None) -> Path:
    """Save model with checksum for integrity verification."""
    from utils.safe_pickle import safe_dump
    
    base = Path.cwd() / "models"
    if run_id is not None:
        base = base / run_id
    base.mkdir(parents=True, exist_ok=True)
    path = secure_join(base, filename)
    
    return safe_dump(
        model,
        path,
        add_checksum=True,
        metadata={"run_id": run_id, "filename": filename},
    )


def load_model(filename: str = "model.pkl", run_id: str | None = None, verify: bool = True):
    """Load model with optional checksum verification."""
    from utils.safe_pickle import safe_load
    
    base = Path.cwd() / "models"
    if run_id is not None:
        base = base / run_id
    path = secure_join(base, filename)
    
    return safe_load(
        path,
        verify=verify,
        allow_missing_checksum=True,  # For backwards compatibility
    )


import pickle

from config.feature_flags import ENABLE_HEAVY_EXPLANATIONS
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

    try:  # optional dependency
        import shap  # type: ignore
    except Exception:  # pragma: no cover - SHAP may be unavailable
        return explanations

    if hasattr(model, "feature_importances_"):
        try:  # tree-based models
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)
            try:
                explanations["shap_values"] = shap_vals.tolist()
            except Exception:  # pragma: no cover - when shap returns list
                explanations["shap_values"] = [sv.tolist() for sv in shap_vals]
        except Exception:  # pragma: no cover - shap may fail
            pass

    return explanations


def train_model(X, y, datalake_dfs):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame({"text_column": list(X)})
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    analysis = analyze_dataset(X)
    print("Dataset Analysis and Recommendations:", analysis["summary"])

    similarity_scores = score_dataset_similarity(X, datalake_dfs)
    print("Similarity Scores with Datalake:", similarity_scores)


    X_processed = preprocess_data_with_llm(X)
    X_processed = clean_missing_values(X_processed, strategy="fill", fill_value=0)
    if "text_column" in X_processed.columns:
        X_processed = normalize_text_columns(X_processed, column_name="text_column")
        if X_processed["text_column"].dtype == object:
            X_processed["text_column"] = (
                X_processed["text_column"].str.extract(r"(\d+)").astype(int)
            )

    model = select_best_model(X_processed, y)
    model.fit(X_processed, y)
    initial_score = model.score(X_processed, y)

    if initial_score < 0.8:
        if "date_column" in X_processed.columns:
            X_processed = convert_to_datetime(X_processed, column_name="date_column")
        if "numeric_column" in X_processed.columns:
            X_processed = remove_outliers(X_processed, column_name="numeric_column")
        if "category_column" in X_processed.columns:
            X_processed = encode_categorical_columns(X_processed, column_name="category_column")
        if "product_name" in X_processed.columns:
            X_processed = fuzzy_match_columns(X_processed, column_name="product_name")

        model = select_best_model(X_processed, y)
        model.fit(X_processed, y)

    class TextModel:
        def __init__(self, model):
            self.model = model

        def _transform(self, texts):
            if not isinstance(texts, list):
                texts = [texts]
            df = pd.DataFrame({"text_column": texts})
            df["text_column"] = (
                df["text_column"].astype(str).str.extract(r"(\d+)").astype(int)
            )
            return df

        def predict(self, X):
            # Pass-through for DataFrame inputs (standard pipeline usage)
            if isinstance(X, pd.DataFrame):
                return self.model.predict(X)
            # Otherwise, support simple text inputs for convenience
            df = self._transform(X)
            return self.model.predict(df)

    text_model = TextModel(model)
    text_model.explanations = get_model_explanations(model, X_processed)
    return text_model


from pathlib import Path
from utils.security import secure_join


def save_model(model, filename: str = "model.pkl", run_id: str | None = None) -> None:
    base = Path.cwd() / "models"
    if run_id is not None:
        base = base / run_id
    base.mkdir(parents=True, exist_ok=True)
    path = secure_join(base, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(filename: str = "model.pkl", run_id: str | None = None):
    base = Path.cwd() / "models"
    if run_id is not None:
        base = base / run_id
    path = secure_join(base, filename)
    with open(path, "rb") as f:
        return pickle.load(f)


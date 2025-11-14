import os
N_JOBS = int(os.getenv("ML_N_JOBS", "-1"))
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

from config.feature_flags import ALLOW_FULL_COMPARE_MODELS
from config.model_allowlist import MODEL_ALLOWLIST


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def accuracy_within_tolerance(y_true, y_pred, tolerance=0.1):
    """Percentage of predictions within tolerance"""
    within_tolerance = np.abs((y_true - y_pred) / y_true) <= tolerance
    return np.mean(within_tolerance) * 100


def cost_weighted_error(y_true, y_pred, cost_function):
    """Weighted error based on a custom cost function"""
    error = y_true - y_pred
    return np.sum(cost_function(error))


def custom_mae(y_true, y_pred):
    """Custom Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def custom_rmse(y_true, y_pred):
    """Custom Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def initialize_metrics(add_metric):
    if add_metric is None:
        return
    add_metric(
        "Custom MAE",
        "Custom Mean Absolute Error",
        custom_mae,
        greater_is_better=False,
    )
    add_metric(
        "Custom RMSE",
        "Custom Root Mean Squared Error",
        custom_rmse,
        greater_is_better=False,
    )
    add_metric(
        "MAPE",
        "Mean Absolute Percentage Error",
        mape,
        greater_is_better=False,
    )
    add_metric(
        "Accuracy Within Tolerance",
        "Percentage of predictions within tolerance",
        accuracy_within_tolerance,
        greater_is_better=True,
    )
    add_metric(
        "Cost Weighted Error",
        "Weighted error based on a custom cost function",
        cost_weighted_error,
        greater_is_better=False,
    )


def _quick_model_search(X, y, task: str):
    """Return a model chosen via a small CV sweep over a sampled subset."""

    subset = X.sample(n=min(len(X), 200), random_state=0)
    y_subset = y.loc[subset.index]
    if len(subset) < 2:
        # Not enough samples for cross-validation, return a default model
        if task == "classification":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000)
        else:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        model.fit(X, y)
        return model
    cv = min(3, len(subset))
    cv = max(cv, 2)

    candidates = []
    allowlist = MODEL_ALLOWLIST.get(task, [])
    if task == "classification":
        from sklearn.linear_model import LogisticRegression

        if "lightgbm" in allowlist:
            try:
                from lightgbm import LGBMClassifier
                candidates.append(LGBMClassifier(n_estimators=200, n_jobs=N_JOBS, random_state=0))
            except Exception:  # pragma: no cover - optional dependency
                pass
        if "xgboost" in allowlist:
            try:
                from xgboost import XGBClassifier
                candidates.append(
                    XGBClassifier(
                        use_label_encoder=False,
                        eval_metric="logloss",
                        n_estimators=200,
                        tree_method="hist",
                        n_jobs=N_JOBS,
                        random_state=0,
                    )
                )
            except Exception:  # pragma: no cover
                pass
        if "lr" in allowlist:
            candidates.append(LogisticRegression(max_iter=1000))
        scoring = "accuracy"
    else:
        from sklearn.linear_model import Ridge

        if "lightgbm" in allowlist:
            try:
                from lightgbm import LGBMRegressor
                candidates.append(LGBMRegressor(n_estimators=200, n_jobs=N_JOBS, random_state=0))
            except Exception:  # pragma: no cover - optional dependency
                pass
        if "xgboost" in allowlist:
            try:
                from xgboost import XGBRegressor
                candidates.append(
                    XGBRegressor(
                        n_estimators=200,
                        tree_method="hist",
                        n_jobs=N_JOBS,
                        random_state=0,
                    )
                )
            except Exception:  # pragma: no cover
                pass
        if "ridge" in allowlist:
            candidates.append(Ridge())
        scoring = "r2"

    best_model = None
    best_score = -np.inf
    for model in candidates:
        try:
            score = cross_val_score(model, subset, y_subset, cv=cv, scoring=scoring, n_jobs=N_JOBS).mean()
        except Exception:  # pragma: no cover - model may fail
            continue
        if score > best_score:
            best_score = score
            best_model = model

    if best_model is None:
        from sklearn.linear_model import LinearRegression

        best_model = LinearRegression()

    best_model.fit(X, y)
    return best_model


def select_best_model(X, y, task: str = "regression"):
    if ALLOW_FULL_COMPARE_MODELS:
        try:
            if task == "classification":
                from pycaret.classification import setup, compare_models, add_metric
            else:
                from pycaret.regression import setup, compare_models, add_metric
        except Exception:  # pragma: no cover - optional dependency
            setup = compare_models = add_metric = None
    else:  # pragma: no cover - feature flag disabled
        setup = compare_models = add_metric = None

    if setup is not None and compare_models is not None:
        setup(
            data=pd.concat([X, y], axis=1),
            target=y.name,
            silent=True,
            verbose=True,
        )
        if task != "classification":
            initialize_metrics(add_metric)
            best_model = compare_models(
                include=MODEL_ALLOWLIST.get(task, []),
                sort="Custom RMSE",
                tune=True,
                verbose=True,
            )
        else:
            best_model = compare_models(
                include=MODEL_ALLOWLIST.get(task, []),
                verbose=True,
            )
        return best_model

    return _quick_model_search(X, y, task)


def evaluate_model(model, X_val, y_val):
    """Return evaluation metrics for ``model`` on validation data."""
    preds = model.predict(X_val)
    r2 = r2_score(y_val, preds)
    mape_val = mape(y_val, preds)
    return {"r2": float(r2), "mape": float(mape_val)}


def run_regression(X: pd.DataFrame, y: pd.Series) -> dict:
    """Train a simple regression model and return predictions and metrics."""

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error

    model = RandomForestRegressor(random_state=0, n_estimators=200, n_jobs=N_JOBS)
    model.fit(X, y)
    preds = model.predict(X)
    metrics = evaluate_model(model, X, y)
    metrics["mae"] = mean_absolute_error(y, preds)
    raw = {"model": model, "predictions": preds, "metrics": metrics}
    return {"results": raw, "model_name": model.__class__.__name__}

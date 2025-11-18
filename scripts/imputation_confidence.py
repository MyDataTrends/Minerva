import pandas as pd
import numpy as np
from typing import Optional, Dict

from sklearn.neighbors import NearestNeighbors


def _score_stat_impute(value: float, mean: float, std: float) -> float:
    if pd.isna(value):
        return 0.0
    if pd.isna(std) or std == 0:
        return 1.0
    z = abs(value - mean) / (std * 3)
    return float(max(0.0, 1 - min(z, 1)))


def _score_model_impute(value: float, original: float, std: float) -> float:
    if pd.isna(value):
        return 0.0
    if pd.isna(original) or pd.isna(std) or std == 0:
        # fallback when residual can't be computed
        return 0.8
    residual = abs(value - original)
    z = residual / (std * 3)
    return float(max(0.0, 1 - min(z, 1)))


def _knn_row_scores(df: pd.DataFrame, k: int = 5) -> np.ndarray:
    numeric_cols = df.select_dtypes(include=np.number)
    if numeric_cols.empty:
        return np.full(len(df), 0.5)

    features = numeric_cols.fillna(numeric_cols.mean())
    n_neighbors = min(k, len(features))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(features)
    distances, _ = nbrs.kneighbors(features)
    avg_dist = distances.mean(axis=1)
    max_dist = avg_dist.max() or 1.0
    scores = 1 - (avg_dist / max_dist)
    return scores


def score_imputations(
    df: pd.DataFrame,
    imputed_mask: pd.DataFrame,
    method_map: Dict[str, str],
    df_original: Optional[pd.DataFrame] = None,
) -> Dict:
    """Score confidence of imputed values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after imputation.
    imputed_mask : pd.DataFrame
        Boolean DataFrame indicating which cells were imputed.
    method_map : dict[str, str]
        Mapping of column names to the imputation method used.
    df_original : pd.DataFrame, optional
        Original DataFrame before imputation.

    Returns
    -------
    dict
        Dictionary with confidence scores, summary and flagged cells.
    """
    if df.shape != imputed_mask.shape:
        raise ValueError("imputed_mask must have the same shape as df")

    confidence_scores = pd.DataFrame(np.nan, index=df.index, columns=df.columns)

    knn_scores = None
    if any(method_map.get(col) == "knn" for col in df.columns):
        knn_scores = _knn_row_scores(df)

    for col in df.columns:
        method = method_map.get(col, "mean")
        imputed_indices = df.index[imputed_mask[col]]
        if imputed_indices.empty:
            continue
        if method in {"mean", "median", "mode"}:
            base_series = df_original[col] if df_original is not None else df[col]
            mean = base_series.mean()
            std = base_series.std()
            for idx in imputed_indices:
                val = df.at[idx, col]
                confidence_scores.at[idx, col] = _score_stat_impute(val, mean, std)
        elif method == "model":
            std = (df_original[col].std() if df_original is not None else df[col].std())
            for idx in imputed_indices:
                val = df.at[idx, col]
                orig = df_original.at[idx, col] if df_original is not None else np.nan
                confidence_scores.at[idx, col] = _score_model_impute(val, orig, std)
        elif method == "knn":
            if knn_scores is None:
                knn_scores = _knn_row_scores(df)
            for idx in imputed_indices:
                row_pos = df.index.get_loc(idx)
                confidence_scores.at[idx, col] = knn_scores[row_pos]
        else:
            # default conservative score
            for idx in imputed_indices:
                confidence_scores.at[idx, col] = 0.5

    flagged_mask = imputed_mask & (confidence_scores < 0.3)
    flagged = flagged_mask.stack().loc[lambda s: s].index.tolist()

    overall_mean = float(confidence_scores.stack().mean()) if not confidence_scores.stack().empty else np.nan
    per_column = confidence_scores.mean().to_dict()
    per_method = {}
    for col, method in method_map.items():
        col_scores = confidence_scores[col]
        if method not in per_method:
            per_method[method] = []
        per_method[method].extend(col_scores.dropna().tolist())
    per_method_stats = {m: np.mean(v) if v else np.nan for m, v in per_method.items()}

    summary = {
        "overall_mean": overall_mean,
        "per_column_mean": per_column,
        "per_method_mean": per_method_stats,
    }

    return {
        "confidence_scores": confidence_scores,
        "summary": summary,
        "flagged_low_confidence": flagged,
    }


if __name__ == "__main__":
    # Demonstration of scoring on a small example
    data = {
        "A": [1.0, 2.0, np.nan, 4.0, 5.0],
        "B": [10.0, np.nan, 30.0, 40.0, 50.0],
        "C": [100.0, 200.0, 300.0, np.nan, 500.0],
        "D": [7.0, 8.0, np.nan, 10.0, 11.0],
    }
    df_original = pd.DataFrame(data)

    # Mean imputation for column A
    df = df_original.copy()
    mean_A = df["A"].mean()
    df["A"].fillna(mean_A, inplace=True)
    mask_A = df_original["A"].isna()

    # Median imputation for column B
    median_B = df["B"].median()
    df["B"].fillna(median_B, inplace=True)
    mask_B = df_original["B"].isna()

    # KNN imputation for column C
    # simple fill of NaNs with mean for demonstration
    knn_val = df["C"].mean()
    df["C"].fillna(knn_val, inplace=True)
    mask_C = df_original["C"].isna()

    # Model-based imputation for column D - here we just use mean
    model_val = df["D"].mean()
    df["D"].fillna(model_val, inplace=True)
    mask_D = df_original["D"].isna()

    imputed_mask = pd.DataFrame({
        "A": mask_A,
        "B": mask_B,
        "C": mask_C,
        "D": mask_D,
    })

    method_map = {
        "A": "mean",
        "B": "median",
        "C": "knn",
        "D": "model",
    }

    result = score_imputations(df, imputed_mask, method_map, df_original=df_original)
    print("Confidence Scores:\n", result["confidence_scores"])
    print("Summary:\n", result["summary"])
    print("Flagged Low Confidence:\n", result["flagged_low_confidence"])

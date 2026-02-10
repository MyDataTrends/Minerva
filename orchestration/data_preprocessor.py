from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sklearn.metrics import mean_absolute_error

from utils.metrics import REQUESTS
from utils.user_profile import get_user_tier
from utils.usage_tracker import increment_request, usage_info
from storage.get_backend import backend
from storage.local_backend import parse_file, load_run_metadata
from preprocessing.data_cleaning import (
    clean_missing_values,
    normalize_text_columns,
    remove_duplicates,
    convert_to_datetime,
    remove_outliers,
    encode_categorical_columns,
)
from preprocessing.metadata_parser import parse_metadata
from preprocessing.misaligned_row_detector import detect_misaligned_rows
from preprocessing.advanced_schema_validator import validate_schema
from preprocessing.context_missing_finder import find_contextual_missingness
from scripts.imputation_confidence import score_imputations
from scripts.alignment_drift_monitor import generate_historical_stats, monitor_alignment_drift


class DataPreprocessor:
    """Handle basic data cleaning and validation."""

    def clean(
        self,
        data: pd.DataFrame,
        *,
        check_misalignment: bool = False,
        misalignment_schema: Optional[dict[str, type]] = None,
        check_context_missing: bool = False,
        score_imputations_flag: bool = False,
        monitor_drift: bool = False,
        baseline_stats: Optional[dict[str, Any]] = None,
        return_diagnostics: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
        """Clean ``data`` with optional diagnostic steps."""

        diagnostics: dict[str, Any] = {}

        if check_misalignment and misalignment_schema:
            diagnostics["misalignment"] = detect_misaligned_rows(
                data, misalignment_schema
            )

        if check_context_missing:
            diagnostics["context_missing"] = find_contextual_missingness(data)

        original_df = data.copy()
        imputed_mask = data.isna()
        preprocessing_errors = []

        # Resilient preprocessing - try each step, skip on failure
        metadata = parse_metadata(data)
        
        for column in metadata["columns"]:
            if column not in data.columns:
                continue
                
            col_dtype = metadata["dtypes"].get(column, "unknown")
            
            # Try text normalization and encoding for object columns
            if col_dtype == "object":
                try:
                    data = normalize_text_columns(data, column)
                except Exception as e:
                    preprocessing_errors.append(f"normalize_text({column}): {e}")
                
                try:
                    data = encode_categorical_columns(data, column)
                except Exception as e:
                    preprocessing_errors.append(f"encode_categorical({column}): {e}")
                    # Fallback: try label encoding or just drop
                    try:
                        data[column] = pd.factorize(data[column])[0]
                    except:
                        try:
                            data = data.drop(columns=[column])
                        except:
                            pass
            
            # Try datetime conversion
            elif "date" in column.lower() or col_dtype == "datetime64[ns]":
                try:
                    data = convert_to_datetime(data, column)
                except Exception as e:
                    preprocessing_errors.append(f"convert_datetime({column}): {e}")
                    # Fallback: convert to numeric timestamp or drop
                    try:
                        data[column] = pd.to_datetime(data[column], errors='coerce')
                        # Convert to numeric for modeling
                        if data[column].dtype == 'datetime64[ns]':
                            data[column] = data[column].astype('int64') // 10**9  # Unix timestamp
                    except:
                        try:
                            data = data.drop(columns=[column])
                        except:
                            pass
            
            # Try outlier removal for numeric columns
            elif col_dtype in ["int64", "float64"]:
                try:
                    data = remove_outliers(data, column)
                except Exception as e:
                    preprocessing_errors.append(f"remove_outliers({column}): {e}")
                    # Fallback: clip extreme values instead
                    try:
                        q1, q99 = data[column].quantile([0.01, 0.99])
                        data[column] = data[column].clip(q1, q99)
                    except:
                        pass
        
        # Convert any remaining datetime columns to numeric
        for col in data.columns:
            if data[col].dtype == 'datetime64[ns]':
                try:
                    data[col] = data[col].astype('int64') // 10**9
                except:
                    try:
                        data = data.drop(columns=[col])
                    except:
                        pass
            # Handle Timestamp objects
            elif data[col].dtype == 'object':
                try:
                    # Check if it's actually timestamps stored as objects
                    sample = data[col].dropna().iloc[0] if len(data[col].dropna()) > 0 else None
                    if sample is not None and hasattr(sample, 'timestamp'):
                        data[col] = pd.to_datetime(data[col], errors='coerce').astype('int64') // 10**9
                except:
                    pass

        # Clean missing values with fallback strategies
        try:
            data = clean_missing_values(data, strategy="fill", fill_value=0)
        except Exception as e:
            preprocessing_errors.append(f"clean_missing_values: {e}")
            # Fallback: simple fillna
            try:
                for col in data.columns:
                    if data[col].dtype in ['int64', 'float64']:
                        data[col] = data[col].fillna(0)
                    else:
                        data[col] = data[col].fillna('')
            except:
                pass

        if score_imputations_flag:
            try:
                common_cols = [c for c in imputed_mask.columns if c in data.columns]
                common_rows = imputed_mask.index.intersection(data.index)
                if common_cols and len(common_rows) > 0:
                    aligned_mask = imputed_mask.loc[common_rows, common_cols].reindex(
                        index=data.index, columns=data.columns, fill_value=False
                    )
                    diagnostics["imputation_confidence"] = score_imputations(
                        data,
                        aligned_mask,
                        {c: "fill" for c in data.columns},
                        df_original=original_df,
                    )
                else:
                    diagnostics["imputation_confidence"] = {"skipped": "shape_mismatch"}
            except Exception as e:
                diagnostics["imputation_confidence"] = {"error": str(e)}

        # Remove duplicates with fallback
        try:
            data = remove_duplicates(data)
        except Exception as e:
            preprocessing_errors.append(f"remove_duplicates: {e}")
        
        # Log preprocessing issues but don't fail
        if preprocessing_errors:
            diagnostics["preprocessing_warnings"] = preprocessing_errors
            logging.warning(f"Preprocessing completed with {len(preprocessing_errors)} warnings")

        if monitor_drift:
            stats = baseline_stats or generate_historical_stats(data)
            diagnostics["alignment_drift"] = monitor_alignment_drift(data, stats)

        if return_diagnostics:
            return data, diagnostics
        return data

    def validate(
        self,
        data: pd.DataFrame,
        expected_schema: Optional[dict[str, str]] = None,
        valid_categories: Optional[dict[str, set[str]]] = None,
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("Input data is empty")
        if data.shape[0] < 5:
            raise ValueError("Input data must contain at least 5 rows")
        if not any(pd.api.types.is_numeric_dtype(data[col]) for col in data.columns):
            raise ValueError("Input data must contain at least one numeric column")

        if expected_schema:
            result = validate_schema(
                data,
                expected_schema,
                valid_categories=valid_categories,
                quarantine_invalid=False,
                coerce=False,
            )
            if result.get("mismatch_counts"):
                raise ValueError(f"Schema validation failed: {result['mismatch_counts']}")

    def guess_target_column(self, data: pd.DataFrame) -> str:
        non_unique = data.columns[data.nunique() != len(data)]
        mask = data[non_unique].apply(lambda c: pd.api.types.is_numeric_dtype(c))
        candidates = list(non_unique[mask])
        non_unnamed = [c for c in candidates if not c.startswith("Unnamed")]
        if non_unnamed:
            return data[non_unnamed].nunique().idxmax()
        raise ValueError(
            "No suitable target column found. Ensure the data contains numeric columns suitable for modeling."
        )

    def load_dataset(
        self,
        user_id: str,
        file_name: str,
        run_id: Optional[str] = None,
    ) -> tuple[pd.DataFrame, Path]:
        if run_id:
            stored = load_run_metadata(run_id)
            if stored and stored.get("dataset_path"):
                path = Path(stored["dataset_path"])
                return parse_file(path.read_bytes(), path.name), path
        path = backend.get_file(f"User_Data/{user_id}/{file_name}")
        return parse_file(path.read_bytes(), file_name), path

    def run(
        self,
        user_id: str,
        file_name: str,
        target_column: Optional[str],
        run_id: Optional[str],
    ) -> tuple[Optional[pd.DataFrame], Optional[str], Optional[dict], Path]:
        REQUESTS.inc()
        tier = get_user_tier(user_id)
        if tier == "free" and run_id:
            logging.warning("Free tier user attempted rerun: %s", user_id)
            return None, None, {"error": "Feature not available for free-tier users"}, Path()

        info = usage_info(user_id)
        from config import MAX_REQUESTS_FREE, MAX_GB_FREE
        if info.get("requests", 0) >= MAX_REQUESTS_FREE or info.get("bytes", 0) >= MAX_GB_FREE * 1024 * 1024 * 1024:
            increment_request(user_id, 0)
            logging.warning("User %s exceeded free tier limits", user_id)
            return None, None, {"error": "Free tier limit exceeded"}, Path()

        if run_id:
            stored = load_run_metadata(run_id)
            if not stored:
                logging.error("No metadata found for run_id %s", run_id)
                return None, None, {"error": "Invalid run_id"}, Path()
        data, path = self.load_dataset(user_id, file_name, run_id)
        file_bytes = path.stat().st_size if path else 0
        increment_request(user_id, file_bytes)
        if isinstance(data, dict) and "error" in data:
            return None, None, {"error": data["error"]}, path

        try:
            self.validate(data)
        except ValueError as exc:
            return None, None, {"error": str(exc)}, path

        if run_id:
            stored = load_run_metadata(run_id)
            if stored and stored.get("model_path"):
                from modeling.model_training import load_model
                from modeling.model_selector import evaluate_model
                from utils.safe_pickle import safe_load
                
                if target_column is None:
                    try:
                        target_column = self.guess_target_column(data)
                    except ValueError:
                        target_column = data.select_dtypes(include="number").columns[0]
                mp = Path(stored["model_path"]) if isinstance(stored["model_path"], str) else None
                model = None
                if mp and mp.exists():
                    # Use safe_load with checksum verification
                    try:
                        model = safe_load(mp, verify=True, allow_missing_checksum=True)
                    except Exception as e:
                        logging.warning(f"Safe load failed, using fallback: {e}")
                        model = load_model(mp.name, run_id)
                else:
                    # Fallback to standard loader using filename and run_id
                    try:
                        fname = mp.name if mp else str(stored["model_path"])  # type: ignore[arg-type]
                        model = load_model(fname, run_id)
                    except Exception:
                        model = load_model("best_model", run_id)
                preds = model.predict(data.drop(columns=[target_column]))
                mae_val = mean_absolute_error(data[target_column], preds)
                em = evaluate_model(model, data.drop(columns=[target_column]), data[target_column])
                em["mae"] = mae_val
                from output.output_formatter import format_output, format_analysis
                from preprocessing.llm_summarizer import generate_summary
                result = {
                    "output": format_output(preds),
                    "analysis": format_analysis("Re-run evaluation complete"),
                    "metrics": {"rerun": em},
                    "recommended_models": {},
                    "actions": [],
                    "needs_role_review": stored.get("needs_role_review", False),
                    "run_id": run_id,
                    "user_id": user_id,
                    "file_name": file_name,
                    "model_info": {
                        "model_type": type(model).__name__,
                        "merge_report": stored.get("merge_report"),
                    },
                }
                summary_output = generate_summary(
                    data_stats=result["metrics"],
                    model_results=result["model_info"],
                    prompt=(
                        "Given these data statistics and model outputs, write a one-page business summary, "
                        "list the top 3 next steps, and include a clear call to action."
                    ),
                )
                result.update(summary_output)
                return None, None, result, path

        if target_column is None:
            try:
                target_column = self.guess_target_column(data)
            except ValueError:
                target_column = data.select_dtypes(include="number").columns[0]
        return data, target_column, None, path

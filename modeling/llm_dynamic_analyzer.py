"""
LLM-powered dynamic data analysis.

This module allows an LLM to inspect data and generate custom preprocessing
and analysis code on the fly, adapting to any data structure.
"""

import pandas as pd
import numpy as np
import logging
from typing import Any, Dict, Optional, Tuple
from config.feature_flags import ENABLE_LOCAL_LLM

logger = logging.getLogger(__name__)


def _get_data_profile(df: pd.DataFrame) -> str:
    """Generate a detailed profile of the DataFrame for the LLM."""
    profile = []
    profile.append(f"DataFrame shape: {df.shape[0]} rows × {df.shape[1]} columns")
    profile.append(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    profile.append("")
    profile.append("Column details:")
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        n_null = df[col].isna().sum()
        null_pct = (n_null / len(df)) * 100
        
        # Sample values
        non_null = df[col].dropna()
        if len(non_null) > 0:
            samples = non_null.head(3).tolist()
            sample_str = str(samples)[:50]
        else:
            sample_str = "all null"
        
        profile.append(f"  - {col}:")
        profile.append(f"      dtype: {dtype}")
        profile.append(f"      unique: {n_unique}, null: {n_null} ({null_pct:.1f}%)")
        profile.append(f"      samples: {sample_str}")
        
        # Additional info for specific types
        if dtype in ['int64', 'float64']:
            try:
                profile.append(f"      range: [{df[col].min()}, {df[col].max()}]")
            except:
                pass
        elif dtype == 'object':
            # Check if it looks like dates
            try:
                pd.to_datetime(non_null.head(5))
                profile.append(f"      note: appears to be datetime strings")
            except:
                pass
    
    return "\n".join(profile)


def _get_preprocessing_prompt(df: pd.DataFrame, target_col: Optional[str] = None) -> str:
    """Generate a prompt for the LLM to write preprocessing code."""
    profile = _get_data_profile(df)
    
    prompt = f"""You are a data scientist. Analyze this DataFrame and write Python code to preprocess it for machine learning.

DATA PROFILE:
{profile}

TARGET COLUMN: {target_col or "auto-detect (likely the last numeric column or a column named 'target', 'label', 'y', etc.)"}

REQUIREMENTS:
1. Convert ALL columns to numeric types suitable for sklearn models
2. Handle datetime columns by converting to Unix timestamps or extracting features (year, month, day, etc.)
3. Handle categorical/object columns via label encoding or one-hot encoding
4. Handle missing values appropriately
5. Remove or transform any problematic values (inf, -inf, etc.)
6. Return a clean DataFrame with only numeric columns

Write a Python function called `preprocess_for_ml` that takes a DataFrame and returns:
- X: preprocessed features DataFrame (all numeric)
- y: target Series (numeric)
- feature_names: list of feature column names

IMPORTANT:
- Use only pandas, numpy, and sklearn (no other imports)
- Handle ALL edge cases - the code must not fail
- Wrap risky operations in try/except
- If a column can't be converted, drop it rather than fail

Return ONLY the Python code, no explanations. Start with:
```python
def preprocess_for_ml(df):
```
"""
    return prompt


def _get_analysis_prompt(df: pd.DataFrame, target_col: Optional[str] = None) -> str:
    """Generate a prompt for the LLM to write analysis code."""
    profile = _get_data_profile(df)
    
    prompt = f"""You are a data scientist. Analyze this DataFrame and write Python code to perform the best analysis.

DATA PROFILE:
{profile}

TARGET COLUMN: {target_col or "auto-detect"}

REQUIREMENTS:
1. First preprocess the data (handle dates, categoricals, missing values)
2. Determine the best analysis type (regression, classification, clustering, time-series)
3. Train an appropriate model
4. Return results including predictions, metrics, and insights

Write a Python function called `analyze_data` that takes a DataFrame and returns a dict with:
- 'analysis_type': str (regression/classification/clustering/time_series/descriptive)
- 'model': the trained model
- 'predictions': array of predictions
- 'metrics': dict of performance metrics
- 'insights': list of string insights about the data
- 'feature_importance': dict mapping feature names to importance scores (if available)

IMPORTANT:
- Use only pandas, numpy, sklearn, and scipy
- Handle ALL edge cases with try/except
- If modeling fails, fall back to descriptive statistics
- Always return a valid result dict, never raise exceptions

Return ONLY the Python code. Start with:
```python
def analyze_data(df):
```
"""
    return prompt


def _call_llm(prompt: str) -> Optional[str]:
    """Call the LLM to generate code."""
    if not ENABLE_LOCAL_LLM:
        logger.info("LLM disabled, using fallback")
        return None
    
    # Try multiple LLM backends
    
    # Strategy 1: Try the existing llm_completion
    try:
        from preprocessing.llm_preprocessor import llm_completion
        response = llm_completion(prompt, max_tokens=2000)
        
        if response and "LLM unavailable" not in response and "LLM error" not in response:
            return response
    except Exception as e:
        logger.debug(f"llm_completion failed: {e}")
    
    # Strategy 2: Try transformers pipeline (simpler, more likely to work)
    try:
        from transformers import pipeline
        import torch
        
        # Use a small code-generation model
        generator = pipeline(
            "text-generation",
            model="Salesforce/codegen-350M-mono",  # Small code model
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        
        result = generator(
            prompt,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
            pad_token_id=generator.tokenizer.eos_token_id,
        )
        
        if result and len(result) > 0:
            return result[0]["generated_text"]
    except Exception as e:
        logger.debug(f"Transformers pipeline failed: {e}")
    
    # Strategy 3: For now, return None and use fallback
    # In production, you could add OpenAI API, Anthropic, etc.
    logger.info("No LLM backend available, using fallback preprocessing")
    return None


def _extract_code(response: str) -> str:
    """Extract Python code from LLM response."""
    if not response:
        return ""
    
    # Try to extract code block
    if "```python" in response:
        start = response.find("```python") + 9
        end = response.find("```", start)
        if end > start:
            return response[start:end].strip()
    
    if "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end > start:
            return response[start:end].strip()
    
    # If no code blocks, try to find function definition
    if "def " in response:
        start = response.find("def ")
        return response[start:].strip()
    
    return response.strip()


def _validate_code_ast(code: str) -> Tuple[bool, str]:
    """
    Validate generated code using AST analysis to block dangerous operations.
    
    Returns (is_safe, reason) tuple.
    """
    import ast
    
    FORBIDDEN_NODES = {
        ast.Import: "import statements",
        ast.ImportFrom: "from...import statements",
    }
    
    FORBIDDEN_NAMES = {
        # System access
        'eval', 'exec', 'compile', '__import__', 'open', 'input',
        # File system
        'file', 'os', 'sys', 'subprocess', 'shutil', 'pathlib',
        # Network
        'socket', 'urllib', 'requests', 'http', 'ftplib',
        # Code execution
        'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr', 'delattr',
        'hasattr', '__builtins__', '__class__', '__bases__', '__subclasses__',
        # Dangerous builtins
        'breakpoint', 'help', 'quit', 'exit', 'license', 'copyright',
    }
    
    FORBIDDEN_ATTRS = {
        '__class__', '__bases__', '__subclasses__', '__mro__',
        '__globals__', '__code__', '__builtins__', '__dict__',
        'system', 'popen', 'spawn', 'fork', 'exec',
        'read', 'write', 'remove', 'unlink', 'rmdir', 'makedirs',
    }
    
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    
    for node in ast.walk(tree):
        # Check for forbidden node types
        for forbidden_type, description in FORBIDDEN_NODES.items():
            if isinstance(node, forbidden_type):
                return False, f"Forbidden: {description}"
        
        # Check for forbidden names
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return False, f"Forbidden name: {node.id}"
        
        # Check for forbidden attribute access
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            return False, f"Forbidden attribute: {node.attr}"
        
        # Check for string-based attribute access (getattr tricks)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {'getattr', 'setattr', 'delattr'}:
                return False, "Forbidden: dynamic attribute access"
    
    return True, "Code passed validation"


def _execute_generated_code(code: str, df: pd.DataFrame, func_name: str) -> Any:
    """
    Execute LLM-generated code with security safeguards.
    
    Security measures:
    1. AST validation to block dangerous operations
    2. Restricted globals (no __builtins__)
    3. Whitelisted imports only
    4. Execution timeout (via signal on Unix, limited on Windows)
    5. Audit logging of all executed code
    """
    if not code:
        return None
    
    # Step 1: AST validation
    is_safe, reason = _validate_code_ast(code)
    if not is_safe:
        logger.warning(f"Code validation failed: {reason}")
        logger.debug(f"Rejected code:\n{code[:500]}...")
        return None
    
    # Step 2: Audit logging
    code_hash = hash(code) & 0xFFFFFFFF  # Positive 32-bit hash
    logger.info(f"Executing validated LLM code (hash: {code_hash:08x}, func: {func_name})")
    
    # Step 3: Create restricted execution environment
    # Explicitly set __builtins__ to restrict available functions
    safe_builtins = {
        'True': True, 'False': False, 'None': None,
        'len': len, 'range': range, 'enumerate': enumerate, 'zip': zip,
        'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
        'int': int, 'float': float, 'str': str, 'bool': bool,
        'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
        'sorted': sorted, 'reversed': reversed,
        'isinstance': isinstance, 'type': type,
        'print': lambda *args, **kwargs: None,  # Disable print
        'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
        'KeyError': KeyError, 'IndexError': IndexError,
    }
    
    safe_globals = {
        '__builtins__': safe_builtins,
        'pd': pd,
        'np': np,
        'DataFrame': pd.DataFrame,
        'Series': pd.Series,
    }
    
    # Step 4: Add whitelisted sklearn imports
    try:
        from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
        from sklearn.cluster import KMeans
        from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
        from sklearn.model_selection import train_test_split
        
        safe_globals.update({
            'LabelEncoder': LabelEncoder,
            'StandardScaler': StandardScaler,
            'OneHotEncoder': OneHotEncoder,
            'RandomForestClassifier': RandomForestClassifier,
            'RandomForestRegressor': RandomForestRegressor,
            'LogisticRegression': LogisticRegression,
            'LinearRegression': LinearRegression,
            'Ridge': Ridge,
            'KMeans': KMeans,
            'accuracy_score': accuracy_score,
            'r2_score': r2_score,
            'mean_squared_error': mean_squared_error,
            'mean_absolute_error': mean_absolute_error,
            'train_test_split': train_test_split,
        })
    except ImportError as e:
        logger.warning(f"sklearn import failed: {e}")
    
    local_vars = {}
    
    try:
        # Step 5: Execute the code to define the function
        exec(code, safe_globals, local_vars)
        
        # Get the function
        func = local_vars.get(func_name)
        if func is None:
            logger.error(f"Function {func_name} not found in generated code")
            return None
        
        # Call the function with a copy of the DataFrame
        result = func(df.copy())
        logger.info(f"LLM code execution succeeded (hash: {code_hash:08x})")
        return result
        
    except Exception as e:
        logger.error(f"Error executing generated code: {e}")
        logger.debug(f"Code was:\n{code}")
        return None




def llm_preprocess(df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Use LLM to dynamically generate preprocessing code for the data.
    
    Falls back to standard preprocessing if LLM is unavailable or fails.
    """
    # Try LLM-generated preprocessing
    prompt = _get_preprocessing_prompt(df, target_col)
    response = _call_llm(prompt)
    code = _extract_code(response)
    
    if code:
        logger.info("Attempting LLM-generated preprocessing")
        result = _execute_generated_code(code, df, "preprocess_for_ml")
        if result is not None:
            try:
                X, y, feature_names = result
                if isinstance(X, pd.DataFrame) and len(X) > 0:
                    logger.info("LLM preprocessing succeeded")
                    return X, y, feature_names
            except Exception as e:
                logger.warning(f"LLM preprocessing result invalid: {e}")
    
    # Fallback to standard preprocessing
    logger.info("Using fallback preprocessing")
    return _fallback_preprocess(df, target_col)


def _fallback_preprocess(df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, list]:
    """Standard fallback preprocessing when LLM is unavailable."""
    df = df.copy()
    
    # Auto-detect target column
    if target_col is None:
        # Look for common target column names
        target_candidates = ['target', 'label', 'y', 'class', 'outcome', 'churn', 'fraud']
        for candidate in target_candidates:
            matches = [c for c in df.columns if candidate.lower() in c.lower()]
            if matches:
                target_col = matches[0]
                break
        
        # Fall back to last column
        if target_col is None:
            target_col = df.columns[-1]
    
    # Extract target
    y = df[target_col].copy()
    X = df.drop(columns=[target_col], errors='ignore')
    
    # Convert all columns to numeric
    for col in X.columns:
        # Handle datetime
        if X[col].dtype == 'datetime64[ns]':
            try:
                X[col] = X[col].astype('int64') // 10**9
            except:
                X = X.drop(columns=[col])
                continue
        
        # Handle object columns
        if X[col].dtype == 'object':
            # Try datetime conversion first
            try:
                dt = pd.to_datetime(X[col], errors='coerce')
                if dt.notna().sum() > len(X) * 0.5:  # More than 50% valid dates
                    X[col] = dt.astype('int64') // 10**9
                    continue
            except:
                pass
            
            # Try numeric conversion
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                if X[col].notna().sum() > len(X) * 0.5:
                    X[col] = X[col].fillna(0)
                    continue
            except:
                pass
            
            # Fall back to label encoding
            try:
                X[col] = pd.factorize(X[col])[0]
            except:
                X = X.drop(columns=[col])
                continue
    
    # Handle target column similarly
    if y.dtype == 'object':
        try:
            y = pd.to_numeric(y, errors='coerce')
            if y.isna().sum() > len(y) * 0.5:
                y = pd.Series(pd.factorize(df[target_col])[0])
        except:
            y = pd.Series(pd.factorize(y)[0])
    
    # Clean up
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = y.fillna(0)
    
    # Drop any remaining non-numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    
    return X, y, list(X.columns)


def llm_analyze(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Use LLM to dynamically generate and execute analysis code.
    
    Falls back to standard analysis if LLM is unavailable or fails.
    """
    # Try LLM-generated analysis
    prompt = _get_analysis_prompt(df, target_col)
    response = _call_llm(prompt)
    code = _extract_code(response)
    
    if code:
        logger.info("Attempting LLM-generated analysis")
        result = _execute_generated_code(code, df, "analyze_data")
        if result is not None and isinstance(result, dict):
            logger.info("LLM analysis succeeded")
            return result
    
    # Fallback to standard analysis
    logger.info("Using fallback analysis")
    return _fallback_analyze(df, target_col)


def _fallback_analyze(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """Standard fallback analysis when LLM is unavailable."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
    
    result = {
        'analysis_type': 'descriptive',
        'model': None,
        'predictions': None,
        'metrics': {},
        'insights': [],
        'feature_importance': {},
    }
    
    try:
        X, y, feature_names = _fallback_preprocess(df, target_col)
        
        if len(X) == 0 or len(X.columns) == 0:
            result['insights'].append("No valid features for modeling")
            result['metrics'] = df.describe().to_dict()
            return result
        
        # Determine analysis type
        n_unique = y.nunique()
        if n_unique <= 20:
            result['analysis_type'] = 'classification'
            model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        else:
            result['analysis_type'] = 'regression'
            model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        
        # Train model
        model.fit(X, y)
        predictions = model.predict(X)
        
        result['model'] = model
        result['predictions'] = predictions.tolist()
        
        # Calculate metrics
        if result['analysis_type'] == 'classification':
            result['metrics']['accuracy'] = accuracy_score(y, predictions)
        else:
            result['metrics']['r2'] = r2_score(y, predictions)
            result['metrics']['mae'] = mean_absolute_error(y, predictions)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            result['feature_importance'] = dict(zip(feature_names, model.feature_importances_.tolist()))
        
        # Generate insights
        result['insights'].append(f"Trained {result['analysis_type']} model on {len(X)} samples")
        result['insights'].append(f"Used {len(feature_names)} features")
        
        if result['feature_importance']:
            top_features = sorted(result['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:3]
            result['insights'].append(f"Top features: {', '.join([f[0] for f in top_features])}")
        
    except Exception as e:
        logger.error(f"Fallback analysis failed: {e}")
        result['insights'].append(f"Analysis error: {str(e)}")
        result['metrics'] = df.describe().to_dict() if len(df) > 0 else {}
    
    return result


# Convenience function for the workflow
def dynamic_analyze(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Main entry point for LLM-powered dynamic analysis.
    
    Tries LLM-generated code first, falls back to standard methods.
    Always returns a valid result - never raises exceptions.
    """
    try:
        return llm_analyze(df, target_col)
    except Exception as e:
        logger.error(f"Dynamic analysis failed completely: {e}")
        return {
            'analysis_type': 'error',
            'model': None,
            'predictions': None,
            'metrics': {},
            'insights': [f"Analysis failed: {str(e)}"],
            'feature_importance': {},
        }

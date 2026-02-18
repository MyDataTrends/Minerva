
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DebugFilter")

def test_debug_filter():
    print("Testing Filter Logic (Mock)...")
    
    # 1. Simulate data loading from CSV where numbers might be strings
    data = {"Sales": ["100", "200", "150"], "Region": ["North", "South", "East"]}
    df = pd.DataFrame(data)
    
    column = "Sales"
    operator = ">"
    value = 120.0
    
    print(f"DEBUG INPUT: col={column}, op={operator}, val={value} (type={type(value)})")
    print(f"DEBUG DF DTYPE BEFORE: {df[column].dtype}")
    
    # Logic to fix:
    is_numeric_value = isinstance(value, (int, float))
    
    # Check if column is numeric
    is_numeric_col = pd.api.types.is_numeric_dtype(df[column])
    
    print(f"DEBUG CHECK: is_numeric_value={is_numeric_value}, is_numeric_col={is_numeric_col}")

    if is_numeric_value and not is_numeric_col:
        try:
            print("DEBUG: Converting column to numeric...")
            df = df.copy()
            df[column] = pd.to_numeric(df[column], errors='coerce')
            print(f"DEBUG DF DTYPE AFTER: {df[column].dtype}")
        except (ValueError, TypeError) as e:
            print(f"DEBUG: Conversion failed: {e}")
    
    # Apply filter
    try:
        if operator == "==":
            mask = df[column] == value
        elif operator == ">":
            mask = df[column] > value
        elif operator == "<":
            mask = df[column] < value
            
        res = df[mask]
        print("✅ Filter Success")
        print(res)
    except Exception as e:
        print(f"❌ Filter Failed: {e}")

if __name__ == "__main__":
    test_debug_filter()

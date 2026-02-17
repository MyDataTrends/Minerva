import sys
import os
from pathlib import Path

# Print sys.path to verify
print(f"PYTHONPATH: {sys.path}")

try:
    print("Attempting import of llm_manager.llm_interface...")
    from llm_manager.llm_interface import get_llm_completion
    print("Import successful.")
    
    print("Attempting completion...")
    res = get_llm_completion("Test prompt: indicate if working.", max_tokens=10)
    print(f"Result: {res}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

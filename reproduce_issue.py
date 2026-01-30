
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath("C:\\Projects\\Minerva\\Minerva"))

try:
    print("Attempting to import orchestration.orchestrate_workflow...")
    from orchestration.orchestrate_workflow import run_workflow
    print("Successfully imported orchestration.orchestrate_workflow")
    
    print("Attempting to import preprocessing.llm_summarizer...")
    from preprocessing.llm_summarizer import generate_summary
    print("Successfully imported preprocessing.llm_summarizer")
    
    print("Imports successful! definitions found:")
    print(f"run_workflow: {run_workflow}")
    print(f"generate_summary: {generate_summary}")
    
except ImportError as e:
    print(f"ImportError caught: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)


import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath("C:\\Projects\\Minerva\\Minerva"))

try:
    print("Attempting to import chatbot.llm_intent_classifier...")
    from chatbot import llm_intent_classifier
    print("Successfully imported chatbot.llm_intent_classifier")
    
    print("Attempting to import cli...")
    import cli
    print("Successfully imported cli")
    
except ImportError as e:
    print(f"ImportError caught: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)

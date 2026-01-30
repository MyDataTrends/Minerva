
import os
from llama_cpp import Llama

# Path to your model
MODEL_PATH = r"C:\Projects\Minerva\Minerva\models\Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at {MODEL_PATH}")
    exit(1)

print(f"Loading model from {MODEL_PATH}...")
print("Attempting to offload 33 layers to GPU...")

try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=33, 
        n_ctx=2048,
        verbose=True  # This forces the C++ logs to show up
    )
    
    print("\nAttempting generation...")
    output = llm("Q: Name the planets in the solar system. A: ", max_tokens=32, stop=["Q:", "\n"])
    print("\nGeneration Result:")
    print(output)
    
except Exception as e:
    print(f"Error: {e}")

"""Quick GPU verification script."""
import sys
print("Testing llama-cpp-python GPU support...")

try:
    from llama_cpp import Llama
    
    # Load model with GPU layers
    llm = Llama(
        model_path="models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        n_gpu_layers=33,  # Offload all layers to GPU
        n_ctx=2048,
        verbose=True  # Show CUDA info
    )
    
    print("\n=== QUICK GENERATION TEST ===")
    output = llm("Hello", max_tokens=10)
    print(f"Output: {output['choices'][0]['text']}")
    print("\n✅ If you see 'CUDA' or 'GPU' in the logs above, success!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

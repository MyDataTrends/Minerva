"""
LLM Server - Runs llama-cpp-python in isolated process.

This script is meant to be run as a separate process, not imported.
It provides an HTTP API for LLM completions, isolating the C++ model
from Streamlit's threading model.

Usage:
    python -m llm_manager.llm_server --model /path/to/model.gguf --port 8765
"""
import argparse
import logging
import sys
import os
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("server_debug.log", mode='w')
    ]
)
logger = logging.getLogger("llm_server")

# Globals
_llm = None
_model_path = None


def load_model(model_path: str, n_ctx: int = 2048, n_gpu_layers: int = 0):
    """Load the LLM model."""
    global _llm, _model_path
    
    if _llm is not None and _model_path == model_path:
        logger.info("Model already loaded")
        return True
    
    try:
        from llama_cpp import Llama
        
        logger.info(f"Loading model: {model_path}")
        logger.info(f"Context: {n_ctx}, GPU layers: {n_gpu_layers}")
        
        _llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        _model_path = model_path
        
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def create_app():
    """Create Flask app."""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify({
            "status": "ok",
            "model_loaded": _llm is not None,
            "model_path": _model_path,
        })
    
    @app.route("/complete", methods=["POST"])
    def complete():
        """Generate completion."""
        if _llm is None:
            return jsonify({"error": "Model not loaded"}), 503
        
        data = request.json or {}
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 256)
        temperature = data.get("temperature", 0.7)
        stop = data.get("stop", ["</s>", "<|eot_id|>", "<|end_of_text|>"])
        
        try:
            result = _llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
            text = result["choices"][0]["text"].strip()
            if not text:
               logger.warning("Model generated empty text")
               
            return jsonify({
                "text": text,
                "usage": result.get("usage", {}),
            })
        except Exception as e:
            logger.error(f"Completion error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/chat", methods=["POST"])
    def chat():
        """Chat completion."""
        import time
        start_time = time.time()
        
        if _llm is None:
            return jsonify({"error": "Model not loaded"}), 503
        
        data = request.json or {}
        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens", 256)
        temperature = data.get("temperature", 0.7)
        
        try:
            if hasattr(_llm, "create_chat_completion"):
                result = _llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text = result["choices"][0]["message"]["content"].strip()
                usage = result.get("usage", {})
            else:
                # Fallback to prompt
                prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
                prompt += "\nASSISTANT:"
                result = _llm(prompt, max_tokens=max_tokens, temperature=temperature)
                text = result["choices"][0]["text"].strip()
                usage = result.get("usage", {})
            
            duration = time.time() - start_time
            logger.info(f"Chat completed in {duration:.2f}s | Tokens: {usage.get('total_tokens', 'N/A')}")
            
            return jsonify({"text": text})
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/load", methods=["POST"])
    def load():
        """Load a model."""
        data = request.json or {}
        model_path = data.get("model_path")
        n_ctx = data.get("n_ctx", 2048)
        n_gpu_layers = data.get("n_gpu_layers", 0)
        
        if not model_path:
            return jsonify({"error": "model_path required"}), 400
        
        if not Path(model_path).exists():
            return jsonify({"error": f"Model not found: {model_path}"}), 404
        
        success = load_model(model_path, n_ctx, n_gpu_layers)
        if success:
            return jsonify({"status": "loaded", "model_path": model_path})
        else:
            return jsonify({"error": "Failed to load model"}), 500
    
    @app.route("/unload", methods=["POST"])
    def unload():
        """Unload the model."""
        global _llm, _model_path
        
        if _llm is not None:
            del _llm
            _llm = None
            _model_path = None
            logger.info("Model unloaded")
        
        return jsonify({"status": "unloaded"})
    
    return app


def main():
    parser = argparse.ArgumentParser(description="LLM Server")
    parser.add_argument("--model", type=str, help="Path to model file")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--n-ctx", type=int, default=2048, help="Context size")
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="GPU layers")
    
    args = parser.parse_args()
    
    # Load model if specified
    if args.model:
        if not load_model(args.model, args.n_ctx, args.n_gpu_layers):
            logger.error("Failed to load initial model")
            # Continue anyway - can load via API
    
    # Start server
    app = create_app()
    logger.info(f"Starting LLM server on {args.host}:{args.port}")
    
    # Use waitress for production-quality server
    try:
        from waitress import serve
        serve(app, host=args.host, port=args.port, threads=1)
    except ImportError:
        # Fallback to Flask dev server
        logger.warning("waitress not installed, using Flask dev server")
        app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()

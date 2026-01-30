"""
LLM Subprocess Manager - Start and manage the LLM server process.

Handles:
- Starting the LLM server subprocess
- Health checks and auto-restart
- Clean shutdown
- Loading/unloading models via HTTP
"""
import subprocess
import threading
import time
import logging
import os
import sys
import socket
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Default server settings
DEFAULT_PORT = 8765
DEFAULT_HOST = "127.0.0.1"
SERVER_SCRIPT = Path(__file__).parent / "llm_server.py"


class LLMSubprocess:
    """
    Manages an LLM server running in a separate process.
    
    This isolates llama-cpp-python from the main application,
    preventing C-level crashes from killing the app.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._process: Optional[subprocess.Popen] = None
        self._port = DEFAULT_PORT
        self._host = DEFAULT_HOST
        self._model_path: Optional[str] = None
        self._n_gpu_layers: int = 0
        self._lock = threading.Lock()
        self._health_thread: Optional[threading.Thread] = None
        self._running = False
        self._initialized = True
    
    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"
    
    def is_running(self) -> bool:
        """Check if server is running and healthy."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False
    
    def _find_free_port(self) -> int:
        """Find an available port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    
    def start(self, port: Optional[int] = None) -> bool:
        """
        Start the LLM server subprocess.
        
        Returns True if server started successfully.
        """
        with self._lock:
            if self.is_running():
                logger.info("LLM server already running")
                return True
            
            # Use specified port or find free one
            self._port = port or self._find_free_port()
            
            # Find the correct Python executable - prefer venv
            python_exe = sys.executable
            
            # Check if we're in a venv but sys.executable is wrong
            script_dir = Path(__file__).resolve().parent.parent
            venv_python = script_dir / ".venv" / "Scripts" / "python.exe"
            if venv_python.exists():
                python_exe = str(venv_python)
                logger.info(f"Using venv Python: {python_exe}")
            
            cmd = [
                python_exe,
                str(SERVER_SCRIPT),
                "--port", str(self._port),
                "--host", self._host,
            ]
            
            # Add model if previously loaded
            if self._model_path:
                cmd.extend(["--model", self._model_path])
                if self._n_gpu_layers > 0:
                    cmd.extend(["--n-gpu-layers", str(self._n_gpu_layers)])
            
            logger.info(f"Starting LLM server: {' '.join(cmd)}")
            
            try:
                # Start process
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                )
                
                # Wait for server to be ready
                for _ in range(30):  # 30 second timeout
                    time.sleep(1)
                    if self.is_running():
                        logger.info(f"LLM server started on port {self._port}")
                        self._running = True
                        return True
                    
                    # Check if process died
                    if self._process.poll() is not None:
                        stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                        logger.error(f"LLM server died: {stderr}")
                        return False
                
                logger.error("LLM server startup timeout")
                self.stop()
                return False
                
            except Exception as e:
                logger.error(f"Failed to start LLM server: {e}")
                return False
    
    def stop(self) -> None:
        """Stop the LLM server subprocess."""
        with self._lock:
            self._running = False
            
            if self._process is not None:
                logger.info("Stopping LLM server...")
                try:
                    self._process.terminate()
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                except Exception:
                    pass
                finally:
                    self._process = None
    
    def load_model(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = 0) -> bool:
        """
        Load a model in the server.
        
        Starts the server if not running.
        """
        # Ensure server is running
        if not self.is_running():
            self._model_path = model_path
            self._n_gpu_layers = n_gpu_layers
            if not self.start():
                return False
            # Model loaded during startup
            return self.is_running()
        
        # Load via API
        try:
            resp = requests.post(
                f"{self.base_url}/load",
                json={
                    "model_path": model_path,
                    "n_ctx": n_ctx,
                    "n_gpu_layers": n_gpu_layers,
                },
                timeout=120,  # Model loading can take a while
            )
            
            if resp.status_code == 200:
                self._model_path = model_path
                logger.info(f"Model loaded: {model_path}")
                return True
            else:
                logger.error(f"Failed to load model: {resp.json()}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def unload_model(self) -> bool:
        """Unload the current model."""
        if not self.is_running():
            return True
        
        try:
            resp = requests.post(f"{self.base_url}/unload", timeout=10)
            if resp.status_code == 200:
                self._model_path = None
                return True
        except Exception:
            pass
        return False
    
    def complete(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stop: Optional[list] = None,
    ) -> str:
        """
        Generate a completion.
        
        Returns empty string on error.
        """
        if not self.is_running():
            if not self.start():
                return ""
        
        try:
            resp = requests.post(
                f"{self.base_url}/complete",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": stop or ["</s>", "\n\n"],
                },
                timeout=1200,  # 20 minutes
            )
            
            if resp.status_code == 200:
                return resp.json().get("text", "")
            else:
                logger.error(f"Completion error: {resp.json()}")
                return ""
                
        except Exception as e:
            logger.error(f"Completion request failed: {e}")
            return ""
    
    def chat(
        self,
        messages: list,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a chat response.
        
        Returns empty string on error.
        """
        if not self.is_running():
            if not self.start():
                return ""
        
        try:
            resp = requests.post(
                f"{self.base_url}/chat",
                json={
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=1200,  # 20 minutes
            )
            
            if resp.status_code == 200:
                return resp.json().get("text", "")
            else:
                logger.error(f"Chat error: {resp.json()}")
                return ""
                
        except Exception as e:
            logger.error(f"Chat request failed: {e}")
            return ""
    
    def get_status(self) -> dict:
        """Get server status."""
        if not self.is_running():
            return {
                "running": False,
                "port": self._port,
                "model_path": self._model_path,
            }
        
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=2)
            data = resp.json()
            data["running"] = True
            data["port"] = self._port
            return data
        except Exception:
            return {
                "running": False,
                "port": self._port,
                "model_path": self._model_path,
            }


# Convenience function
def get_llm_subprocess() -> LLMSubprocess:
    """Get the global LLM subprocess manager."""
    return LLMSubprocess()

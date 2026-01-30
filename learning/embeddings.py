"""
Embedding model wrapper for generating vector representations of code and text.
Uses FastEmbed for lightweight, fast, local inference if available, 
falling back to SentenceTransformers.
"""
import os
from typing import List, Union
import numpy as np

# Default to a high-quality but small model
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"

class EmbeddingModel:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._client = None
        self._provider = None
        self._load_model()

    def _load_model(self):
        """Load the embedding model."""
        try:
            # Try FastEmbed first (lighter, faster ONNX runtime)
            from fastembed import TextEmbedding
            # FastEmbed supports BAAI/bge-small-en-v1.5, let's map base to supported if needed
            # For now, let's just interpret the model name loosely or default to supported
            # FastEmbed's default is often BAAI/bge-small-en-v1.5 which is very good
            self._client = TextEmbedding(model_name=self.model_name)
            self._provider = "fastembed"
            print(f"Loaded embedding model {self.model_name} via FastEmbed")
        except ImportError:
            try:
                # Fallback to SentenceTransformers (pytorch dependency)
                from sentence_transformers import SentenceTransformer
                self._client = SentenceTransformer(self.model_name)
                self._provider = "sentence_transformers"
                print(f"Loaded embedding model {self.model_name} via SentenceTransformers")
            except ImportError:
                raise ImportError(
                    "No embedding library found. Please install 'fastembed' (recommended) "
                    "or 'sentence-transformers'."
                )

    def embed(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for text.
        
        Args:
            text: Single string or list of strings
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(text, str):
            text = [text]

        if self._provider == "fastembed":
            # FastEmbed returns a generator of numpy arrays
            embeddings = list(self._client.embed(text))
            # Convert to single numpy array
            return np.array(embeddings)
            
        elif self._provider == "sentence_transformers":
            return self._client.encode(text, convert_to_numpy=True)
            
        return np.array([])

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.
        Specific method in case we need to add instructions (common for BGE models)
        """
        # BGE models work best with "Represent this sentence for searching relevant passages:"
        # But for code retrieval, direct matching often works well.
        # We can add a prefix if specific model requires it.
        # For now, keep it simple.
        embeddings = self.embed([query])
        return embeddings[0]

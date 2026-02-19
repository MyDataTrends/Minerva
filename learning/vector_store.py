"""
Vector Store implementation using SQLite for local, lightweight RAG.
Stores code examples and their embeddings for retrieval during chat.
"""
import sqlite3
import json
import numpy as np
import os
from typing import List, Dict, Optional, Any
from datetime import datetime

class VectorStore:
    def __init__(self, db_path: str = "assay_memory.db"):
        """Initialize the vector store with a SQLite database."""
        # Ensure directory exists if path contains one
        if os.path.dirname(db_path):
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create the necessary tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Enable WAL mode for better concurrency
        c.execute("PRAGMA journal_mode=WAL;")
        
        # Create examples table
        # embedding is stored as a BLOB (numpy array bytes)
        c.execute("""
            CREATE TABLE IF NOT EXISTS examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intent TEXT NOT NULL,
                code TEXT NOT NULL,
                explanation TEXT,
                embedding BLOB NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source TEXT DEFAULT 'user_feedback'
            )
        """)
        
        # Create index on source for valid filtering
        c.execute("CREATE INDEX IF NOT EXISTS idx_source ON examples(source)")
        
        conn.commit()
        conn.close()

    def add_example(self, 
                   intent: str, 
                   code: str, 
                   embedding: np.ndarray, 
                   explanation: str = "",
                   metadata: Dict[str, Any] = None,
                   source: str = "user_feedback") -> int:
        """
        Add a new code example to the store.
        
        Args:
            intent: The natural language request (e.g., "plot correlation matrix")
            code: The Python code implementing the intent
            embedding: Number array representing the intent semantics
            explanation: Optional text explaining the code
            metadata: Additional info (e.g., user_rating, execution_time)
            source: Where this came from ('user_feedback', 'kaggle', 'cortex')
            
        Returns:
            The ID of the inserted row
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Ensure embedding is float32 for consistency
        embedding = embedding.astype(np.float32)
        embedding_blob = embedding.tobytes()
        
        metadata_json = json.dumps(metadata) if metadata else "{}"
        
        c.execute("""
            INSERT INTO examples (intent, code, explanation, embedding, metadata, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (intent, code, explanation, embedding_blob, metadata_json, source))
        
        row_id = c.lastrowid
        conn.commit()
        conn.close()
        return row_id

    def search(self, query_embedding: np.ndarray, limit: int = 3, threshold: float = 0.0) -> List[Dict]:
        """
        Find the most similar examples to the query embedding.
        Uses cosine similarity.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        # Retrieve all embeddings to calculate similarity in numpy
        # For a local app with <100k examples, this is fast enough.
        # If it grows, we can switch to using sqlite-vss or similar, 
        # but pure numpy is surprisingly robust for this scale.
        c.execute("SELECT id, intent, code, explanation, embedding, metadata, source FROM examples")
        rows = c.fetchall()
        
        if not rows:
            conn.close()
            return []
            
        # Unpack embeddings
        ids = []
        vectors = []
        data_map = {}
        
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
            
        for row in rows:
            vec_blob = row['embedding']
            vec = np.frombuffer(vec_blob, dtype=np.float32)
            
            ids.append(row['id'])
            vectors.append(vec)
            
            data_map[row['id']] = {
                'id': row['id'],
                'intent': row['intent'],
                'code': row['code'],
                'explanation': row['explanation'],
                'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                'source': row['source']
            }
            
        # Matrix multiplication for cosine similarity
        # (N, D) @ (D,) -> (N,)
        matrix = np.vstack(vectors)
        norms = np.linalg.norm(matrix, axis=1)
        
        # Avoid division by zero
        norms[norms == 0] = 1e-10
        
        scores = np.dot(matrix, query_embedding) / (norms * query_norm)
        
        # Sort by score descending
        # argsort gives ascending, so we reverse
        top_indices = np.argsort(scores)[-limit:][::-1]
        
        results = []
        for idx in top_indices:
            score = scores[idx]
            if score >= threshold:
                item_id = ids[idx]
                item = data_map[item_id]
                item['score'] = float(score)
                results.append(item)
                
        conn.close()
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the stored examples."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*), source FROM examples GROUP BY source")
        stats = {row[1]: row[0] for row in c.fetchall()}
        c.execute("SELECT COUNT(*) FROM examples")
        total = c.fetchone()[0]
        stats['total'] = total
        conn.close()
        return stats

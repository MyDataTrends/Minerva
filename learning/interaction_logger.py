"""
Interaction Logger for Minerva.
Collects user-assistant interactions to build a fine-tuning dataset (GGUF ready).
"""
import json
import os
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

class InteractionLogger:
    def __init__(self, log_dir: str = "local_data/learning", log_file: str = "interactions.jsonl"):
        """Initialize the logger."""
        self.log_dir = log_dir
        self.log_file = log_file
        self.log_path = os.path.join(self.log_dir, self.log_file)
        
        # Ensure directory exists
        if os.path.dirname(self.log_path):
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log_interaction(self, 
                       messages: List[Dict[str, str]], 
                       metadata: Optional[Dict[str, Any]] = None,
                       success: bool = True):
        """
        Log a chat interaction to JSONL.
        
        Args:
            messages: List of message dicts [{"role": "user", "content": ...}, ...]
            metadata: Additional context (e.g. execution time, dataset schema)
            success: Whether the interaction was successful (feedback or no error)
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "messages": messages,
            "metadata": metadata or {},
            "success": success
        }
        
        # We append to the file
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Return content statistics for the log file."""
        if not os.path.exists(self.log_path):
            return {"count": 0, "success_count": 0}
            
        count = 0
        success_count = 0
        
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        count += 1
                        if entry.get("success", False):
                            success_count += 1
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
            
        return {
            "total_interactions": count,
            "successful_interactions": success_count,
            "file_size_bytes": os.path.getsize(self.log_path)
        }
        
    def export_to_sharegpt(self, output_path: str = None) -> str:
        """
        Export successful logs to ShareGPT format for fine-tuning.
        Returns the path to the exported file.
        """
        if output_path is None:
            output_path = os.path.join(self.log_dir, "minerva_finetune_data.json")
            
        data = []
        if os.path.exists(self.log_path):
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("success", False):
                            data.append({
                                "conversations": [
                                    {"from": "human" if m["role"] == "user" else "gpt", "value": m["content"]}
                                    for m in entry["messages"]
                                ]
                            })
                    except:
                        continue
                        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        return output_path

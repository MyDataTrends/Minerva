"""
MCP Dataset Resources.

Exposes active session datasets as readable resources.
"""
from __future__ import annotations

from typing import Any, List, Optional, TYPE_CHECKING
import json
from . import BaseResourceProvider, ResourceInfo, register_provider

if TYPE_CHECKING:
    import pandas as pd


class DatasetResourceProvider(BaseResourceProvider):
    """
    Provider for accessing datasets in the current session.
    
    URI Scheme:
        resource://datasets/{dataset_id}
        resource://datasets/list
    """
    
    name = "datasets"
    
    def list_resources(self, session=None) -> List[ResourceInfo]:
        """List available datasets in the session."""
        if not session:
            return []
            
        resources = []
        for dataset_id, metadata in session.dataset_metadata.items():
            resources.append(ResourceInfo(
                uri=f"resource://datasets/{dataset_id}",
                name=f"Dataset: {dataset_id}",
                description=f"Dataset with {metadata.get('rows', '?')} rows and {len(metadata.get('columns', []))} columns",
                mime_type="application/json"
            ))
        return resources
    
    async def read_resource(self, uri: str, session=None) -> Any:
        """Read a dataset resource."""
        if not session:
            raise ValueError("Session required to read dataset resources")
            
        # Parse path
        path = uri.replace("resource://datasets/", "")
        
        if path == "list":
            return self.list_resources(session)
            
        # Get dataset ID
        dataset_id = path
        
        df = session.get_dataset(dataset_id)
        if df is None:
            raise ValueError(f"Dataset not found: {dataset_id}")
            
        # Convert to JSON-oriended dict for resource consumption
        # We limit to first 100 rows by default to avoid huge payloads
        # In a real scenario, we might support pagination params in the URI
        return {
            "metadata": session.dataset_metadata.get(dataset_id, {}),
            "data": df.head(100).to_dict(orient="records"),
            "schema": {
                "columns": list(df.columns),
                "dtypes": {k: str(v) for k, v in df.dtypes.items()}
            }
        }


# Register the provider
register_provider(DatasetResourceProvider())

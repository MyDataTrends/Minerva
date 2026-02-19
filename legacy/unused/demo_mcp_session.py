"""
Assay MCP Demo Script.

This script demonstrates a self-orchestrated analysis session using the Assay MCP server.
It simulates a user request to analyze a dataset, detects anomalies, and creates a visualization.
"""
import asyncio
import logging
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Set up logging to show the flow
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # Simplified format for demo
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("demo")

async def run_demo():
    print("\n" + "="*80)
    print("🚀 ASSAY MCP SERVER DEMO")
    print("="*80 + "\n")
    
    from mcp_server.session import SessionManager
    from mcp_server.tools.connectors import ConnectFileTool
    from mcp_server.tools.analysis import DescribeDatasetTool, DetectAnomaliesTool
    from mcp_server.tools.visualization import SuggestVisualizationsTool, CreateChartTool
    
    # 1. Initialize Session
    print("Expected: Initialize a new session for the user")
    manager = SessionManager()
    session = manager.create_session(user_id="demo_user")
    print(f"✅ Session Created: {session.session_id}")
    print("-" * 40)
    
    # 2. Fabricate Data
    print("\n📝 Generating synthetic sales data...")
    dates = pd.date_range(start="2023-01-01", periods=30)
    sales = np.random.normal(100, 10, size=30)
    sales[15] = 200  # Inject anomaly
    df = pd.DataFrame({"date": dates, "sales": sales, "region": ["North"]*15 + ["South"]*15})
    
    # Save to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w', newline='') as f:
        df.to_csv(f, index=False)
        csv_path = f.name
    print(f"   Created temporary CSV at {csv_path}")

    try:
        # 3. Connect (Load Data)
        print("\n🔧 Tool Call: connect_file")
        tool = ConnectFileTool()
        result = await tool.execute({"path": csv_path}, session=session)
        
        if not result["success"]:
            print(f"❌ Connection Failed: {result}")
            return
            
        dataset_id = result["data"]["connection_id"]
        print(f"✅ Data Loaded. ID: {dataset_id} ({result['data']['rows']} rows)")
        
        # 4. Analyze
        print("\n🔍 Tool Call: detect_anomalies")
        tool = DetectAnomaliesTool()
        result = await tool.execute({"dataset_id": dataset_id, "method": "zscore"}, session=session)
        anomalies = result["data"]["anomalies"]
        print(f"✅ Anomalies Detected: {json_str(anomalies)}")
        
        # 5. Visualize
        print("\n📊 Tool Call: suggest_visualizations")
        tool = SuggestVisualizationsTool()
        result = await tool.execute({"dataset_id": dataset_id}, session=session)
        suggestion = result["data"]["suggestions"][0]
        print(f"✅ Suggestion: {suggestion['type']} chart ({suggestion['reason']})")
        
        print("\n🎨 Tool Call: create_chart")
        tool = CreateChartTool()
        result = await tool.execute({
            "dataset_id": dataset_id,
            "chart_type": "line", 
            "x": "date", 
            "y": "sales",
            "title": "Sales Over Time (Anomaly Detection)"
        }, session=session)
        chart_id = result["data"]["chart_id"]
        print(f"✅ Chart Created: {chart_id}")
        
        # 6. Resources
        print("\n📦 Resource Access: resource://datasets/list")
        from mcp_server.resources import list_all_resources
        resources = list_all_resources()
        print(f"✅ Available Resources: {len(resources)}")
        for r in resources:
            if "dataset" in r.uri:
                print(f"   - {r.uri} ({r.name})")

    finally:
        import os
        if os.path.exists(csv_path):
            os.unlink(csv_path)

    print("\n" + "="*80)
    print("✨ DEMO COMPLETE")
    print("="*80 + "\n")

def json_str(d):
    import json
    return json.dumps(d, default=str)

if __name__ == "__main__":
    asyncio.run(run_demo())

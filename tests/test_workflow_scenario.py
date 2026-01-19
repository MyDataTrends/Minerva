"""
End-to-End Workflow Test for MCP Server.

Simulates a user session:
1. Connect to data (CSV)
2. Analyze data (Describe, Anomalies)
3. Visualize data (Charts)
4. Verify session state
"""
import pytest
import asyncio
import pandas as pd
import tempfile
import os
import logging
from mcp_server.session import SessionManager
from mcp_server.tools.connectors import ConnectFileTool
from mcp_server.tools.analysis import DescribeDatasetTool, DetectAnomaliesTool
from mcp_server.tools.visualization import CreateChartTool, SuggestVisualizationsTool

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("workflow_test")

class TestWorkflowScenario:
    @pytest.mark.asyncio
    async def test_end_to_end_analysis(self):
        # 0. Setup
        manager = SessionManager()
        session = manager.create_session(user_id="test_user")
        logger.info(f"Created session: {session.session_id}")
        
        # Create dummy CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            f.write("id,category,value,date\n")
            for i in range(100):
                val = i * 1.5
                if i == 50: val = 1000  # Outlier
                f.write(f"{i},cat_{i%3},{val},2023-01-{i%30+1:02d}\n")
            csv_path = f.name
            
        try:
            # 1. Connect (Load Data)
            logger.info("Step 1: Connecting to file...")
            connect_tool = ConnectFileTool()
            result = await connect_tool.execute({"path": csv_path}, session=session)
            assert result["success"] is True
            dataset_id = result["data"]["connection_id"]
            logger.info(f"Loaded dataset: {dataset_id}")
            
            # 2. Analyze (Describe)
            logger.info("Step 2: Describing dataset...")
            describe_tool = DescribeDatasetTool()
            desc_result = await describe_tool.execute({"dataset_id": dataset_id}, session=session)
            assert desc_result["success"] is True
            assert desc_result["data"]["shape"]["rows"] == 100
            
            # 3. Analyze (Anomalies)
            logger.info("Step 3: Detecting anomalies...")
            anomaly_tool = DetectAnomaliesTool()
            anom_result = await anomaly_tool.execute({"dataset_id": dataset_id}, session=session)
            assert anom_result["success"] is True
            # Expect outlier at index 50
            assert "value" in anom_result["data"]["anomalies"]
            
            # 4. Suggest Visualizations
            logger.info("Step 4: Getting suggestions...")
            suggest_tool = SuggestVisualizationsTool()
            sugg_result = await suggest_tool.execute({"dataset_id": dataset_id}, session=session)
            assert sugg_result["success"] is True
            assert len(sugg_result["data"]["suggestions"]) > 0
            
            # 5. Create Chart
            logger.info("Step 5: Creating chart...")
            chart_tool = CreateChartTool()
            chart_result = await chart_tool.execute({
                "dataset_id": dataset_id,
                "chart_type": "scatter",
                "x": "id",
                "y": "value",
                "title": "Value over ID"
            }, session=session)
            assert chart_result["success"] is True
            chart_id = chart_result["data"]["chart_id"]
            logger.info(f"Created chart: {chart_id}")
            
            # Verify session state
            assert dataset_id in session.datasets
            assert chart_id in session.charts
            assert len(session.tool_calls) == 5
            
            logger.info("Workflow completed successfully!")
            
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

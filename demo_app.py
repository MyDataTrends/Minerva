import streamlit as st
import asyncio
import pandas as pd
import numpy as np
import json
import plotly.io as pio
import plotly.graph_objects as go
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="Minerva MCP Demo",
    page_icon="🧠",
    layout="wide"
)

# Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "logs" not in st.session_state:
    st.session_state.logs = []
if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None
if "mcp_manager" not in st.session_state:
    from mcp_server.session import SessionManager
    st.session_state.mcp_manager = SessionManager()

def log(message, level="info"):
    st.session_state.logs.append({"msg": message, "level": level})

async def run_step(step_name, func, *args, **kwargs):
    with st.spinner(f"Running {step_name}..."):
        try:
            result = await func(*args, **kwargs)
            log(f"✅ {step_name} completed", "success")
            return result
        except Exception as e:
            st.error(f"Error in {step_name}: {e}")
            log(f"❌ {step_name} failed: {e}", "error")
            return None

# --- UI Layout ---

st.title("🧠 Minerva MCP Server Demo")
st.markdown("""
This interactive demo showcases the **Model Context Protocol (MCP)** server capabilities.
You will see how an Agent (simulated here) uses tools to:
1. **Connect** to a data source (Real-time Crypto API)
2. **Analyze** the data for anomalies
3. **Visualize** the insights
""")

# Sidebar
with st.sidebar:
    st.header("Control Panel")
    if st.button("Reset Session", type="primary"):
        st.session_state.session_id = None
        st.session_state.dataset_id = None
        st.session_state.logs = []
        st.rerun()
    
    st.subheader("Session Info")
    if st.session_state.session_id:
        st.success(f"Active: {st.session_state.session_id}")
    else:
        st.warning("No active session")

# Main Content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Data Loading")
    
    # Initialize Session Button
    if not st.session_state.session_id:
        if st.button("Start New Session"):
            session = st.session_state.mcp_manager.create_session(user_id="demo_user")
            st.session_state.session_id = session.session_id
            log(f"Session created: {session.session_id}")
            st.rerun()
    
    if st.session_state.session_id:
        session = st.session_state.mcp_manager.get_session(st.session_state.session_id)
        
        # Connection Settings
        with st.expander("🔌 Data Source Settings", expanded=not st.session_state.dataset_id):
            tab_api, tab_upload, tab_sample = st.tabs(["🌐 REST API", "📂 Upload File", "📉 Sample Data"])
            
            # --- TAB 1: API ---
            with tab_api:
                st.caption("Connect to a live API (CryptoCompare).")
                api_url = st.text_input("API Base URL", value="https://min-api.cryptocompare.com")
                
                enable_auth = st.checkbox("Enable Authentication")
                auth_config = {}
                if enable_auth:
                    auth_type = st.selectbox("Auth Type", ["bearer", "api_key", "basic"])
                    auth_token = st.text_input("Token / Key", type="password")
                    
                    if auth_type == "api_key":
                        header_name = st.text_input("Header Name", value="X-API-Key")
                        auth_config = {"type": "api_key", "token": auth_token, "header_name": header_name}
                    elif auth_type == "bearer":
                        auth_config = {"type": "bearer", "token": auth_token}
                    elif auth_type == "basic":
                        st.warning("Basic auth requires 'username:password' as token")
                        auth_config = {"type": "basic", "token": auth_token}

                if st.button("Connect & Fetch (API)"):
                    if not st.session_state.dataset_id:
                         from mcp_server.tools.connectors import ConnectAPITool, FetchAPIDataTool
                         
                         async def load_api_data():
                            # 1. Connect
                            connect_tool = ConnectAPITool()
                            connect_args = {"url": api_url}
                            if enable_auth and auth_token:
                                connect_args["auth"] = auth_config
                                
                            conn_result = await connect_tool.execute(connect_args, session=session)
                            
                            if not conn_result["success"]:
                                return conn_result
                                
                            conn_id = conn_result["data"]["connection_id"]
                            log(f"Connected to API: {conn_id}")
                            
                            # 2. Fetch
                            fetch_tool = FetchAPIDataTool()
                            data_result = await fetch_tool.execute({
                                "connection_id": conn_id,
                                "endpoint": "data/v2/histoday",
                                "params": {"fsym": "BTC", "tsym": "USD", "limit": 30},
                                "save_as": "btc_history"
                            }, session=session)
                            
                            return data_result
                        
                         result = asyncio.run(load_api_data())
                         if result and result["success"]:
                            st.session_state.dataset_id = result["data"]["dataset_id"]
                            st.success(f"Dataset Loaded: {st.session_state.dataset_id}")
                            st.json(result)
                         else:
                            st.error("Failed to load data")
                            if result: st.json(result)

            # --- TAB 2: UPLOAD ---
            with tab_upload:
                st.caption("Upload a CSV file from your computer.")
                uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
                
                if uploaded_file and st.button("Load Uploaded File"):
                    from mcp_server.tools.connectors import ConnectFileTool
                    
                    # Save to temp
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    st.session_state.csv_path = tmp_path
                    st.session_state.cleanup = True
                    
                    # Connect
                    tool = ConnectFileTool()
                    result = asyncio.run(tool.execute({"path": tmp_path}, session=session))
                    
                    if result and result["success"]:
                        st.session_state.dataset_id = result["data"]["connection_id"]
                        st.success(f"File Loaded: {uploaded_file.name}")
                        st.json(result)
                    else:
                        st.error("Failed to load file")

            # --- TAB 3: SAMPLE ---
            with tab_sample:
                st.caption("Use offline sample data (Bitcoin 2024).")
                if st.button("Load Sample Data"):
                    from mcp_server.tools.connectors import ConnectFileTool
                    sample_path = os.path.abspath("data/bitcoin_history.csv")
                    
                    if not os.path.exists(sample_path):
                        st.error(f"Sample file not found at {sample_path}")
                    else:
                        tool = ConnectFileTool()
                        result = asyncio.run(tool.execute({"path": sample_path}, session=session))
                        
                        if result and result["success"]:
                            st.session_state.dataset_id = result["data"]["connection_id"]
                            st.success("Sample Data Loaded")
                            st.json(result)
                        else:
                            st.error("Failed to load sample data")
                            if result: st.json(result)

    st.subheader("3. Visualization")
    if st.session_state.dataset_id:
        if st.button("Create Chart (MCP Tool)"):
            from mcp_server.tools.visualization import CreateChartTool
            
            # Auto-detect columns
            df = session.get_dataset(st.session_state.dataset_id)
            cols = list(df.columns)
            x_col = "time" if "time" in cols else ("date" if "date" in cols else cols[0])
            y_col = "close" if "close" in cols else ("priceUsd" if "priceUsd" in cols else cols[1])

            tool = CreateChartTool()
            
            async def chart():
                return await tool.execute({
                    "dataset_id": st.session_state.dataset_id,
                    "chart_type": "line",
                    "x": x_col,
                    "y": y_col,
                    "title": f"Bitcoin Price ({y_col})"
                }, session=session)
                
            result = asyncio.run(chart())
            if result and result["success"]:
                chart_json = result["data"]["plotly_json"]
                st.session_state.chart_json = chart_json
                st.success("Chart created!")
            else:
                st.error("Chart failed")
                if result: st.write(result)

    if "chart_json" in st.session_state:
        fig = pio.from_json(st.session_state.chart_json)
        st.plotly_chart(fig, use_container_width=True)


with col2:
    st.subheader("2. Assistant & Analysis")
    
    # --- CHAT INTERFACE ---
    with st.expander("💬 Chat Assistant", expanded=True):
        user_query = st.text_input("Ask about your data:", placeholder="e.g. 'Show me a plot of price' or 'Find anomalies'")
        
        if st.button("Ask Agent") and user_query:
            if not st.session_state.dataset_id:
                st.warning("Please load data first!")
            else:
                query = user_query.lower()
                
                # Heuristic Intent Detection (Simulating an LLM Router)
                if any(x in query for x in ["chart", "plot", "graph", "visualize"]):
                    st.info("Agent: Creating visualization...")
                    
                    # Detect chart type
                    chart_type = "line"
                    if "bar" in query: chart_type = "bar"
                    elif "scatter" in query: chart_type = "scatter"
                    elif "hist" in query: chart_type = "histogram"
                    elif "box" in query: chart_type = "box"
                    
                    # Reuse Chart Logic
                    from mcp_server.tools.visualization import CreateChartTool
                    df = session.get_dataset(st.session_state.dataset_id)
                    cols = list(df.columns)
                    
                    # Simple heuristic for X/Y
                    x_col = "time" if "time" in cols else ("date" if "date" in cols else cols[0])
                    y_col = "close" if "close" in cols else ("priceUsd" if "priceUsd" in cols else cols[1])
                    
                    tool = CreateChartTool()
                    result = asyncio.run(tool.execute({
                        "dataset_id": st.session_state.dataset_id,
                        "chart_type": chart_type,
                        "x": x_col, 
                        "y": y_col,
                        "title": f"Agent Generated: {y_col} vs {x_col} ({chart_type})"
                    }, session=session))
                    
                    if result["success"]:
                        st.session_state.chart_json = result["data"]["plotly_json"]
                        st.success("Agent: I've updated the visualization!")
                        st.rerun()
                    else:
                        st.error(f"Agent: Failed to create chart. {result.get('error')}")

                elif any(x in query for x in ["anomaly", "outlier", "weird"]):
                    st.info("Agent: Checking for data anomalies...")
                    from mcp_server.tools.analysis import DetectAnomaliesTool
                    df = session.get_dataset(st.session_state.dataset_id)
                    cols = list(df.columns)
                    target_col = "close" if "close" in cols else ("priceUsd" if "priceUsd" in cols else cols[1])
                    
                    tool = DetectAnomaliesTool()
                    result = asyncio.run(tool.execute({
                        "dataset_id": st.session_state.dataset_id,
                        "columns": [target_col],
                        "method": "zscore"
                    }, session=session))
                    
                    if result["success"]:
                        st.session_state.anomalies = result["data"]["anomalies"]
                        st.success(f"Agent: Found {len(result['data']['anomalies'])} anomalies.")
                        st.json(result["data"]["anomalies"])
                    else:
                        st.error("Agent: Analysis failed.")

                elif "stats" in query or "describe" in query:
                    st.info("Agent: Generating statistics...")
                    from mcp_server.tools.analysis import DescribeDatasetTool
                    tool = DescribeDatasetTool()
                    result = asyncio.run(tool.execute({
                        "dataset_id": st.session_state.dataset_id
                    }, session=session))
                    
                    if result["success"]:
                        st.write(result["data"]["statistics"])
                    else:
                        st.error("Stats failed.")
                        
                else:
                    st.info("Agent: I can help you 'Visualize trends', 'Find anomalies', or 'Describe statistics'.")

    # --- MANUAL TOOLS ---
    with st.expander("🛠️ Manual Analysis Tools", expanded=False):
        if st.session_state.dataset_id:
            if st.button("Run Anomaly Detection"):
                # ... (Existing logic shifted here)
                from mcp_server.tools.analysis import DetectAnomaliesTool
                df = session.get_dataset(st.session_state.dataset_id)
                cols = list(df.columns)
                target_col = "close" if "close" in cols else ("priceUsd" if "priceUsd" in cols else cols[1])

                tool = DetectAnomaliesTool()
                
                async def analyze():
                    return await tool.execute({
                        "dataset_id": st.session_state.dataset_id,
                        "columns": [target_col],
                        "method": "zscore"
                    }, session=session)
                
                result = asyncio.run(analyze())
                if result and result["success"]:
                    anomalies = result["data"]["anomalies"]
                    st.session_state.anomalies = anomalies
                    st.info("Analysis Complete")
                    st.json(anomalies)
                else:
                    st.error("Analysis failed")

    st.subheader("4. Resources Exploration")
    if st.button("List Resources (MCP Resource)"):
        from mcp_server.resources import list_all_resources
        resources = list_all_resources()
        res_list = [{"uri": r.uri, "name": r.name} for r in resources]
        st.table(res_list)

    st.subheader("Agent Logs")
    for l in st.session_state.logs[-5:]:
        if l["level"] == "success":
            st.success(l["msg"])
        elif l["level"] == "error":
            st.error(l["msg"])
        else:
            st.info(l["msg"])

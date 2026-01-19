# === Ensure project root is in Python path ===
import sys
import time
from pathlib import Path

# Debug profiling
t0 = time.time()
# Force absolute path to avoid CWD confusion
PROFILE_LOG = Path(r"C:\Projects\Minerva\Minerva\startup_profile.log")

# Open with buffering=1 (line buffered) or flush immediately
with open(PROFILE_LOG, "w", encoding="utf-8") as f:
    f.write(f"[{0.0:.3f}s] Startup: Begin imports\n")
    f.flush()

def log_profile(msg):
    try:
        with open(PROFILE_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{time.time()-t0:.3f}s] {msg}\n")
    except Exception as e:
        print(f"LOG ERROR: {e}")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# === Debug & crash logging (optional) ===
import logging, faulthandler
faulthandler.enable()
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),              # Streamlit console
        logging.FileHandler("app_debug.log")  # Persistent log
    ]
)

# Tell Streamlit to emit DEBUG logs
import os
os.environ.setdefault("STREAMLIT_SERVER_ENABLECORS", "true")
log_profile("Importing streamlit...")
import streamlit as st

log_profile("Importing orchestration...")
from orchestration.orchestrate_workflow import run_workflow, orchestrate_workflow
log_profile("Importing chatbot...")
from chatbot.chatbot import chatbot_interface
log_profile("Importing pandas...")
import pandas as pd
import json
import hashlib
from feedback.ratings import store_rating
from feedback.role_corrections import store_role_corrections
log_profile("Importing ui modules...")
from ui.column_review import column_review
from orchestration.analysis_selector import select_analyzer
from storage.local_backend import load_datalake_dfs
from preprocessing.save_meta import (
    load_column_descriptions,
    load_column_roles,
)
log_profile("Importing metadata parser...")
from preprocessing.metadata_parser import infer_column_meta, merge_user_labels
log_profile("Importing llm_preprocessor...")
from preprocessing.llm_preprocessor import recommend_models_with_llm
from preprocessing.sanitize import scrub_df
log_profile("Importing visualizations...")
from ui.visualizations import (
    generate_bar_chart,
    generate_scatter_plot,
    generate_histogram,
    generate_heatmap,
    generate_pie_chart,
    generate_area_chart,
)
log_profile("Importing orchestrator core...")
from orchestration.orchestrator import orchestrate_dashboard
from orchestration.data_quality_scorer import summarize_for_display
from ui import redaction_banner
from ui.exploratory_tab import render_exploratory_tab
from ui.action_center import render_action_center
from ui.llm_settings import render_llm_settings, render_llm_settings_compact
from reports.report_generator import generate_report_bytes

log_profile("Imports complete!")


def _hash_df(df: pd.DataFrame) -> str:
    """Return a SHA1 hash for the contents of ``df``."""
    data = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha1(data).hexdigest()


def _available_identifiers(df: pd.DataFrame, dest_dir: str = "metadata") -> list[str]:
    """Return identifiers for saved metadata versions of ``df``."""
    h = _hash_df(df)
    path = Path(dest_dir)
    ids = []
    pattern = f"*_{h}_roles.json"
    for p in path.glob(pattern):
        stem = p.stem
        if stem.count("_") >= 2:
            ids.append(stem.split("_")[0])
    return sorted(set(ids))


def _analysis_suggestions(df: pd.DataFrame, meta) -> list[str]:
    """Return analysis suggestions based on LLM or column metadata."""
    try:
        # SKIP LLM on startup to prevent 30s hang
        # rec = recommend_models_with_llm(df)
        rec = "" 
    except Exception:
        rec = ""
    suggestions: list[str] = []
    if rec and "LLM unavailable" not in rec:
        suggestions = [s.strip("- ") for s in rec.splitlines() if s.strip()]
    if not suggestions:
        roles = {m.role for m in meta}
        if any(r in {"date", "time"} for r in roles):
            suggestions.append("Looks like a time-series – forecast sales?")
        if "categorical" in roles:
            suggestions.append("Try a classification model?")
        if "numeric" in roles:
            suggestions.append("Maybe run regression analysis?")
    return suggestions

# Mock dataset for demonstration
mock_data = pd.DataFrame(
    {
        "Date": pd.date_range(start="2022-01-01", periods=100, freq="ME"),
        "Sales": [i * 1.05 for i in range(100)],
    }
)


# Mock visualization functions
def generate_line_chart(data):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(data["Date"], data["Sales"], label="Sales Over Time")
    plt.title("Line Chart of Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    st.pyplot(plt)


visualizations = {
    "line_chart": generate_line_chart,
    "bar_chart": generate_bar_chart,
    "scatter_plot": generate_scatter_plot,
    "histogram": generate_histogram,
    "heatmap": generate_heatmap,
    "pie_chart": generate_pie_chart,
    "area_chart": generate_area_chart,
}


def build_dashboard(data: pd.DataFrame, result: dict, target_column: str) -> dict:
    """Run dashboard orchestration based on computation results."""
    model_output = {
        "predictions": result.get("model_info", {}).get("predictions", []),
        "confidence_score": result.get("model_info", {}).get("confidence_score", 80),
    }
    model_type = (
        "time-series" if any("date" in col.lower() for col in data.columns) else "generic"
    )
    try:
        return orchestrate_dashboard(
            data,
            model_output,
            model_type,
            target_column,
            {"general": {}},
            "",
        )
    except RuntimeError:
        return {}


# Mock model functions for scenarios
def generate_scenario(params):
    """
    Simulates a scenario based on the provided parameters.

    Args:
        params (dict): Parameters for the scenario.

    Returns:
        str: A summary of the simulated scenario.
    """
    return f"Simulated scenario with adjustments: {params}"
# ────────────────────────────────────────────────────────────────────────

# Main dashboard logic

st.title("Business Insights Dashboard")
st.write("Welcome to the interactive dashboard with chatbot support!")
redaction_banner()

# Display past run ids if available
history_file = Path(os.getenv("LOCAL_DATA_DIR", "local_data")) / "run_history.json"
run_history = []
if os.getenv("DYNAMO_SESSIONS_TABLE"):
    from storage import session_db

    for sess in session_db.list_sessions():
        meta = session_db.get_run_by_id(sess["run_id"])
        if meta:
            run_history.append(meta)
else:
    try:
        run_history = json.loads(history_file.read_text())
    except Exception:
        run_history = []

if run_history:
    st.sidebar.subheader("Run History")
    for idx, entry in enumerate(run_history):
        rid = entry.get("run_id")
        if not rid:
            continue
        st.sidebar.write(rid)
        if st.sidebar.button("Rerun", key=f"rerun_{rid}_{idx}"):
            res = orchestrate_workflow(rid, rid, load_datalake_dfs())
            st.session_state["result"] = res
            st.rerun()

# Sidebar for data upload and selection

# LLM Model Status
try:
    from llm_manager.registry import get_registry
    _llm_registry = get_registry()
    _active_model = _llm_registry.get_active_model()
    if _active_model:
        st.sidebar.success(f"🤖 {_active_model.name}")
    else:
        st.sidebar.warning("🤖 No LLM selected")
except Exception:
    st.sidebar.info("🤖 LLM: Setup needed")

st.sidebar.divider()

st.sidebar.divider()

# Secure Key Storage
try:
    from utils.key_storage import KeyStorage
    key_store = KeyStorage()
    
    # Auto-load keys into env
    for svc in key_store.list_services():
        key = key_store.get_key(svc)
        if key:
            # Map friendly service name to env var if possible
            # For now, we rely on the manual mapping or presets
            # Ideally, detailed mapping would be good. 
            # Simple hack: 
            if svc == "fred": os.environ["FRED_API_KEY"] = key
            if svc == "alphavantage": os.environ["ALPHAVANTAGE_API_KEY"] = key
            
            # OpenAI/Anthropic (if managed here later)
            if svc == "openai": os.environ["OPENAI_API_KEY"] = key
    
except Exception as e:
    st.sidebar.error(f"Key Store Error: {e}")
    key_store = None

st.sidebar.divider()

# === Data Manager ===
st.sidebar.divider()
st.sidebar.subheader("🗄️ Data Manager")

# Initialize datasets store
if "datasets" not in st.session_state:
    st.session_state["datasets"] = {}
if "primary_dataset_id" not in st.session_state:
    st.session_state["primary_dataset_id"] = None

# Show active datasets
if st.session_state["datasets"]:
    active_ds = st.sidebar.radio(
        "Primary Dataset", 
        options=list(st.session_state["datasets"].keys()),
        index=0 if not st.session_state["primary_dataset_id"] else list(st.session_state["datasets"].keys()).index(st.session_state["primary_dataset_id"])
    )
    st.session_state["primary_dataset_id"] = active_ds
    data = st.session_state["datasets"][active_ds]
    
    # Simple list of others (secondary)
    others = [k for k in st.session_state["datasets"].keys() if k != active_ds]
    if others:
        st.sidebar.caption(f"Secondary: {', '.join(others)}")
else:
    data = None
    st.sidebar.info("No datasets loaded")


with st.sidebar.expander("📥 Add Dataset", expanded=data is None):
    add_mode = st.radio("Source", ["Upload CSV", "Connect API", "Sample Data"], horizontal=True)

    if add_mode == "Upload CSV":
        uploaded_files = st.file_uploader("Upload CSV", type="csv", accept_multiple_files=True)
        if uploaded_files:
            for f in uploaded_files:
                name = f.name
                if name not in st.session_state["datasets"]:
                    with st.spinner(f"Loading {name}..."):
                        from preprocessing.data_cleaning import standardize_dataframe
                        df = pd.read_csv(f)
                        df = standardize_dataframe(df)
                        st.session_state["datasets"][name] = df
                        if not st.session_state["primary_dataset_id"]:
                            st.session_state["primary_dataset_id"] = name
            st.rerun()

    elif add_mode == "Connect API":
        from config.api_presets import API_PRESETS
        
        # Preset selection
        presets = list(API_PRESETS.keys())
        presets.insert(0, "Custom URL")
        
        preset_choice = st.selectbox("Connector", presets, format_func=lambda x: API_PRESETS[x]["name"] if x in API_PRESETS else x)
        
        api_url = ""
        endpoint = ""
        params_str = "{}"
        enable_auth = False
        headers_str = "{}"
        
        if preset_choice != "Custom URL":
            p_config = API_PRESETS[preset_choice]
            st.caption(p_config["description"])
            api_url = p_config["base_url"]
            
            # Auth UI
            auth_meta = p_config.get("auth", {})
            if auth_meta.get("type") == "api_key":
                 env_var = auth_meta.get("env_var")
                 signup_url = auth_meta.get("signup_url")
                 
                 has_key = env_var and os.getenv(env_var)
                 
                 if env_var and not has_key:
                     st.warning(f"⚠️ {preset_choice} requires an API Key")
                     if signup_url:
                         st.markdown(f"[🔑 **Get Free API Key**]({signup_url})")
                     
                     col_k1, col_k2 = st.columns([3, 1])
                     with col_k1:
                        manual_key = st.text_input("Enter API Key", type="password")
                     with col_k2:
                        save_key = st.checkbox("Save")
                     
                     if manual_key and save_key and key_store:
                         # normalize service name
                         svc_name = preset_choice.lower()
                         if "fred" in svc_name: svc_name = "fred"
                         if "alpha" in svc_name: svc_name = "alphavantage"
                         
                         key_store.save_key(svc_name, manual_key)
                         st.success("Saved!")
                         time.sleep(1) # feedback
                         st.rerun()
                         
                 elif env_var:
                     st.success(f"✅ Key found in environment")
                     # Option to clear/update?
                     if st.checkbox("Update Key", key=f"upd_{preset_choice}"):
                         new_key = st.text_input("New Key", type="password")
                         if st.button("Save New Key"):
                             svc_name = preset_choice.lower()
                             if "fred" in svc_name: svc_name = "fred"
                             if "alpha" in svc_name: svc_name = "alphavantage"
                             key_store.save_key(svc_name, new_key)
                             st.rerun()
            
            # Endpoints
            if "endpoints" in p_config:
                 ep_choices = list(p_config["endpoints"].keys())
                 ep_selection = st.selectbox("Indicator", ep_choices)
                 ep_meta = p_config["endpoints"][ep_selection]
                 endpoint = ep_meta["path"]
                 st.caption(ep_meta.get("description", ""))
                 if "params" in ep_meta:
                     import json
                     params_str = json.dumps(ep_meta["params"])
            else:
                 endpoint = st.text_input("Endpoint")

        else:
            api_url = st.text_input("Base URL")
            endpoint = st.text_input("Endpoint")
            enable_auth = st.checkbox("Auth")
            if enable_auth:
                 token = st.text_input("Token", type="password")
        
        dataset_name = st.text_input("Save as (Name)", value=f"{preset_choice}_data" if preset_choice != "Custom URL" else "api_data")

        if st.button("Fetch"):
            import asyncio
            from mcp_server.tools.connectors import ConnectAPITool, FetchAPIDataTool
            from mcp_server.session import MCPSession
            
            temp_sess = MCPSession("temp_dash")
            
            status_cont = st.empty()
            status_cont.info("Connecting...")
            
            c = ConnectAPITool()
            c_args = {}
            if preset_choice != "Custom URL":
                c_args["preset"] = preset_choice
                # If manual key provided for preset
                if 'manual_key' in locals() and manual_key:
                     c_args["auth"] = {
                         "type": "api_key", 
                         "token": manual_key, 
                         "header_name": API_PRESETS[preset_choice]["auth"].get("param_name", "X-API-Key")
                     }
            else:
                c_args["url"] = api_url
                if enable_auth and 'token' in locals():
                     c_args["auth"] = {"type": "bearer", "token": token}

            try:
                # Run sync wrapper for async
               def run_async(coro):
                   loop = asyncio.new_event_loop()
                   asyncio.set_event_loop(loop)
                   return loop.run_until_complete(coro)

               # Connect
               res = run_async(c.execute(c_args, session=temp_sess))
               if not res["success"]:
                   status_cont.error(f"Connect failed: {res.get('error')}")
               else:
                   conn_id = res["data"]["connection_id"]
                   status_cont.info(f"Connected! Fetching {endpoint}...")
                   
                   # Fetch
                   f = FetchAPIDataTool()
                   p_dict = json.loads(params_str)
                   
                   # Use the human-readable label as the context for why/what this data is
                   context_label = f"Indicator: {endpoint} ({dataset_name})"
                   # If we have the full indicator object from presets, use its name 
                   
                   f_res = run_async(f.execute({
                       "connection_id": conn_id,
                       "endpoint": endpoint,
                       "params": p_dict,
                       "save_as": dataset_name,
                       "context": context_label
                   }, session=temp_sess))
                   
                   if f_res["success"]:
                       new_df = temp_sess.get_dataset(dataset_name)
                       if new_df is not None:
                           st.session_state["datasets"][dataset_name] = new_df
                           if not st.session_state["primary_dataset_id"]:
                               st.session_state["primary_dataset_id"] = dataset_name
                           status_cont.success("Success!")
                           time.sleep(1)
                           st.rerun()
                   else:
                       status_cont.error(f"Fetch failed: {f_res.get('error')}")

            except Exception as e:
                status_cont.error(f"Error: {e}")

    elif add_mode == "Sample Data":
        from preprocessing.data_cleaning import standardize_dataframe
        if st.button("Load Bitcoin Sample"):
             try:
                 df = pd.read_csv("data/bitcoin_history.csv")
                 df = standardize_dataframe(df)
                 st.session_state["datasets"]["Bitcoin"] = df
                 if not st.session_state["primary_dataset_id"]:
                      st.session_state["primary_dataset_id"] = "Bitcoin"
                 st.rerun()
             except:
                 st.error("Sample file missing")
                 # Fallback
                 st.session_state["datasets"]["Mock Sales"] = standardize_dataframe(mock_data)
                 st.session_state["primary_dataset_id"] = "Mock Sales"
                 st.rerun()

if data is None and not st.session_state["datasets"]:
    # First run fallback
    st.session_state["datasets"]["Mock Sales"] = mock_data
    st.session_state["primary_dataset_id"] = "Mock Sales"
    data = mock_data

# Retrieve any stored metadata
ids = _available_identifiers(data)
sel = None
if ids:
    choice = st.sidebar.selectbox("Metadata version", ["Most recent"] + ids)
    if choice != "Most recent":
        sel = choice

roles = load_column_roles(data, identifier=sel) or {}
descriptions = load_column_descriptions(data, identifier=sel) or {}
meta = infer_column_meta(data, descriptions)
if roles:
    meta = merge_user_labels(meta, roles)
st.session_state["column_meta"] = meta
st.session_state["column_descriptions"] = descriptions

preview = scrub_df(data.head(50))
if descriptions:
    preview["Description"] = preview.index.map(lambda c: descriptions.get(c, ""))

# Main content tabs
main_tab, explore_tab, action_tab, chat_tab, llm_tab = st.tabs([
    "📊 Data Preview", "🔍 Explore", "🎯 Actions", "💬 Chat", "🤖 LLM Settings"
])

with main_tab:
    st.dataframe(preview)

with explore_tab:
    try:
        # Get metadata for the current dataset
        dataset_id = st.session_state.get("primary_dataset_id")
        current_meta = st.session_state.get("dataset_metadata", {}).get(dataset_id, {})
        render_exploratory_tab(data, meta=current_meta)
    except Exception as e:
        st.error(f"⚠️ Exploratory analysis encountered an issue: {str(e)}")
        st.info("Try refreshing the page or uploading a different dataset.")

with action_tab:
    try:
        render_action_center(data, meta)
    except Exception as e:
        st.error(f"⚠️ Action Center encountered an issue: {str(e)}")
        st.info("Try refreshing the page or check your data format.")

with llm_tab:
    render_llm_settings()

# Suggest analyses based on data
if "analysis_suggestions" not in st.session_state:
    st.session_state["analysis_suggestions"] = _analysis_suggestions(data, meta)

if st.session_state["analysis_suggestions"]:
    st.sidebar.subheader("Suggested Analyses")
    for idx, sugg in enumerate(st.session_state["analysis_suggestions"]):
        if st.sidebar.button(sugg, key=f"sugg_{idx}"):
            st.session_state["suggestion"] = sugg
            st.rerun()

models = {
    "scenario_generator": generate_scenario,
    "classification": lambda p: select_analyzer(data).run(data),
    "regression": lambda p: select_analyzer(data).run(data),
    "clustering": lambda p: select_analyzer(data).run(data),
    "anomaly_detection": lambda p: select_analyzer(data).run(data),
    "forecasting": lambda p: select_analyzer(data).run(data),
}

# Banner for low accuracy prompting column review
if st.session_state.get("needs_role_review") and not st.session_state.get("show_column_review"):
    st.warning("Accuracy is low. Clarify column meanings to improve?")
    if st.button("Review Columns"):
        st.session_state["show_column_review"] = True

if st.session_state.get("show_column_review"):
    new_roles = column_review(data, st.session_state.get("column_meta", []))
    if new_roles:
        st.session_state["user_roles"] = new_roles
        store_role_corrections(data, new_roles)

# Put chatbot in the chat tab
with chat_tab:
    st.write("Chat with your data below.")
    chatbot_interface(data, visualizations, models, prefill=st.session_state.pop("suggestion", None))

# Rating widget
st.subheader("Rate the results")
rating = st.slider("Select a score", 1, 5, 3, key="rating_slider")
if st.button("Submit Rating", key="submit_rating"):
    store_rating(rating)
    st.success("Thank you for your feedback!")

# Display analysis results if available
result = st.session_state.get("result")
if result:
    target_col = result.get("model_info", {}).get("target") or (
        data.columns[-1] if len(data.columns) else ""
    )
    
    # Display Data Quality Report (if diagnostics available)
    diagnostics = result.get("diagnostics", {})
    if diagnostics:
        quality_summary = summarize_for_display(diagnostics)
        with st.expander(f"{quality_summary['status_emoji']} Data Quality Report", expanded=quality_summary['status'] != 'good'):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Quality Score", f"{quality_summary['score']}%")
            with col2:
                st.metric("Data Completeness", f"{quality_summary['completeness_pct']}%")
            with col3:
                st.metric("Complete Rows", f"{quality_summary['rows_complete_pct']}%")
            
            if quality_summary['warnings']:
                st.warning("**Warnings:**")
                for warning in quality_summary['warnings']:
                    st.write(f"- {warning}")
            
            if quality_summary['high_risk_columns']:
                st.error(f"**High-risk columns:** {', '.join(quality_summary['high_risk_columns'])}")
            
            if quality_summary['missing_columns']:
                with st.expander("Columns with missing data"):
                    st.write(", ".join(quality_summary['missing_columns']))
            
            if quality_summary['proceed_with_caution']:
                st.info("💡 Consider reviewing column roles or providing additional data to improve results.")
    
    dash = build_dashboard(data, result, target_col)
    if dash.get("chart_result") is not None:
        st.write(dash["chart_result"])
    st.header("Analysis Results")
    decision = result.get("modeling_decision")
    if decision and not decision.get("modeling_required", True):
        st.info(
            f"\u2139\ufe0f No predictive model was run. Reason: {decision.get('reasoning', '')}"
        )
    elif decision and decision.get("modeling_required"):
        st.info(
            f"Predictive modeling applied: {decision.get('modeling_type', '')}. Reason: {decision.get('reasoning', '')}"
        )
    if result.get("modeling_failed"):
        st.error("Modeling failed. Showing descriptive results only.")
        if result.get("failure_reason"):
            st.info(result["failure_reason"])
    atype = result.get("analysis_type")

    if atype == "regression":
        st.line_chart(result.get("predictions"))
    elif atype == "classification":
        st.json(result.get("report"))
    elif atype == "forecasting":
        st.line_chart(result.get("forecast"))
    elif atype == "clustering":
        st.write("Cluster Centers")
        st.write(result.get("centers"))
        if data.shape[1] >= 2:
            df_plot = data.iloc[:, :2].copy(deep=False)
            df_plot["label"] = result.get("labels")
            st.scatter_chart(df_plot, x="label", y=df_plot.columns[0])
    elif atype == "descriptive":
        st.dataframe(result.get("stats"))

    # Anomalies are shown once
    st.sidebar.write("Anomalies", result.get("anomalies"))

    if result.get("summary"):
        st.subheader("Business Summary")
        st.write(result["summary"])


    rec_models = result.get("recommended_models")
    if rec_models:
        with st.sidebar.expander("Recommended Models"):
            st.json(rec_models)

    merge_report = result.get("model_info", {}).get("merge_report")
    if merge_report:
        with st.expander("Merge Report"):
            st.download_button(
                "Download merge_report.json",
                json.dumps(merge_report, indent=2),
                file_name="merge_report.json",
                mime="application/json",
            )
            st.json(merge_report)


    # Premium-only feature
    dl_button = getattr(st.sidebar, "download_button", None)
    if callable(dl_button):
        dl_button(
            "Download One Pager (Premium)",
            result.get("summary", ""),
            file_name="one_pager.txt",
            disabled=not st.session_state.get("is_paid", False),
        )

    model_path = Path("best_model.pkl")
    if model_path.exists():
        with st.sidebar:
            st.download_button(
                "Download best_model.pkl",
                model_path.read_bytes(),
                file_name="best_model.pkl",
                mime="application/octet-stream",
            )
    
    # HTML Report Export
    try:
        report_bytes = generate_report_bytes(data, result, title="Minerva Analysis Report")
        st.sidebar.download_button(
            "📄 Export Full Report (HTML)",
            report_bytes,
            file_name="analysis_report.html",
            mime="text/html",
        )
    except Exception as e:
        st.sidebar.caption(f"Report generation unavailable")
            
    explanations = result.get("model_info", {}).get("explanations")
    if explanations:
        fi = explanations.get("feature_importances") or explanations.get("coefficients")
        if fi is not None:
            with st.expander("Feature Importances"):
                st.bar_chart(pd.Series(fi))
        if explanations.get("shap_values") is not None:
            with st.expander("SHAP Values"):
                st.download_button(
                    "Download shap_values.json",
                    json.dumps(explanations["shap_values"], indent=2),
                    file_name="shap_values.json",
                    mime="application/json",
                )



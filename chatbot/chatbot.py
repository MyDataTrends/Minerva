import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # avoid OpenMP double-load crash
from dotenv import load_dotenv
import streamlit as st
from orchestrator import orchestrate_dashboard
from .decision import decide_action

def chatbot_interface(data, visualizations, models, target=None, prefill=None):
    """
    Streamlit chatbot interface for handling user queries.

    Args:
        data (pd.DataFrame): Dataset for analysis.
        visualizations (dict): Available visualization functions.
        models (dict): Available model configurations.
        prefill (str | None): Optional query to run automatically.

    Returns:
        None: Displays chatbot UI and responses.
    """
    st.sidebar.title("Chatbot Assistant")
    user_query = st.sidebar.text_input("Ask a question:", value=prefill or "")

    if user_query:
        # Determine whether the chat corresponds to a model request,
        # a visualization or if we should run the automatic analysis.
        try:
            action, _ = decide_action(user_query)
        except RuntimeError:
            st.sidebar.error("LLM shard unavailable — check model config")
            return
        if action == "analysis":
            st.sidebar.write("Running automatic analysis based on your question...")

        target_variable = target or data.select_dtypes(include="number").columns[0]
        try:
            result = orchestrate_dashboard(
                data,
                {},
                "generic",
                target_variable,
                {"general": {}},
                user_query,
                visualizations=visualizations,
                models=models,
            )
        except RuntimeError:
            st.sidebar.error("LLM shard unavailable — check model config")
            return

        if result.get("chart_result") is not None:
            st.write(result["chart_result"])
        if result.get("model_result") is not None:
            st.session_state["result"] = result["model_result"]
            st.write(result["model_result"])
        if not result.get("chart_result") and not result.get("model_result"):
            st.sidebar.write("Sorry, I didn't understand that. Try rephrasing!")

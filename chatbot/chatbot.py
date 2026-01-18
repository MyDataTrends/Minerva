import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # avoid OpenMP double-load crash
from dotenv import load_dotenv
import streamlit as st
from orchestration.orchestrator import orchestrate_dashboard
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
    st.subheader("💬 Chat with Your Data")
    
    # Chat input in main area
    user_query = st.text_input("Ask a question about your data:", value=prefill or "", 
                                placeholder="e.g., 'What are the top selling products?' or 'Show me sales trends'")

    if user_query:
        # Determine whether the chat corresponds to a model request,
        # a visualization or if we should run the automatic analysis.
        try:
            action, _ = decide_action(user_query)
        except RuntimeError:
            st.error("⚠️ LLM unavailable — configure one in the LLM Settings tab")
            return
        if action == "analysis":
            st.info("🔄 Running automatic analysis based on your question...")

        target_variable = target or data.select_dtypes(include="number").columns[0]
        with st.spinner("Analyzing..."):
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
                st.error("⚠️ LLM unavailable — configure one in the LLM Settings tab")
                return

        if result.get("chart_result") is not None:
            st.write(result["chart_result"])
        if result.get("model_result") is not None:
            st.session_state["result"] = result["model_result"]
            st.write(result["model_result"])
        if not result.get("chart_result") and not result.get("model_result"):
            st.warning("Sorry, I didn't understand that. Try rephrasing your question!")
    else:
        # Generate data-aware example questions
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        st.caption("💡 Try asking:")
        
        if numeric_cols and categorical_cols:
            st.caption(f"• 'What's the average {numeric_cols[0]} by {categorical_cols[0]}?'")
        if len(numeric_cols) >= 2:
            st.caption(f"• 'Show me how {numeric_cols[0]} relates to {numeric_cols[1]}'")
        if categorical_cols:
            st.caption(f"• 'Which {categorical_cols[0]} has the highest values?'")
        if numeric_cols:
            st.caption(f"• 'Show me a chart of {numeric_cols[0]}'")
        
        st.caption(f"\n📊 Your data has {len(data)} rows and columns: {', '.join(data.columns[:5])}{'...' if len(data.columns) > 5 else ''}")

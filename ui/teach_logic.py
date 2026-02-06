"""
Teaching Mode - Explicitly teach Minerva new skills.
Allows users to add examples to the Vector Store for RAG retrieval.
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import traceback

# Ensure project root is in path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import learning components
try:
    from learning.vector_store import VectorStore
    from learning.embeddings import EmbeddingModel
    LEARNING_AVAILABLE = True
except ImportError as e:
    LEARNING_AVAILABLE = False
    IMPORT_ERROR = str(e)

def init_components():
    """Initialize learning system components."""
    if not LEARNING_AVAILABLE:
        return None, None
    try:
        vs = VectorStore()
        emb = EmbeddingModel()
        return vs, emb
    except Exception as e:
        st.error(f"Failed to initialize components: {e}")
        return None, None

def render_teaching_mode():
    st.subheader("🧠 Teach Minerva")
    st.markdown("""
    Add new examples to Minerva's long-term memory. 
    These examples will be retrieved when similar questions are asked in the future.
    """)

    if not LEARNING_AVAILABLE:
        st.error(f"Learning system dependencies not found. Error: {IMPORT_ERROR}")
        st.info("Please install 'fastembed' to use this feature.")
        return

    # Tabs for different teaching methods
    tab1, tab2, tab3 = st.tabs(["📝 Manual Entry", "🎨 Style Rules", "🔍 View Memory"])

    vs, emb = init_components()
    if not vs or not emb:
        return

    with tab1:
        st.subheader("Add New Skill")
        
        with st.form("teaching_form"):
            intent = st.text_input(
                "User Question / Intent",
                placeholder="e.g., Calculate a 7-day rolling average"
            )
            
            explanation = st.text_area(
                "Logic Explanation (Optional)",
                placeholder="Briefly explain the logic used...",
                height=60
            )
            
            code = st.text_area(
                "Python Code (Pandas/Plotly)",
                placeholder="result = df['sales'].rolling(7).mean()",
                height=200,
                help="Write code that uses 'df' and stores output in 'result' or 'fig'"
            )
            
            source = st.selectbox("Source", ["manual_teaching", "kaggle_import", "reference_doc"])
            
            submitted = st.form_submit_button("💾 Save to Memory")
            
            if submitted and intent and code:
                try:
                    # 1. Validate syntax
                    compile(code, "<string>", "exec")
                    
                    # 2. Generate embedding
                    with st.spinner("Embedding intent..."):
                        vector = emb.embed_query(intent)
                    
                    # 3. Save to store
                    vs.add_example(
                        intent=intent,
                        code=code,
                        embedding=vector,
                        explanation=explanation,
                        source=source,
                        metadata={"user_taught": True, "created_at": str(pd.Timestamp.now())}
                    )
                    
                    st.success("✅ Example saved! Minerva will now use this pattern.")
                    
                except SyntaxError as e:
                    st.error(f"❌ Invalid Python code: {e}")
                except Exception as e:
                    st.error(f"❌ Error saving example: {e}")

    with tab2:
        st.subheader("🎨 Style & Formatting Rules")
        st.markdown("Teach Minerva how to format output (e.g., fonts, colors, huge numbers).")
        
        with st.form("style_form"):
            rule_name = st.text_input("Rule Name", placeholder="e.g. Corporate Color Palette")
            rule_content = st.text_area("Rule Description", placeholder="Use #0055AA for primary headers. Dates should be YYYY-MM-DD.")
            
            style_submitted = st.form_submit_button("Save Style Rule")
            
            if style_submitted and rule_name and rule_content:
                try:
                    # Embed the rule content so we can retrieve it based on relevance if needed
                    # For now, we might just load ALL style rules, but embedding allows scaling.
                    with st.spinner("Saving rule..."):
                        vector = emb.embed_query(rule_name + " " + rule_content)
                        
                        vs.add_example(
                            intent=rule_name,
                            code=rule_content, # Storing rule text in 'code' column
                            embedding=vector,
                            explanation="Style Rule",
                            source="style_rule",
                            metadata={"type": "style", "created_at": str(pd.Timestamp.now())}
                        )
                    st.success("✅ Style Rule Saved!")
                except Exception as e:
                    st.error(f"Error saving rule: {e}")

    with tab3:
        st.subheader("Existing Memories")
        stats = vs.get_stats()
        st.write("Current Knowledge Base Stats:", stats)
        
        search_query = st.text_input("Search Memory:", placeholder="Type to search stored examples...")
        
        if search_query:
            vector = emb.embed_query(search_query)
            results = vs.search(vector, limit=5)
            
            for res in results:
                with st.expander(f"{res['intent']} (Score: {res['score']:.2f})"):
                    st.code(res['code'], language="python")
                    if res.get('explanation'):
                        st.info(res['explanation'])
                    st.caption(f"Source: {res['source']}")

"""
Assay Ops Center - Standalone Administrative Interface.
"""
import streamlit as st
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set Page Config (Distinct from Main App)
st.set_page_config(
    page_title="Assay Ops Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Admin" feel
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #ff4b4b !important; 
    }
    .stButton>button {
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

from ui.agent_control import render_agent_control

def main():
    with st.sidebar:
        st.title("🛡️ Ops Center")
        st.info("System Administration & Agent Orchestration")
        
        st.divider()
        st.caption(f"Assay Core v2.0")
        st.caption(f"Python {sys.version.split(' ')[0]}")
    
    # Main Content
    render_agent_control()

if __name__ == "__main__":
    main()

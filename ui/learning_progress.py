"""
Learning Progress UI - Gamification widget for LLM learning.

Shows:
- Progress bar toward next milestone
- Current milestone badge
- Example count
- "Your LLM is learning!" messaging
"""

import streamlit as st
from typing import Dict, Any


def render_learning_progress(compact: bool = False):
    """
    Render the learning progress widget.
    
    Args:
        compact: If True, render minimal version for sidebar
    """
    try:
        from llm_learning.interaction_logger import get_interaction_logger
        logger = get_interaction_logger()
        progress = logger.get_learning_progress()
    except Exception as e:
        # Module not available or error
        if not compact:
            st.caption("Learning progress unavailable")
        return
    
    quality_examples = progress["quality_examples"]
    progress_pct = progress["progress_pct"]
    current = progress["current_milestone"]
    next_milestone = progress["next_milestone"]
    
    if compact:
        # Sidebar compact view
        if current:
            st.caption(f"{current['name']} • {quality_examples} examples")
        else:
            st.caption(f"🌱 {quality_examples} examples collected")
        st.progress(progress_pct / 100)
        return
    
    # Full view
    st.subheader("🧠 LLM Learning Progress")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Progress bar
        if next_milestone:
            st.progress(
                progress_pct / 100,
                text=f"{quality_examples} / {next_milestone['count']} examples toward {next_milestone['name']}"
            )
        else:
            st.progress(1.0, text="Maximum learning achieved! 🏆")
        
        # Messaging
        if quality_examples == 0:
            st.info("💡 Start chatting to help your LLM learn! Rate responses to improve accuracy.")
        elif quality_examples < 10:
            st.info(f"🌱 Getting started! {10 - quality_examples} more high-quality examples to first milestone.")
        elif next_milestone:
            remaining = next_milestone['count'] - quality_examples
            st.success(f"✨ Great progress! {remaining} more examples to unlock '{next_milestone['name']}'")
        else:
            st.success("⭐ Your LLM has achieved legendary status! Maximum learning unlocked.")
    
    with col2:
        if current:
            st.metric("Current Level", current['name'].split()[0])  # Just the emoji
        else:
            st.metric("Current Level", "🌱")
        st.metric("Examples", quality_examples)


def render_learning_sidebar():
    """Render compact learning progress in sidebar."""
    with st.sidebar:
        st.divider()
        render_learning_progress(compact=True)


def get_milestone_message(quality_examples: int) -> str:
    """Get an encouraging message based on example count."""
    if quality_examples >= 500:
        return "Your LLM is highly tuned! 🏆"
    elif quality_examples >= 100:
        return "Your LLM is getting smarter! 🚀"
    elif quality_examples >= 50:
        return "Building knowledge... 📚"
    elif quality_examples >= 10:
        return "Learning in progress... 🌱"
    else:
        return "Rate responses to help your LLM learn!"

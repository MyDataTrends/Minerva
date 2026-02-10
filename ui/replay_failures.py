"""
Replay Failures UI Component for Minerva.

Provides a "Replay last failure" button for debugging ephemeral code.
Shows failed executions from the ArtifactStore and allows replaying them.

Usage:
    from ui.replay_failures import render_replay_panel
    
    render_replay_panel()
"""
import streamlit as st
import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


def render_replay_panel():
    """
    Render the failure replay panel in the UI.
    
    Shows:
    - Recent failed executions
    - Failure details (query, error, steps)
    - "Replay" button to re-execute with fresh context
    """
    st.subheader("🔄 Replay Failed Executions")
    
    try:
        from orchestration import get_artifact_store, get_planner
        
        store = get_artifact_store()
        failed_runs = store.list_recent(limit=10, failed_only=True)
        
        if not failed_runs:
            st.info("✅ No failed executions to replay. Everything is working!")
            return
        
        st.caption(f"Found {len(failed_runs)} failed execution(s)")
        
        # Show each failed run
        for i, run_info in enumerate(failed_runs):
            run_id = run_info.get("file", "").replace(".json", "")
            if not run_id:
                continue
            
            # Load full artifact
            artifact = store.load(run_id)
            if not artifact:
                continue
            
            with st.expander(
                f"❌ {artifact.query[:50]}{'...' if len(artifact.query) > 50 else ''}",
                expanded=(i == 0),  # Expand first one
            ):
                _render_failure_details(artifact, store, run_id)
    
    except ImportError as e:
        st.warning(f"Orchestration module not available: {e}")
    except Exception as e:
        st.error(f"Error loading failure data: {e}")
        logger.exception("Failed to load failure data")


def _render_failure_details(artifact, store, run_id: str):
    """Render details for a single failed execution."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"**Query:** {artifact.query}")
        st.markdown(f"**Intent:** `{artifact.intent}`")
        st.markdown(f"**Time:** {_format_timestamp(artifact.timestamp)}")
    
    with col2:
        # Replay button
        if st.button("🔄 Replay", key=f"replay_{run_id}"):
            _handle_replay(artifact, store, run_id)
    
    # Error details
    if artifact.error:
        st.error(f"**Error:** {artifact.error}")
    
    # Step details
    if artifact.step_results:
        st.markdown("**Execution Steps:**")
        for step in artifact.step_results:
            status_icon = "✅" if step.get("status") == "success" else "❌"
            step_error = step.get("error", "")
            
            step_text = f"{status_icon} **{step.get('action', 'Unknown')}** → `{step.get('tool', 'Unknown')}`"
            if step_error:
                step_text += f"\n  - Error: {step_error}"
            
            st.markdown(step_text)
    
    # Data snapshot
    if artifact.data_snapshot:
        with st.expander("📊 Data Snapshot"):
            snapshot = artifact.data_snapshot
            st.write(f"**Shape:** {snapshot.get('shape', 'Unknown')}")
            st.write(f"**Columns:** {', '.join(snapshot.get('columns', [])[:10])}")
            
            sample = snapshot.get("sample_data")
            if sample:
                st.write("**Sample:**")
                st.json(sample[:3])
    
    # JSON export
    with st.expander("📄 Raw Artifact (JSON)"):
        st.json(artifact.to_dict())


def _handle_replay(artifact, store, run_id: str):
    """Handle the replay button click."""
    st.info("🔄 Replaying execution...")
    
    try:
        from orchestration import get_planner
        import pandas as pd
        
        # Check if we have the original data in session state
        df = st.session_state.get("df")
        
        if df is None:
            st.warning("⚠️ No data loaded. Please load a dataset first to replay.")
            return
        
        # Replay the execution
        context = {"df": df}
        result = store.replay(run_id, context=context)
        
        if result and result.success:
            st.success(f"✅ Replay succeeded! Steps: {result.steps_completed}/{result.total_steps}")
            
            # Show output
            if result.output:
                st.write("**Output:**")
                if isinstance(result.output, dict):
                    st.json(result.output)
                elif isinstance(result.output, pd.DataFrame):
                    st.dataframe(result.output.head(10))
                else:
                    st.write(result.output)
        else:
            st.error(f"❌ Replay failed: {result.error if result else 'Unknown error'}")
    
    except Exception as e:
        st.error(f"Replay error: {e}")
        logger.exception("Failed to replay execution")


def _format_timestamp(ts: str) -> str:
    """Format ISO timestamp to readable string."""
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts


# =============================================================================
# Quick Replay Widget (for embedding in other views)
# =============================================================================

def render_quick_replay_button():
    """
    Render a simple "Replay Last Failure" button.
    
    Use this in other views to provide quick access to replay.
    """
    try:
        from orchestration import get_artifact_store
        
        store = get_artifact_store()
        failed = store.list_recent(limit=1, failed_only=True)
        
        if not failed:
            return  # No failures to show
        
        run_info = failed[0]
        run_id = run_info.get("file", "").replace(".json", "")
        query = run_info.get("query", "Unknown query")[:30]
        
        col1, col2 = st.columns([4, 1])
        with col1:
            st.caption(f"⚠️ Last failure: {query}...")
        with col2:
            if st.button("🔄 Replay", key="quick_replay"):
                artifact = store.load(run_id)
                if artifact:
                    _handle_replay(artifact, store, run_id)
    
    except ImportError:
        pass  # Orchestration not available
    except Exception as e:
        logger.debug(f"Quick replay button error: {e}")


# =============================================================================
# Failure Summary Stats
# =============================================================================

def get_failure_summary() -> Dict[str, Any]:
    """
    Get a summary of recent failures for dashboard display.
    
    Returns:
        Dict with failure_count, last_failure_time, common_errors
    """
    try:
        from orchestration import get_artifact_store
        
        store = get_artifact_store()
        failed = store.list_recent(limit=20, failed_only=True)
        
        if not failed:
            return {
                "failure_count": 0,
                "last_failure_time": None,
                "common_errors": [],
            }
        
        # Extract common error patterns
        errors = []
        for run in failed:
            artifact = store.load(run.get("file", "").replace(".json", ""))
            if artifact and artifact.error:
                errors.append(artifact.error[:50])
        
        # Count error occurrences
        error_counts = {}
        for err in errors:
            error_counts[err] = error_counts.get(err, 0) + 1
        
        common = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "failure_count": len(failed),
            "last_failure_time": failed[0].get("timestamp") if failed else None,
            "common_errors": [e[0] for e in common],
        }
    
    except Exception as e:
        logger.debug(f"Failed to get failure summary: {e}")
        return {"failure_count": 0, "last_failure_time": None, "common_errors": []}


def render_failure_badge():
    """
    Render a badge showing failure count (for sidebar/header).
    
    Shows nothing if no failures.
    """
    summary = get_failure_summary()
    count = summary.get("failure_count", 0)
    
    if count > 0:
        st.markdown(
            f'<span style="background-color: #ff4b4b; color: white; '
            f'padding: 2px 8px; border-radius: 10px; font-size: 0.8em;">'
            f'⚠️ {count} failed</span>',
            unsafe_allow_html=True,
        )

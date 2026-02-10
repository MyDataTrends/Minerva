"""
Developer Settings UI for Minerva.

Provides UI toggles for diagnostic mode settings that persist in session state.

Usage:
    from ui.dev_settings import render_dev_toggle
    
    # In sidebar:
    render_dev_toggle()
"""
import streamlit as st
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _get_setting(key: str, default: bool = False) -> bool:
    """Get a dev setting from session state."""
    return st.session_state.get(f"dev_{key}", default)


def _set_setting(key: str, value: bool):
    """Set a dev setting in session state."""
    st.session_state[f"dev_{key}"] = value


def is_dev_mode() -> bool:
    """Check if dev mode is enabled (via UI or environment)."""
    from config.feature_flags import DEV_MODE
    return DEV_MODE or _get_setting("mode")


def is_loud_failures() -> bool:
    """Check if loud failures is enabled."""
    from config.feature_flags import LOUD_FAILURES
    return LOUD_FAILURES or _get_setting("loud_failures")


def is_verbose_logging() -> bool:
    """Check if verbose logging is enabled."""
    from config.feature_flags import VERBOSE_LOGGING
    return VERBOSE_LOGGING or _get_setting("verbose_logging")


def render_dev_toggle():
    """
    Render a simple dev mode toggle in the sidebar.
    
    Shows a single toggle that enables all diagnostic features.
    """
    if "dev_mode" not in st.session_state:
        # Initialize from environment
        from config.feature_flags import DEV_MODE
        st.session_state["dev_mode"] = DEV_MODE
    
    dev_enabled = st.sidebar.toggle(
        "🛠️ Dev Mode",
        value=st.session_state.get("dev_mode", False),
        help="Enable diagnostic features: loud failures, verbose logging, execution traces",
    )
    
    if dev_enabled != st.session_state.get("dev_mode"):
        st.session_state["dev_mode"] = dev_enabled
        _set_setting("loud_failures", dev_enabled)
        _set_setting("verbose_logging", dev_enabled)
        _set_setting("trace_executions", dev_enabled)
        
        if dev_enabled:
            st.sidebar.success("🛠️ Dev mode ON")
            _configure_verbose_logging()
        else:
            st.sidebar.info("Dev mode OFF")


def render_dev_settings_panel():
    """
    Render a full dev settings panel with individual toggles.
    
    Use this in an admin/settings page for fine-grained control.
    """
    st.subheader("🛠️ Developer Settings")
    
    # Initialize from environment
    from config.feature_flags import DEV_MODE, LOUD_FAILURES, VERBOSE_LOGGING, TRACE_EXECUTIONS
    
    col1, col2 = st.columns(2)
    
    with col1:
        dev_mode = st.toggle(
            "Dev Mode (Master)",
            value=st.session_state.get("dev_mode", DEV_MODE),
            help="Enable all diagnostic features at once",
        )
        st.session_state["dev_mode"] = dev_mode
        
        loud = st.toggle(
            "Loud Failures",
            value=st.session_state.get("dev_loud_failures", LOUD_FAILURES or dev_mode),
            help="Raise exceptions instead of catching silently",
        )
        _set_setting("loud_failures", loud)
    
    with col2:
        verbose = st.toggle(
            "Verbose Logging",
            value=st.session_state.get("dev_verbose_logging", VERBOSE_LOGGING or dev_mode),
            help="Enable DEBUG-level logging",
        )
        _set_setting("verbose_logging", verbose)
        
        trace = st.toggle(
            "Trace Executions",
            value=st.session_state.get("dev_trace_executions", TRACE_EXECUTIONS or dev_mode),
            help="Log detailed cascade planner traces",
        )
        _set_setting("trace_executions", trace)
    
    # Apply verbose logging if enabled
    if verbose:
        _configure_verbose_logging()
    
    # Show current status
    st.info(f"""
    **Current Settings:**
    - Dev Mode: {'✅' if dev_mode else '❌'}
    - Loud Failures: {'✅' if loud else '❌'}
    - Verbose Logging: {'✅' if verbose else '❌'}
    - Trace Executions: {'✅' if trace else '❌'}
    """)


def _configure_verbose_logging():
    """Configure logging for verbose mode."""
    import sys
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,
    )
    
    # Suppress noisy loggers
    for noisy in ["urllib3", "httpx", "httpcore", "openai", "anthropic", "watchdog"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def diagnostic_raise_ui(exception: Exception) -> None:
    """
    Raise exception if loud failures enabled via UI, otherwise log warning.
    
    This is the UI-aware version of diagnostic_raise from feature_flags.
    """
    if is_loud_failures():
        raise exception
    else:
        logger.warning(f"Suppressed exception (enable Dev Mode for full trace): {exception}")

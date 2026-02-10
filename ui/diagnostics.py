"""
Structured Diagnostics Module for Minerva UI.

Provides centralized error handling, logging, and display for the UI layer.
Captures full context (stack traces, session state) for debugging while
showing user-friendly messages in the interface.

Usage:
    from ui.diagnostics import DiagnosticsManager, capture_error
    
    # Context manager approach
    with capture_error("Loading dataset"):
        df = pd.read_csv(path)
    
    # Decorator approach
    @capture_error("Processing query")
    def process_query(query):
        ...
    
    # Manual approach
    diag = get_diagnostics()
    try:
        ...
    except Exception as e:
        diag.log_error("Failed to load model", e, context={"model": model_name})
        diag.show_error("Could not load the model. Please check the logs.")
"""
import streamlit as st
import logging
import traceback
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from functools import wraps
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticEntry:
    """A single diagnostic log entry."""
    timestamp: str
    level: str  # "error", "warning", "info"
    operation: str  # What was being attempted
    message: str  # User-friendly message
    error_type: Optional[str] = None  # Exception class name
    error_detail: Optional[str] = None  # Exception message
    traceback: Optional[str] = None  # Full traceback
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_log_string(self) -> str:
        """Format for log file."""
        parts = [f"[{self.timestamp}] [{self.level.upper()}] {self.operation}: {self.message}"]
        if self.error_type:
            parts.append(f"  Exception: {self.error_type}: {self.error_detail}")
        if self.context:
            parts.append(f"  Context: {json.dumps(self.context, default=str)}")
        if self.traceback:
            parts.append(f"  Traceback:\n{self.traceback}")
        return "\n".join(parts)


class DiagnosticsManager:
    """
    Centralized diagnostics for UI error handling.
    
    Features:
    - Captures errors with full context
    - Logs to file for debugging
    - Shows user-friendly messages in UI
    - Maintains session-scoped error history
    - Provides export for bug reports
    """
    
    SESSION_KEY = "minerva_diagnostics"
    LOG_FILE = Path("local_data/diagnostics.log")
    
    def __init__(self):
        self._ensure_initialized()
    
    def _ensure_initialized(self):
        """Initialize session state for diagnostics."""
        if self.SESSION_KEY not in st.session_state:
            st.session_state[self.SESSION_KEY] = {
                "entries": [],
                "error_count": 0,
                "warning_count": 0,
                "last_error": None,
            }
        
        # Ensure log directory exists
        self.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    @property
    def entries(self) -> List[DiagnosticEntry]:
        """Get all diagnostic entries for this session."""
        return st.session_state[self.SESSION_KEY]["entries"]
    
    @property
    def error_count(self) -> int:
        """Get count of errors this session."""
        return st.session_state[self.SESSION_KEY]["error_count"]
    
    @property
    def last_error(self) -> Optional[DiagnosticEntry]:
        """Get the most recent error."""
        return st.session_state[self.SESSION_KEY]["last_error"]
    
    def log_error(
        self,
        operation: str,
        exception: Optional[Exception] = None,
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        show_in_ui: bool = True,
    ) -> DiagnosticEntry:
        """
        Log an error with full context.
        
        Args:
            operation: What was being attempted (e.g., "Loading dataset")
            exception: The exception that occurred (if any)
            message: User-friendly message (auto-generated if not provided)
            context: Additional context dict
            show_in_ui: Whether to display an error toast in the UI
            
        Returns:
            The created DiagnosticEntry
        """
        return self._log(
            level="error",
            operation=operation,
            exception=exception,
            message=message,
            context=context,
            show_in_ui=show_in_ui,
        )
    
    def log_warning(
        self,
        operation: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        show_in_ui: bool = True,
    ) -> DiagnosticEntry:
        """Log a warning."""
        return self._log(
            level="warning",
            operation=operation,
            message=message,
            context=context,
            show_in_ui=show_in_ui,
        )
    
    def log_info(
        self,
        operation: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticEntry:
        """Log an info message (no UI display)."""
        return self._log(
            level="info",
            operation=operation,
            message=message,
            context=context,
            show_in_ui=False,
        )
    
    def _log(
        self,
        level: str,
        operation: str,
        exception: Optional[Exception] = None,
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        show_in_ui: bool = True,
    ) -> DiagnosticEntry:
        """Internal logging method."""
        # Build entry
        entry = DiagnosticEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            operation=operation,
            message=message or (str(exception) if exception else "Unknown error"),
            error_type=type(exception).__name__ if exception else None,
            error_detail=str(exception) if exception else None,
            traceback=traceback.format_exc() if exception else None,
            context=context or {},
        )
        
        # Add to session history
        st.session_state[self.SESSION_KEY]["entries"].append(entry)
        if level == "error":
            st.session_state[self.SESSION_KEY]["error_count"] += 1
            st.session_state[self.SESSION_KEY]["last_error"] = entry
        elif level == "warning":
            st.session_state[self.SESSION_KEY]["warning_count"] += 1
        
        # Keep history bounded
        if len(st.session_state[self.SESSION_KEY]["entries"]) > 100:
            st.session_state[self.SESSION_KEY]["entries"] = \
                st.session_state[self.SESSION_KEY]["entries"][-50:]
        
        # Log to file
        self._write_to_log(entry)
        
        # Log to Python logger
        if level == "error":
            logger.error(f"{operation}: {entry.message}", exc_info=exception is not None)
        elif level == "warning":
            logger.warning(f"{operation}: {entry.message}")
        else:
            logger.info(f"{operation}: {entry.message}")
        
        # Show in UI
        if show_in_ui:
            self._show_in_ui(entry)
        
        return entry
    
    def _write_to_log(self, entry: DiagnosticEntry):
        """Append entry to log file."""
        try:
            with open(self.LOG_FILE, "a", encoding="utf-8") as f:
                f.write(entry.to_log_string() + "\n\n")
        except Exception as e:
            logger.debug(f"Could not write to diagnostics log: {e}")
    
    def _show_in_ui(self, entry: DiagnosticEntry):
        """Display entry in Streamlit UI."""
        if entry.level == "error":
            st.error(f"❌ **{entry.operation}**: {entry.message}")
        elif entry.level == "warning":
            st.warning(f"⚠️ **{entry.operation}**: {entry.message}")
    
    def show_error(self, message: str, details: Optional[str] = None):
        """Show an error message in the UI with optional expandable details."""
        st.error(f"❌ {message}")
        if details:
            with st.expander("Technical Details"):
                st.code(details)
    
    def show_warning(self, message: str):
        """Show a warning message in the UI."""
        st.warning(f"⚠️ {message}")
    
    def get_recent_errors(self, n: int = 5) -> List[DiagnosticEntry]:
        """Get the N most recent errors."""
        errors = [e for e in self.entries if e.level == "error"]
        return errors[-n:]
    
    def export_for_report(self) -> str:
        """Export diagnostics as a string for bug reports."""
        lines = ["# Minerva Diagnostics Report", f"Generated: {datetime.now().isoformat()}", ""]
        
        lines.append(f"## Summary")
        lines.append(f"- Total entries: {len(self.entries)}")
        lines.append(f"- Errors: {self.error_count}")
        lines.append(f"- Warnings: {st.session_state[self.SESSION_KEY]['warning_count']}")
        lines.append("")
        
        lines.append("## Recent Entries")
        for entry in self.entries[-20:]:
            lines.append(entry.to_log_string())
            lines.append("")
        
        return "\n".join(lines)
    
    def clear(self):
        """Clear all diagnostic entries."""
        st.session_state[self.SESSION_KEY] = {
            "entries": [],
            "error_count": 0,
            "warning_count": 0,
            "last_error": None,
        }


# =============================================================================
# Global Access
# =============================================================================

_diagnostics: Optional[DiagnosticsManager] = None


def get_diagnostics() -> DiagnosticsManager:
    """Get the global diagnostics manager instance."""
    global _diagnostics
    if _diagnostics is None:
        _diagnostics = DiagnosticsManager()
    return _diagnostics


# =============================================================================
# Convenience Decorators and Context Managers
# =============================================================================

@contextmanager
def capture_error(operation: str, context: Optional[Dict[str, Any]] = None, reraise: bool = False):
    """
    Context manager to capture and log errors.
    
    Usage:
        with capture_error("Loading data", {"file": filename}):
            df = pd.read_csv(filename)
    
    Args:
        operation: Description of what's being attempted
        context: Additional context to include in logs
        reraise: If True, re-raise the exception after logging
    """
    try:
        yield
    except Exception as e:
        diag = get_diagnostics()
        diag.log_error(operation, exception=e, context=context)
        if reraise:
            raise


def with_diagnostics(operation: str):
    """
    Decorator to wrap a function with error capturing.
    
    Usage:
        @with_diagnostics("Processing query")
        def process_query(query):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                diag = get_diagnostics()
                diag.log_error(
                    operation,
                    exception=e,
                    context={"function": func.__name__, "args": str(args)[:100]},
                )
                return None
        return wrapper
    return decorator


# =============================================================================
# UI Components
# =============================================================================

def render_diagnostics_panel():
    """Render a diagnostics panel for admin/debug views."""
    diag = get_diagnostics()
    
    st.subheader("🔍 Diagnostics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Errors", diag.error_count)
    with col2:
        st.metric("Warnings", st.session_state[diag.SESSION_KEY]["warning_count"])
    with col3:
        st.metric("Total Entries", len(diag.entries))
    
    if diag.entries:
        st.subheader("Recent Entries")
        for entry in reversed(diag.entries[-10:]):
            icon = "❌" if entry.level == "error" else "⚠️" if entry.level == "warning" else "ℹ️"
            with st.expander(f"{icon} [{entry.timestamp[:19]}] {entry.operation}"):
                st.write(f"**Message:** {entry.message}")
                if entry.error_type:
                    st.write(f"**Exception:** {entry.error_type}")
                if entry.context:
                    st.write("**Context:**")
                    st.json(entry.context)
                if entry.traceback:
                    st.write("**Traceback:**")
                    st.code(entry.traceback, language="python")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export Report"):
            report = diag.export_for_report()
            st.download_button(
                "📥 Download Report",
                report,
                file_name=f"minerva_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )
    with col2:
        if st.button("Clear Diagnostics"):
            diag.clear()
            st.rerun()

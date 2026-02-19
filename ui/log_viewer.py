"""
Admin Log Viewer for Assay.

Provides in-app viewing of server logs with:
- Multi-file log selection
- Live tail / refresh
- Search and filtering
- Log level highlighting
- Download/export functionality

Usage:
    from ui.log_viewer import render_log_viewer
    
    # In an admin page
    render_log_viewer()
"""
import streamlit as st
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Project root for finding logs
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class LogEntry:
    """Parsed log entry."""
    timestamp: str
    level: str
    source: str
    message: str
    line_number: int
    raw: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "source": self.source,
            "message": self.message,
            "line_number": self.line_number,
        }


# Log level colors for Streamlit
LEVEL_COLORS = {
    "DEBUG": "gray",
    "INFO": "blue",
    "WARNING": "orange",
    "ERROR": "red",
    "CRITICAL": "red",
}

LEVEL_ICONS = {
    "DEBUG": "🔍",
    "INFO": "ℹ️",
    "WARNING": "⚠️",
    "ERROR": "❌",
    "CRITICAL": "🚨",
}


def get_log_files() -> List[Path]:
    """Find all log files in the project."""
    log_files = []
    
    # Root level logs
    for pattern in ["*.log"]:
        log_files.extend(_PROJECT_ROOT.glob(pattern))
    
    # Logs directory
    logs_dir = _PROJECT_ROOT / "logs"
    if logs_dir.exists():
        log_files.extend(logs_dir.glob("*.log"))
    
    # Local data logs
    local_data = _PROJECT_ROOT / "local_data"
    if local_data.exists():
        log_files.extend(local_data.glob("*.log"))
    
    # Sort by modification time (most recent first)
    log_files.sort(key=lambda f: f.stat().st_mtime if f.exists() else 0, reverse=True)
    
    return log_files


def read_log_file(
    file_path: Path,
    tail_lines: int = 500,
    level_filter: Optional[str] = None,
    search_query: Optional[str] = None,
) -> List[LogEntry]:
    """
    Read and parse a log file.
    
    Args:
        file_path: Path to the log file
        tail_lines: Number of lines to read from end
        level_filter: Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        search_query: Filter by search string
        
    Returns:
        List of parsed LogEntry objects
    """
    if not file_path.exists():
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Failed to read log file {file_path}: {e}")
        return []
    
    # Take last N lines
    if tail_lines and len(lines) > tail_lines:
        lines = lines[-tail_lines:]
        start_line = len(lines) - tail_lines
    else:
        start_line = 0
    
    entries = []
    
    # Common log patterns
    # Pattern 1: 2024-01-15 10:30:45,123 - module - LEVEL - message
    pattern1 = re.compile(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:,\d{3})?)\s*-?\s*(\w+)?\s*-?\s*(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s*-?\s*(.*)$")
    
    # Pattern 2: [LEVEL] timestamp message
    pattern2 = re.compile(r"^\[(DEBUG|INFO|WARNING|ERROR|CRITICAL)\]\s*(?:\[?(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})?\]?)?\s*(.*)$")
    
    # Pattern 3: timestamp LEVEL message (common in simple logs)
    pattern3 = re.compile(r"^(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})\s+(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+(.*)$")
    
    for i, line in enumerate(lines):
        line = line.rstrip()
        if not line:
            continue
        
        entry = None
        
        # Try each pattern
        match = pattern1.match(line)
        if match:
            entry = LogEntry(
                timestamp=match.group(1),
                source=match.group(2) or "",
                level=match.group(3),
                message=match.group(4),
                line_number=start_line + i + 1,
                raw=line,
            )
        
        if not entry:
            match = pattern2.match(line)
            if match:
                entry = LogEntry(
                    timestamp=match.group(2) or "",
                    source="",
                    level=match.group(1),
                    message=match.group(3),
                    line_number=start_line + i + 1,
                    raw=line,
                )
        
        if not entry:
            match = pattern3.match(line)
            if match:
                entry = LogEntry(
                    timestamp=match.group(1),
                    source="",
                    level=match.group(2),
                    message=match.group(3),
                    line_number=start_line + i + 1,
                    raw=line,
                )
        
        # Fallback: unparsed line
        if not entry:
            # Check if line contains a level keyword
            level = "INFO"
            for lvl in ["ERROR", "CRITICAL", "WARNING", "DEBUG"]:
                if lvl in line.upper():
                    level = lvl
                    break
            
            entry = LogEntry(
                timestamp="",
                source="",
                level=level,
                message=line,
                line_number=start_line + i + 1,
                raw=line,
            )
        
        # Apply filters
        if level_filter and entry.level != level_filter:
            continue
        
        if search_query and search_query.lower() not in entry.raw.lower():
            continue
        
        entries.append(entry)
    
    return entries


def render_log_viewer():
    """Render the admin log viewer UI."""
    st.subheader("📋 Log Viewer")
    
    # Get available log files
    log_files = get_log_files()
    
    if not log_files:
        st.info("No log files found.")
        return
    
    # File selector
    col1, col2, col3 = st.columns([3, 2, 1])
    
    with col1:
        file_options = {f.name: f for f in log_files}
        selected_name = st.selectbox(
            "Select Log File",
            options=list(file_options.keys()),
            help="Choose a log file to view",
        )
        selected_file = file_options.get(selected_name)
    
    with col2:
        level_filter = st.selectbox(
            "Filter by Level",
            options=["All", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        )
        if level_filter == "All":
            level_filter = None
    
    with col3:
        tail_lines = st.number_input("Lines", min_value=50, max_value=5000, value=500, step=100)
    
    # Search
    search_query = st.text_input("🔍 Search", placeholder="Filter by keyword...")
    
    # Refresh button
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🔄 Refresh"):
            st.rerun()
    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=False)
    
    if auto_refresh:
        st.caption("Refreshing every 5 seconds...")
        import time
        time.sleep(5)
        st.rerun()
    
    # Read and display log entries
    if selected_file:
        # File info
        if selected_file.exists():
            stat = selected_file.stat()
            st.caption(f"📁 {selected_file} | Size: {stat.st_size / 1024:.1f} KB | Modified: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        
        entries = read_log_file(
            selected_file,
            tail_lines=tail_lines,
            level_filter=level_filter,
            search_query=search_query,
        )
        
        if not entries:
            st.info("No log entries match the current filters.")
        else:
            # Summary stats
            level_counts = {}
            for e in entries:
                level_counts[e.level] = level_counts.get(e.level, 0) + 1
            
            cols = st.columns(5)
            for i, level in enumerate(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]):
                count = level_counts.get(level, 0)
                cols[i].metric(f"{LEVEL_ICONS.get(level, '')} {level}", count)
            
            st.divider()
            
            # Display entries
            st.write(f"**Showing {len(entries)} entries:**")
            
            # Use a container with fixed height for scrolling
            log_container = st.container()
            
            with log_container:
                for entry in reversed(entries[-100:]):  # Show most recent 100
                    icon = LEVEL_ICONS.get(entry.level, "•")
                    
                    if entry.level == "ERROR" or entry.level == "CRITICAL":
                        st.error(f"{icon} **[{entry.level}]** {entry.timestamp} - {entry.message}")
                    elif entry.level == "WARNING":
                        st.warning(f"{icon} **[{entry.level}]** {entry.timestamp} - {entry.message}")
                    elif entry.level == "DEBUG":
                        st.caption(f"{icon} [{entry.level}] {entry.timestamp} - {entry.message}")
                    else:
                        st.info(f"{icon} **[{entry.level}]** {entry.timestamp} - {entry.message}")
            
            # Download button
            st.divider()
            full_text = "\n".join(e.raw for e in entries)
            st.download_button(
                "📥 Download Filtered Logs",
                full_text,
                file_name=f"filtered_{selected_name}",
                mime="text/plain",
            )


def render_log_viewer_compact():
    """Render a compact log viewer for embedding in sidebars or small spaces."""
    log_files = get_log_files()
    
    if not log_files:
        st.caption("No logs available")
        return
    
    # Default to most recent log
    selected_file = log_files[0]
    
    entries = read_log_file(selected_file, tail_lines=50, level_filter="ERROR")
    
    if entries:
        st.caption(f"Recent errors from {selected_file.name}:")
        for entry in entries[-5:]:
            st.caption(f"❌ {entry.message[:80]}...")
    else:
        st.caption("✅ No recent errors")


def get_recent_errors(limit: int = 10) -> List[LogEntry]:
    """Get the most recent error entries across all log files."""
    all_errors = []
    
    for log_file in get_log_files()[:5]:  # Check top 5 most recent
        entries = read_log_file(log_file, tail_lines=200, level_filter="ERROR")
        all_errors.extend(entries)
    
    # Sort by timestamp if available
    all_errors.sort(key=lambda e: e.timestamp or "", reverse=True)
    
    return all_errors[:limit]

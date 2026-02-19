"""
Report Generator Module - Export analysis results as Interactive HTML/PDF reports.

Provides professional-looking, interactive analysis reports that users can download
and share. Embeds Plotly.js for interactive visualizations.
"""

import io
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.io as pio

def _generate_html_styles() -> str:
    """Generate CSS styles for the HTML report."""
    return """
    <style>
        :root {
            --primary-color: #2563eb;
            --secondary-color: #64748b;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-color: #1e293b;
            --border-color: #e2e8f0;
        }
        
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            margin: 0;
            padding: 20px;
        }
        
        .report-container {
            max-width: 1000px;
            margin: 0 auto;
            background: var(--card-bg);
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            padding: 40px;
        }
        
        .report-header {
            text-align: center;
            border-bottom: 2px solid var(--primary-color);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        .report-header h1 {
            color: var(--primary-color);
            margin: 0 0 10px 0;
            font-size: 28px;
        }
        
        .report-meta {
            color: var(--secondary-color);
            font-size: 14px;
        }
        
        .section {
            margin-bottom: 40px;
        }
        
        .section h2 {
            color: var(--primary-color);
            border-left: 4px solid var(--primary-color);
            padding-left: 12px;
            font-size: 20px;
            margin-bottom: 20px;
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background: var(--bg-color);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            color: var(--primary-color);
        }
        
        .metric-label {
            color: var(--secondary-color);
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 14px;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        
        th {
            background: var(--bg-color);
            font-weight: 600;
            color: var(--secondary-color);
        }
        
        .insight-box {
            background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
            border-left: 4px solid var(--success-color);
            padding: 15px 20px;
            border-radius: 0 8px 8px 0;
            margin: 15px 0;
        }

        .chat-bubble {
            background-color: #f1f5f9;
            padding: 15px;
            border-radius: 12px;
            margin-bottom: 15px;
            border-left: 4px solid #94a3b8;
        }
        .chat-bubble.assistant {
            background-color: #dbeafe;
            border-left-color: var(--primary-color);
        }
        .chat-role {
            font-weight: bold;
            font-size: 0.8em;
            text-transform: uppercase;
            margin-bottom: 5px;
            color: var(--secondary-color);
        }

        .plotly-graph-div {
            margin: 20px 0;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px;
        }
        
        .footer {
            text-align: center;
            color: var(--secondary-color);
            font-size: 12px;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
        }
    </style>
    """


def generate_data_summary_html(df: pd.DataFrame) -> str:
    """Generate HTML summary of the dataset."""
    html = f"""
    <div class="section">
        <h2>📊 Dataset Overview</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{df.shape[0]:,}</div>
                <div class="metric-label">Rows</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{df.shape[1]}</div>
                <div class="metric-label">Columns</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{df.isna().sum().sum():,}</div>
                <div class="metric-label">Missing Values</div>
            </div>
        </div>
    </div>
    """
    return html


def generate_figures_html(figures: List[Any]) -> str:
    """Embed Plotly figures using plotly-latest.min.js."""
    if not figures:
        return ""
    
    html = '<div class="section"><h2>📈 Visualizations</h2>'
    
    for i, fig in enumerate(figures):
        if fig is None:
            continue
            
        # Serialize fig to JSON
        fig_json = pio.to_json(fig)
        div_id = f"plotly-div-{uuid.uuid4()}"
        
        html += f"""
        <div id="{div_id}" class="plotly-graph-div"></div>
        <script>
            Plotly.newPlot('{div_id}', {fig_json}, {{responsive: true}});
        </script>
        """
        
    html += "</div>"
    return html


def generate_chat_history_html(chat_history: List[Dict]) -> str:
    """Generate formatted chat history."""
    if not chat_history:
        return ""
        
    html = '<div class="section"><h2>💬 Analysis Narrative</h2>'
    
    for msg in chat_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # Very basic markdown to html (newlines to br)
        content_html = content.replace("\n", "<br>")
        
        bs_class = "assistant" if role == "assistant" else "user"
        
        html += f"""
        <div class="chat-bubble {bs_class}">
            <div class="chat-role">{role}</div>
            <div class="chat-content">{content_html}</div>
        </div>
        """
        
    html += "</div>"
    return html


def generate_html_report(
    df: pd.DataFrame,
    result: Dict[str, Any] = None,
    figures: Optional[List[Any]] = None,
    chat_history: Optional[List[Dict]] = None,
    title: str = "Data Analysis Report",
    include_data_preview: bool = True,
) -> str:
    """
    Generate a complete Interactive HTML report.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <!-- Load Plotly.js from CDN -->
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    {_generate_html_styles()}
</head>
<body>
    <div class="report-container">
        <div class="report-header">
            <h1>{title}</h1>
            <p class="report-meta">Generated by Assay Analytics Platform</p>
            <p class="report-meta">{timestamp}</p>
        </div>
        
        {generate_data_summary_html(df)}
        
        {generate_chat_history_html(chat_history) if chat_history else ""}
        
        {generate_figures_html(figures) if figures else ""}
    """
    
    # Data preview
    if include_data_preview:
        preview = df.head(10)
        html += """
        <div class="section">
            <h2>📋 Data Preview</h2>
        """
        html += preview.to_html(classes="", index=False, border=0)
        html += "</div>"
    
    html += f"""
        <div class="footer">
            <p>Generated by Assay Analytics Platform</p>
            <p>© {datetime.now().year} - Confidential</p>
        </div>
    </div>
</body>
</html>
    """
    
    return html


def generate_report_bytes(
    df: pd.DataFrame,
    result: Dict[str, Any] = None,
    figures: Optional[List[Any]] = None,
    chat_history: Optional[List[Dict]] = None,
    title: str = "Data Analysis Report",
) -> bytes:
    """
    Generate report as bytes for download.
    """
    html_content = generate_html_report(df, result, figures, chat_history, title)
    return html_content.encode("utf-8")

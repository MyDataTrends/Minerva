"""
Report Generator Module - Export analysis results as HTML/PDF reports.

Provides professional-looking analysis reports that users can download
and share with stakeholders.
"""

import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


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
            max-width: 900px;
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
            margin-bottom: 30px;
        }
        
        .section h2 {
            color: var(--primary-color);
            border-left: 4px solid var(--primary-color);
            padding-left: 12px;
            font-size: 20px;
            margin-bottom: 15px;
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
        
        tr:hover {
            background: var(--bg-color);
        }
        
        .insight-box {
            background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
            border-left: 4px solid var(--success-color);
            padding: 15px 20px;
            border-radius: 0 8px 8px 0;
            margin: 15px 0;
        }
        
        .warning-box {
            background: #fffbeb;
            border-left: 4px solid var(--warning-color);
            padding: 15px 20px;
            border-radius: 0 8px 8px 0;
            margin: 15px 0;
        }
        
        .footer {
            text-align: center;
            color: var(--secondary-color);
            font-size: 12px;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
        }
        
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }
        
        .badge-success { background: #dcfce7; color: #166534; }
        .badge-warning { background: #fef3c7; color: #92400e; }
        .badge-info { background: #dbeafe; color: #1e40af; }
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
                <div class="metric-value">{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB</div>
                <div class="metric-label">Memory Usage</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{df.isna().sum().sum():,}</div>
                <div class="metric-label">Missing Values</div>
            </div>
        </div>
    """
    
    # Column types summary
    type_counts = df.dtypes.value_counts()
    html += """
        <h3>Column Types</h3>
        <table>
            <tr><th>Type</th><th>Count</th></tr>
    """
    for dtype, count in type_counts.items():
        html += f"<tr><td>{dtype}</td><td>{count}</td></tr>"
    html += "</table></div>"
    
    return html


def generate_analysis_results_html(result: Dict[str, Any]) -> str:
    """Generate HTML for analysis results."""
    analysis_type = result.get("analysis_type", "unknown")
    
    type_labels = {
        "regression": ("📈", "Regression Analysis", "badge-info"),
        "classification": ("🏷️", "Classification Analysis", "badge-success"),
        "clustering": ("🔮", "Clustering Analysis", "badge-warning"),
        "forecasting": ("📅", "Time Series Forecast", "badge-info"),
        "descriptive": ("📋", "Descriptive Statistics", "badge-info"),
        "anomaly": ("🔍", "Anomaly Detection", "badge-warning"),
    }
    
    emoji, label, badge_class = type_labels.get(analysis_type, ("📊", "Analysis", "badge-info"))
    
    html = f"""
    <div class="section">
        <h2>{emoji} Analysis Results</h2>
        <p><span class="badge {badge_class}">{label}</span></p>
    """
    
    # Model metrics
    model_info = result.get("model_info", {})
    metrics = model_info.get("metrics", {})
    
    if metrics:
        html += '<div class="metric-grid">'
        for name, value in metrics.items():
            formatted = f"{value:.3f}" if isinstance(value, float) else str(value)
            html += f"""
                <div class="metric-card">
                    <div class="metric-value">{formatted}</div>
                    <div class="metric-label">{name.replace('_', ' ').title()}</div>
                </div>
            """
        html += "</div>"
    
    # Feature importance
    explanations = model_info.get("explanations", {})
    fi = explanations.get("feature_importances") or explanations.get("coefficients")
    if fi:
        html += "<h3>Feature Importance</h3><table><tr><th>Feature</th><th>Importance</th></tr>"
        sorted_fi = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        for feature, importance in sorted_fi:
            bar_width = min(abs(importance) * 100, 100)
            html += f"""
                <tr>
                    <td>{feature}</td>
                    <td>
                        <div style="background: linear-gradient(90deg, var(--primary-color) {bar_width}%, transparent {bar_width}%); height: 20px; border-radius: 4px;"></div>
                        {importance:.4f}
                    </td>
                </tr>
            """
        html += "</table>"
    
    html += "</div>"
    return html


def generate_insights_html(result: Dict[str, Any]) -> str:
    """Generate HTML for business insights."""
    summary = result.get("summary", "")
    insights = result.get("insights", [])
    
    html = '<div class="section"><h2>💡 Key Insights</h2>'
    
    if summary:
        html += f'<div class="insight-box">{summary}</div>'
    
    if insights:
        for insight in insights:
            html += f'<div class="insight-box">{insight}</div>'
    
    # Warnings
    diagnostics = result.get("diagnostics", {})
    warnings = []
    if diagnostics.get("missing_pct", 0) > 10:
        warnings.append(f"⚠️ High percentage of missing data ({diagnostics.get('missing_pct', 0):.1f}%)")
    if diagnostics.get("duplicate_rows", 0) > 0:
        warnings.append(f"⚠️ {diagnostics.get('duplicate_rows', 0)} duplicate rows detected")
    
    for warning in warnings:
        html += f'<div class="warning-box">{warning}</div>'
    
    if not summary and not insights:
        html += '<p>No specific insights generated for this analysis.</p>'
    
    html += "</div>"
    return html


def generate_html_report(
    df: pd.DataFrame,
    result: Dict[str, Any],
    title: str = "Data Analysis Report",
    include_data_preview: bool = True,
) -> str:
    """
    Generate a complete HTML report.
    
    Args:
        df: The analyzed DataFrame
        result: Analysis result dictionary
        title: Report title
        include_data_preview: Whether to include a data preview table
        
    Returns:
        Complete HTML document as string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {_generate_html_styles()}
</head>
<body>
    <div class="report-container">
        <div class="report-header">
            <h1>{title}</h1>
            <p class="report-meta">Generated by Minerva Analytics Platform</p>
            <p class="report-meta">{timestamp}</p>
        </div>
        
        {generate_data_summary_html(df)}
        {generate_analysis_results_html(result)}
        {generate_insights_html(result)}
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
    
    # Merge report if available
    merge_report = result.get("model_info", {}).get("merge_report")
    if merge_report:
        html += """
        <div class="section">
            <h2>🔗 Data Enrichment Report</h2>
        """
        if merge_report.get("successful_merges", 0) > 0:
            html += f"""
            <div class="insight-box">
                Successfully merged with <strong>{merge_report.get('successful_merges', 0)}</strong> 
                external dataset(s), adding <strong>{merge_report.get('total_new_columns', 0)}</strong> new columns.
            </div>
            """
            if merge_report.get("merged_tables"):
                html += "<p>Merged tables: " + ", ".join(merge_report["merged_tables"]) + "</p>"
        html += "</div>"
    
    html += f"""
        <div class="footer">
            <p>Generated by Minerva Analytics Platform</p>
            <p>© {datetime.now().year} - Confidential</p>
        </div>
    </div>
</body>
</html>
    """
    
    return html


def export_report_to_file(
    df: pd.DataFrame,
    result: Dict[str, Any],
    output_path: str,
    title: str = "Data Analysis Report",
) -> str:
    """
    Export report to a file.
    
    Args:
        df: The analyzed DataFrame
        result: Analysis result dictionary
        output_path: Where to save the report
        title: Report title
        
    Returns:
        Path to the saved file
    """
    html_content = generate_html_report(df, result, title)
    
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    output.write_text(html_content, encoding="utf-8")
    
    return str(output)


def generate_report_bytes(
    df: pd.DataFrame,
    result: Dict[str, Any],
    title: str = "Data Analysis Report",
) -> bytes:
    """
    Generate report as bytes for download.
    
    Args:
        df: The analyzed DataFrame
        result: Analysis result dictionary
        title: Report title
        
    Returns:
        HTML content as bytes
    """
    html_content = generate_html_report(df, result, title)
    return html_content.encode("utf-8")

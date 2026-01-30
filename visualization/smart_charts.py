"""
Smart Chart Selection - Automatically choose optimal chart types based on data characteristics.

This module analyzes DataFrame columns and suggests the most appropriate visualization types
based on data types, cardinality, distributions, and relationships.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ChartType(Enum):
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    PIE = "pie"
    BOX = "box"
    HEATMAP = "heatmap"
    AREA = "area"
    VIOLIN = "violin"
    TREEMAP = "treemap"


class ColumnRole(Enum):
    TEMPORAL = "temporal"           # Dates, timestamps, years
    CATEGORICAL = "categorical"     # Low cardinality text/categories
    CATEGORICAL_HIGH = "categorical_high"  # High cardinality (e.g., product IDs)
    NUMERIC_CONTINUOUS = "numeric_continuous"  # Measurements, amounts
    NUMERIC_DISCRETE = "numeric_discrete"  # Counts, quantities
    IDENTIFIER = "identifier"       # IDs, keys (not useful for viz)
    BOOLEAN = "boolean"
    GEOGRAPHIC = "geographic"       # Lat/lon, country codes
    PROPORTION = "proportion"       # Percentages, ratios (0-1 or 0-100)


@dataclass
class ColumnProfile:
    """Profile of a single column's characteristics."""
    name: str
    dtype: str
    role: ColumnRole
    cardinality: int
    null_pct: float
    is_monotonic: bool = False
    is_sequential: bool = False
    min_val: Any = None
    max_val: Any = None
    mean_val: float = None
    std_val: float = None


@dataclass
class ChartRecommendation:
    """A single chart recommendation with reasoning."""
    chart_type: ChartType
    x_col: str
    y_col: Optional[str]
    color_col: Optional[str] = None
    title: str = ""
    reason: str = ""
    confidence: float = 0.8
    priority: int = 1  # Lower = higher priority


def profile_column(df: pd.DataFrame, col: str) -> ColumnProfile:
    """Analyze a column and determine its role and characteristics."""
    series = df[col]
    dtype = str(series.dtype)
    cardinality = series.nunique()
    null_pct = series.isna().sum() / len(series) * 100
    n_rows = len(df)
    
    # Determine role
    role = ColumnRole.CATEGORICAL
    is_monotonic = False
    is_sequential = False
    min_val = max_val = mean_val = std_val = None
    
    # Check for temporal
    if 'datetime' in dtype:
        role = ColumnRole.TEMPORAL
        is_monotonic = series.is_monotonic_increasing or series.is_monotonic_decreasing
    elif any(kw in col.lower() for kw in ['date', 'time', 'year', 'month', 'day', 'timestamp', 'created', 'updated']):
        role = ColumnRole.TEMPORAL
        if dtype in ['int64', 'float64']:
            is_monotonic = series.dropna().is_monotonic_increasing
    
    # Check for identifier
    elif any(kw in col.lower() for kw in ['id', 'key', 'uuid', 'guid', 'code']) and cardinality > n_rows * 0.8:
        role = ColumnRole.IDENTIFIER
    
    # Check for boolean
    elif dtype == 'bool' or (cardinality == 2 and series.dropna().isin([0, 1, True, False, 'yes', 'no', 'true', 'false']).all()):
        role = ColumnRole.BOOLEAN
    
    # Check for geographic
    elif any(kw in col.lower() for kw in ['lat', 'lon', 'country', 'state', 'city', 'region', 'zip', 'postal']):
        role = ColumnRole.GEOGRAPHIC
    
    # Numeric analysis
    elif dtype in ['int64', 'float64', 'int32', 'float32']:
        min_val = series.min()
        max_val = series.max()
        mean_val = series.mean()
        std_val = series.std()
        
        # Check for proportion (0-1 or 0-100)
        if min_val >= 0:
            if max_val <= 1 and mean_val < 1:
                role = ColumnRole.PROPORTION
            elif max_val <= 100 and (any(kw in col.lower() for kw in ['pct', 'percent', 'ratio', 'rate', 'share'])):
                role = ColumnRole.PROPORTION
            elif cardinality < 20 and (max_val - min_val) < 50:
                role = ColumnRole.NUMERIC_DISCRETE
            else:
                role = ColumnRole.NUMERIC_CONTINUOUS
        else:
            role = ColumnRole.NUMERIC_CONTINUOUS
        
        is_monotonic = series.dropna().is_monotonic_increasing or series.dropna().is_monotonic_decreasing
        # Check if sequential (1,2,3,4... or dates)
        if is_monotonic and cardinality > n_rows * 0.9:
            is_sequential = True
    
    # Categorical analysis
    elif dtype == 'object' or dtype.name == 'category':
        if cardinality <= 10:
            role = ColumnRole.CATEGORICAL
        elif cardinality <= 50:
            role = ColumnRole.CATEGORICAL_HIGH
        else:
            role = ColumnRole.IDENTIFIER  # Too many categories = probably an ID
    
    return ColumnProfile(
        name=col,
        dtype=dtype,
        role=role,
        cardinality=cardinality,
        null_pct=null_pct,
        is_monotonic=is_monotonic,
        is_sequential=is_sequential,
        min_val=min_val,
        max_val=max_val,
        mean_val=mean_val,
        std_val=std_val
    )


def profile_dataframe(df: pd.DataFrame) -> Dict[str, ColumnProfile]:
    """Profile all columns in a DataFrame."""
    return {col: profile_column(df, col) for col in df.columns}


def recommend_charts(df: pd.DataFrame, max_recommendations: int = 5) -> List[ChartRecommendation]:
    """
    Analyze DataFrame and recommend optimal chart types.
    
    Returns recommendations sorted by priority (best first).
    """
    profiles = profile_dataframe(df)
    recommendations = []
    n_rows = len(df)
    
    # Categorize columns by role
    temporal_cols = [p.name for p in profiles.values() if p.role == ColumnRole.TEMPORAL]
    numeric_cols = [p.name for p in profiles.values() if p.role in [ColumnRole.NUMERIC_CONTINUOUS, ColumnRole.NUMERIC_DISCRETE, ColumnRole.PROPORTION]]
    categorical_cols = [p.name for p in profiles.values() if p.role == ColumnRole.CATEGORICAL]
    categorical_high_cols = [p.name for p in profiles.values() if p.role == ColumnRole.CATEGORICAL_HIGH]
    boolean_cols = [p.name for p in profiles.values() if p.role == ColumnRole.BOOLEAN]
    
    # =====================================================
    # RULE 1: Time Series (highest priority)
    # =====================================================
    if temporal_cols and numeric_cols:
        time_col = temporal_cols[0]
        for num_col in numeric_cols[:2]:
            recommendations.append(ChartRecommendation(
                chart_type=ChartType.LINE,
                x_col=time_col,
                y_col=num_col,
                title=f"{num_col} Over Time",
                reason=f"Time series visualization showing {num_col} trends over {time_col}",
                confidence=0.95,
                priority=1
            ))
    
    # =====================================================
    # RULE 2: Category Comparisons
    # =====================================================
    if categorical_cols and numeric_cols:
        cat_col = categorical_cols[0]
        cat_profile = profiles[cat_col]
        num_col = numeric_cols[0]
        
        if cat_profile.cardinality <= 10:
            # Few categories = bar chart
            recommendations.append(ChartRecommendation(
                chart_type=ChartType.BAR,
                x_col=cat_col,
                y_col=num_col,
                title=f"{num_col} by {cat_col}",
                reason=f"Compare {num_col} across {cat_col} categories",
                confidence=0.9,
                priority=2
            ))
            # Also suggest pie if proportions make sense
            if cat_profile.cardinality <= 6:
                recommendations.append(ChartRecommendation(
                    chart_type=ChartType.PIE,
                    x_col=cat_col,
                    y_col=num_col,
                    title=f"{num_col} Distribution by {cat_col}",
                    reason=f"Show proportion of {num_col} across {cat_profile.cardinality} categories",
                    confidence=0.7,
                    priority=4
                ))
    
    # =====================================================
    # RULE 3: Distributions (for continuous data)
    # =====================================================
    for num_col in numeric_cols[:2]:
        num_profile = profiles[num_col]
        if num_profile.role == ColumnRole.NUMERIC_CONTINUOUS:
            recommendations.append(ChartRecommendation(
                chart_type=ChartType.HISTOGRAM,
                x_col=num_col,
                y_col=None,
                title=f"Distribution of {num_col}",
                reason=f"Understand the distribution and spread of {num_col} values",
                confidence=0.85,
                priority=3
            ))
    
    # =====================================================
    # RULE 4: Correlations (scatter plots)
    # =====================================================
    if len(numeric_cols) >= 2:
        col1, col2 = numeric_cols[0], numeric_cols[1]
        # Calculate correlation to see if scatter is useful
        try:
            corr = df[col1].corr(df[col2])
            if abs(corr) > 0.3:  # Some correlation exists
                recommendations.append(ChartRecommendation(
                    chart_type=ChartType.SCATTER,
                    x_col=col1,
                    y_col=col2,
                    color_col=categorical_cols[0] if categorical_cols else None,
                    title=f"{col1} vs {col2}",
                    reason=f"Explore relationship between {col1} and {col2} (correlation: {corr:.2f})",
                    confidence=0.8,
                    priority=3
                ))
        except:
            pass
    
    # =====================================================
    # RULE 5: Box plots for category + numeric
    # =====================================================
    if categorical_cols and numeric_cols:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        cat_profile = profiles[cat_col]
        
        if 2 <= cat_profile.cardinality <= 8:
            recommendations.append(ChartRecommendation(
                chart_type=ChartType.BOX,
                x_col=cat_col,
                y_col=num_col,
                title=f"{num_col} Distribution by {cat_col}",
                reason=f"Compare distribution of {num_col} across {cat_col} categories",
                confidence=0.75,
                priority=4
            ))
    
    # =====================================================
    # RULE 6: Heatmap for correlation matrix
    # =====================================================
    if len(numeric_cols) >= 3:
        recommendations.append(ChartRecommendation(
            chart_type=ChartType.HEATMAP,
            x_col="*numeric*",
            y_col="*numeric*",
            title="Correlation Heatmap",
            reason=f"Visualize correlations between {len(numeric_cols)} numeric columns",
            confidence=0.8,
            priority=5
        ))
    
    # Sort by priority and limit
    recommendations.sort(key=lambda r: (r.priority, -r.confidence))
    return recommendations[:max_recommendations]


def get_best_chart(df: pd.DataFrame, intent: str = "") -> Optional[ChartRecommendation]:
    """
    Get the single best chart recommendation for a DataFrame.
    
    Args:
        df: DataFrame to visualize
        intent: Optional user intent (e.g., "compare", "trend", "distribution")
    """
    recommendations = recommend_charts(df, max_recommendations=1)
    if recommendations:
        return recommendations[0]
    return None


def render_recommendation(df: pd.DataFrame, rec: ChartRecommendation):
    """
    Render a chart recommendation using Plotly.
    
    Returns a Plotly figure.
    """
    import plotly.express as px
    
    fig = None
    
    if rec.chart_type == ChartType.LINE:
        df_sorted = df.sort_values(rec.x_col) if rec.x_col in df.columns else df
        fig = px.line(df_sorted, x=rec.x_col, y=rec.y_col, title=rec.title)
    
    elif rec.chart_type == ChartType.BAR:
        # Aggregate if needed
        if rec.y_col and rec.y_col in df.columns:
            agg_df = df.groupby(rec.x_col)[rec.y_col].sum().reset_index()
            agg_df = agg_df.nlargest(20, rec.y_col)  # Top 20
            fig = px.bar(agg_df, x=rec.x_col, y=rec.y_col, title=rec.title)
        else:
            counts = df[rec.x_col].value_counts().head(20).reset_index()
            counts.columns = [rec.x_col, 'count']
            fig = px.bar(counts, x=rec.x_col, y='count', title=rec.title)
    
    elif rec.chart_type == ChartType.SCATTER:
        sample = df.sample(min(1000, len(df))) if len(df) > 1000 else df
        fig = px.scatter(sample, x=rec.x_col, y=rec.y_col, color=rec.color_col, 
                        title=rec.title, opacity=0.6)
    
    elif rec.chart_type == ChartType.HISTOGRAM:
        fig = px.histogram(df, x=rec.x_col, title=rec.title)
    
    elif rec.chart_type == ChartType.PIE:
        if rec.y_col and rec.y_col in df.columns:
            agg_df = df.groupby(rec.x_col)[rec.y_col].sum().reset_index()
        else:
            agg_df = df[rec.x_col].value_counts().reset_index()
            agg_df.columns = [rec.x_col, 'count']
            rec.y_col = 'count'
        fig = px.pie(agg_df, names=rec.x_col, values=rec.y_col, title=rec.title)
    
    elif rec.chart_type == ChartType.BOX:
        fig = px.box(df, x=rec.x_col, y=rec.y_col, title=rec.title)
    
    elif rec.chart_type == ChartType.HEATMAP:
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=".2f", title=rec.title, 
                       color_continuous_scale="RdBu_r")
    
    elif rec.chart_type == ChartType.AREA:
        df_sorted = df.sort_values(rec.x_col) if rec.x_col in df.columns else df
        fig = px.area(df_sorted, x=rec.x_col, y=rec.y_col, title=rec.title)
    
    if fig:
        fig.update_layout(template="plotly_dark")
    
    return fig

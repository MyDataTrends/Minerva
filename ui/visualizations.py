import streamlit as st
# import matplotlib.pyplot as plt - moved to functions
import pandas as pd


def generate_bar_chart(data: pd.DataFrame):
    """Display a simple bar chart using the first two columns."""
    if data.empty or data.shape[1] < 2:
        return "Not enough data for bar chart"
    x_col = data.columns[0]
    y_col = data.select_dtypes(include="number").columns
    if len(y_col) == 0:
        return "No numeric column available for bar chart"
    y_col = y_col[0]
    y_col = y_col[0]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.bar(data[x_col], data[y_col])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"Bar Chart of {y_col} by {x_col}")
    st.pyplot(plt)


def generate_scatter_plot(data: pd.DataFrame):
    """Display a scatter plot using the first two numeric columns."""
    numeric = data.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return "Need two numeric columns for scatter plot"
    x_col, y_col = numeric.columns[:2]
    x_col, y_col = numeric.columns[:2]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(data[x_col], data[y_col])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"Scatter Plot of {y_col} vs {x_col}")
    st.pyplot(plt)


def generate_histogram(data: pd.DataFrame):
    """Display a histogram of the first numeric column."""
    numeric = data.select_dtypes(include="number")
    if numeric.empty:
        return "No numeric data for histogram"
    col = numeric.columns[0]
    col = numeric.columns[0]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.hist(data[col], bins=20, edgecolor="black")
    plt.xlabel(col)
    plt.title(f"Histogram of {col}")
    st.pyplot(plt)


def generate_heatmap(data: pd.DataFrame):
    """Display a correlation heatmap for numeric columns."""
    numeric = data.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return "Need multiple numeric columns for heatmap"
    corr = numeric.corr()
    corr = numeric.corr()
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    plt.imshow(corr, cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr)), corr.columns)
    plt.title("Correlation Heatmap")
    st.pyplot(plt)


def generate_pie_chart(data: pd.DataFrame):
    """Display a pie chart of the first categorical column."""
    if data.empty:
        return "No data for pie chart"
    cat_cols = data.select_dtypes(exclude="number").columns
    col = cat_cols[0] if len(cat_cols) else data.columns[0]
    counts = data[col].value_counts().head(10)
    counts = data[col].value_counts().head(10)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%")
    plt.title(f"Pie Chart of {col}")
    st.pyplot(plt)


def generate_area_chart(data: pd.DataFrame):
    """Display an area chart using the first two columns."""
    if data.empty or data.shape[1] < 2:
        return "Not enough data for area chart"
    x_col = data.columns[0]
    y_col = data.select_dtypes(include="number").columns
    if len(y_col) == 0:
        return "No numeric column available for area chart"
    y_col = y_col[0]
    y_col = y_col[0]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.fill_between(data[x_col], data[y_col], alpha=0.4)
    plt.plot(data[x_col], data[y_col], color="blue")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"Area Chart of {y_col} over {x_col}")
    st.pyplot(plt)

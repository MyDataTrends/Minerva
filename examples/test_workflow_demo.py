import pandas as pd
from orchestrator import orchestrate_dashboard


def main():
    mock_data = pd.DataFrame({
        "Date": pd.date_range(start="2022-01-01", periods=100, freq="M"),
        "Sales": [i * 1.05 for i in range(100)],
    })

    mock_model_output = {"predictions": [100, 200, 300], "key_metric": "RMSE", "confidence_score": 85}

    mock_kernel_analysis = {"general": {"line_chart": 0.9, "bar_chart": 0.7, "heatmap": 0.6}}

    mock_chatbot_input = "Show a line chart of sales over time."

    output = orchestrate_dashboard(
        mock_data, mock_model_output, "time-series", "Sales", mock_kernel_analysis, mock_chatbot_input
    )
    print("Dashboard Output:")
    print(output)


if __name__ == "__main__":
    main()

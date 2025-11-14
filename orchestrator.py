
from __future__ import annotations

from typing import Callable, Mapping, Any



def orchestrate_dashboard(
    data,
    model_output,
    model_type,
    target_variable,
    kernel_analysis,
    chatbot_input,
    visualizations: Mapping[str, Callable] | None = None,
    models: Mapping[str, Callable] | None = None,
):
    """
    Orchestrates model selection, visualization, and dashboard creation dynamically, with chatbot integration.

    Args:
        data (pd.DataFrame): Enriched dataset.
        model_output (dict): Results from the selected model (e.g., predictions, key metrics).
        model_type (str): Type of model selected (e.g., time-series, cross-tab, etc.).
        target_variable (str): Name of the target variable in the dataset.
        kernel_analysis (dict): Analysis of dataset kernels for visualization preferences.
        chatbot_input (str): User query from chatbot.

    Returns:
        dict: Dashboard layout with selected visualizations and insights.
    """
    from chatbot.decision import decide_action
    from analysis_selector import select_analyzer
    from feedback.ratings import get_average_rating
    from visualization_selector import infer_visualization_type

    # Step 1: Determine requested action from chatbot
    # ``decide_action`` maps the raw intent to either a modeling request,
    # visualization request or indicates that we should fall back to automatic
    # analysis when the chat message doesn't match known patterns.
    action, params = decide_action(chatbot_input)
    
    # Step 2: Determine Visualization Type or Scenario using heuristics
    requested_visualization = params.get(
        "type"
    ) or infer_visualization_type(chatbot_input, data, model_type)
    visualization_types = [requested_visualization]

    # Score visualizations using kernel analysis
    kernel_preferences = kernel_analysis.get("general", {})
    scored_visualizations = [
        {"type": vis, "score": kernel_preferences.get(vis, 0.5)}
        for vis in visualization_types
    ]
    scored_visualizations = sorted(
        scored_visualizations, key=lambda x: x["score"], reverse=True
    )

    # Step 3: Handle Chatbot Requests
    layout: list[dict[str, Any]] = []
    insights: list[str] = []
    chart_result = None
    model_result = None

    if action == "visualization":
        layout.append({"visualization": requested_visualization, "description": f"{target_variable} Visualization"})
        insights.append(f"Generated a {requested_visualization} for {target_variable}.")
        if visualizations and requested_visualization in visualizations:
            chart_result = visualizations[requested_visualization](data)
    elif action == "modeling":
        # Either a specific modeling task or a scenario request
        task = params.get("task")
        if task and models and task in models:
            model_result = models[task](data)
            layout.append({"visualization": task, "description": f"Modeling task: {task}"})
            insights.append(f"Ran {task} model on {target_variable}.")
        else:
            adjustments = params.get("adjustments", {})
            layout.append({"visualization": "scenario", "description": f"Scenario Simulation with Adjustments: {adjustments}"})
            insights.append(f"Simulated scenario with adjustments: {adjustments}.")
            if models and "scenario_generator" in models:
                model_result = models["scenario_generator"](params)
        if visualizations and requested_visualization in visualizations:
            chart_result = visualizations[requested_visualization](data)
    else:
        # Auto-analysis is triggered when the chat message doesn't contain a
        # recognizable modeling or visualization request
        insights.append(
            "No actionable request detected from the chatbot input. Running automatic analysis."
        )
        if models and "default_model" in models:
            model_result = models["default_model"](data)
        else:
            analyzer = select_analyzer(data)
            model_result = analyzer.run(data)
        layout.append({"visualization": requested_visualization, "description": "Automatic analysis"})
        if visualizations and requested_visualization in visualizations:
            chart_result = visualizations[requested_visualization](data)

    # Step 4: Confidence Score from Model Output
    confidence_score = model_output.get("confidence_score", 80)  # Default confidence score

    return {
        "dashboard": layout,
        "insights": insights,
        "confidence_score": f"The confidence in these results is {confidence_score}%.",
        "scored_visualizations": scored_visualizations,
        "visual_layout": layout,
        "model_result": model_result,
        "chart_result": chart_result,
        "average_rating": get_average_rating(),
    }

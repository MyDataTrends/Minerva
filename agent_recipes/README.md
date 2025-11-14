# Agent Recipes

Agent recipes are small post‑processing hooks executed by `agents.action_agent` after the main result notification step. They allow the application to trigger side effects such as sending requests to an external service whenever certain conditions are met.

The recipes included in this repository are:

- **`suggest_reorder`** – posts model output to `REORDER_API` when any prediction is negative.
- **`send_anomaly_alert`** – alerts `ANOMALY_API` if the latest MAE is 20% worse than the previous run.
- **`request_role_review`** – calls `ROLE_REVIEW_API` when the workflow flags a dataset for manual role review.

The action agent only loads these hooks when the environment variable `ENABLE_RECIPES` is set to `True`.

## Adding a New Recipe

1. **Create a function** in `agent_recipes/` that accepts the workflow result dictionary and performs the desired action. The helper should check for its endpoint variable before making a request:

   ```python
   from config import MY_API
   from utils.net import request_with_retry

   def my_custom_action(output: dict) -> None:
       if not MY_API:
           return
       request_with_retry("post", MY_API, json=output, timeout=5)
   ```

2. **Import and trigger** the recipe in `agents/action_agent.py` inside the `ENABLE_RECIPES` block. Define the condition that should invoke it and append an action name to the returned list for logging.

## Wiring External APIs

Set the following environment variables (for example in `.env`) so the recipes know where to send requests:

- `REORDER_API` – URL for inventory reorder suggestions.
- `ANOMALY_API` – URL that receives anomaly alerts.
- `ROLE_REVIEW_API` – URL to request manual column-role review.
- `ENABLE_RECIPES` – must be `True` to enable execution of any recipes.

These values are read in `config/__init__.py` and are optional; if a variable is empty the related recipe logs a message and exits without sending a request.

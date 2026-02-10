import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestration.cascade_planner import CascadePlanner, Intent, PlanStep

class TestSemanticResolution(unittest.TestCase):
    def setUp(self):
        self.planner = CascadePlanner()
        self.df = pd.DataFrame({
            "id": [1, 2],
            "cost": [10, 20],
            "revenue": [100, 200]
        })
        self.context = {"df": self.df}

    @patch("orchestration.plan_learner.get_learner")
    def test_semantic_input_resolution(self, mock_get_learner):
        # Setup mock learner
        mock_learner = MagicMock()
        mock_get_learner.return_value = mock_learner
        
        query = "visualize revenue"
        intent_value = Intent.VISUALIZE.value
        
        # Configure mock to return specific mapping for this query
        # valid columns in df are: id, cost, revenue
        mock_learner.get_suggested_inputs.side_effect = lambda intent, context, query=None: (
            {"y": "revenue"} if query == "visualize revenue" else {}
        )
        
        # Create a step with placeholders
        step = PlanStep(
            step_id="step_1",
            action="generate_chart",
            tool="chart_generator",
            inputs={"y": "__infer_y__", "x": "__infer_x__"}, # x will fall back to heuristic
            expected_output="chart"
        )
        
        # Invoke the method under test
        resolved = self.planner._llm_resolve_inputs(
            step=step,
            inputs=step.inputs,
            context=self.context,
            intent=intent_value,
            query=query
        )
        
        # Assertions
        print(f"DEBUG: Resolved inputs: {resolved}")
        self.assertEqual(resolved["y"], "revenue", "Should resolve 'y' to 'revenue' via learner")
        
        # Verify heuristic fallback for x (should be first column 'id')
        # Note: logic says 'columns[0]' for x.
        self.assertEqual(resolved["x"], "id", "Should resolve 'x' to 'id' via heuristic")

    @patch("orchestration.plan_learner.get_learner")
    def test_semantic_resolution_failure(self, mock_get_learner):
        # Setup mock learner to return garbage or nothing
        mock_learner = MagicMock()
        mock_get_learner.return_value = mock_learner
        mock_learner.get_suggested_inputs.return_value = {}
        
        query = "visualize unexpected"
        
        step = PlanStep(
            step_id="step_1",
            action="generate_chart",
            tool="chart_generator",
            inputs={"y": "__infer_y__"},
            expected_output="chart"
        )
        
        resolved = self.planner._llm_resolve_inputs(
            step=step,
            inputs=step.inputs,
            context=self.context,
            intent=Intent.VISUALIZE.value,
            query=query
        )
        
        # Assertions
        # Should fall back to heuristic (first numeric col -> 'id' or 'cost' depending on dtype)
        # 'id' is int (numeric). 'cost' is int. 'revenue' is int.
        # df.select_dtypes(include=["number"]) returns all 3.
        # Heuristic picks numeric_cols[0].
        # In pandas, order is preserved. So 'id'.
        self.assertIn(resolved["y"], ["id", "cost"], "Should fall back to heuristic (not revenue unless coincidence)")

if __name__ == "__main__":
    unittest.main()

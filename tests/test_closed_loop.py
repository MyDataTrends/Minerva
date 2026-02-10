import unittest
from unittest.mock import MagicMock, patch, ANY
import pandas as pd
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestration.cascade_planner import CascadePlanner, Intent, PlanStep, ExecutionPlan

class TestClosedLoop(unittest.TestCase):
    def setUp(self):
        self.planner = CascadePlanner()
        self.df = pd.DataFrame({"id": [1], "revenue": [100]})
        self.context = {"df": self.df}

    @patch("orchestration.plan_learner.get_learner")
    @patch("orchestration.cascade_planner.classify_intent")
    def test_plan_from_learned_pattern(self, mock_classify, mock_get_learner):
        # Setup mocks
        mock_classify.return_value = (Intent.VISUALIZE, 0.8)
        
        mock_learner = MagicMock()
        mock_get_learner.return_value = mock_learner
        
        # Mock a learned pattern
        mock_pattern = MagicMock()
        mock_pattern.tool_sequence = ["tool_A", "tool_B"]
        mock_pattern.input_mappings = {"y": "revenue", "strategy": "mean"}
        mock_pattern.confidence = 0.9
        
        mock_learner.get_suggested_plan.return_value = mock_pattern
        
        # Execute
        query = "visualize revenue trend"
        plan = self.planner.plan(query, self.context)
        
        # Verify
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].tool, "tool_A")
        self.assertEqual(plan.steps[1].tool, "tool_B")
        
        # Verify inputs propagated
        # Step 1 should have learned inputs + df
        self.assertEqual(plan.steps[0].inputs["y"], "revenue")
        self.assertEqual(plan.steps[0].inputs["strategy"], "mean")
        self.assertEqual(plan.steps[0].inputs["df"], "__context.df__")
        
        # Verify learner was queried
        mock_learner.get_suggested_plan.assert_called_once_with(Intent.VISUALIZE.value, query)

    @patch("orchestration.plan_learner.get_learner")
    @patch("orchestration.cascade_planner.classify_intent")
    def test_fallback_when_no_pattern(self, mock_classify, mock_get_learner):
        # Setup mocks
        mock_classify.return_value = (Intent.UNKNOWN, 0.5)
        
        mock_learner = MagicMock()
        mock_get_learner.return_value = mock_learner
        mock_learner.get_suggested_plan.return_value = None # No pattern found
        
        # Mock _llm_generate_plan (part of planner)
        with patch.object(self.planner, "_llm_generate_plan") as mock_llm_gen:
            mock_llm_gen.return_value = [PlanStep("step1", "action", "tool", {}, "")]
            
            # Execute
            query = "something novel"
            plan = self.planner.plan(query, self.context)
            
            # Verify fallback to LLM
            mock_llm_gen.assert_called_once()
            self.assertEqual(len(plan.steps), 1)

if __name__ == "__main__":
    unittest.main()

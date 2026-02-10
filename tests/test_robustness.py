import unittest
from unittest.mock import MagicMock, patch, ANY
import pandas as pd
import sys
import os
from datetime import datetime

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestration.cascade_planner import CascadePlanner, PlanStep, ExecutionPlan, Intent, PlanStatus

class TestRobustness(unittest.TestCase):
    def setUp(self):
        self.planner = CascadePlanner()
        self.df = pd.DataFrame({
            "revenue": [100, 200],
            "cost": [50, 150],
            "category": ["A", "B"]
        })
        self.context = {"df": self.df}

    def test_dependency_skipping(self):
        # Plan: Step 1 fails, Step 2 depends on Step 1
        step1 = PlanStep("step_1", "fail_action", "fail_tool", {}, "")
        step2 = PlanStep("step_2", "dependent_action", "dep_tool", {}, "", depends_on=["step_1"])
        
        plan = ExecutionPlan("plan_1", Intent.UNKNOWN, "q", [step1, step2])
        
        # Mock invoke_tool to fail for step 1
        with patch("orchestration.tool_registry.invoke_tool") as mock_invoke:
            result_fail = MagicMock(success=False, error="Tool failed", execution_time_ms=10)
            mock_invoke.return_value = result_fail
            
            self.planner.execute(plan, self.context)
            
            # Verify step 1 failed
            self.assertEqual(plan.steps[0].status, PlanStatus.FAILED)
            # Verify step 2 failed/skipped
            self.assertEqual(plan.steps[1].status, PlanStatus.FAILED)
            self.assertIn("Dependencies failed", plan.steps[1].error)
            
            # Ensure invoke_tool was called only once (for step 1)
            mock_invoke.assert_called_once()

    def test_step_level_fallback(self):
        # Plan: Step 1 uses tool A, fallback tool B
        # Mock: Tool A fails, Tool B succeeds
        step1 = PlanStep("step_1", "action", "primary_tool", {}, "", fallback_tool="fallback_tool")
        plan = ExecutionPlan("plan_2", Intent.UNKNOWN, "q", [step1])
        
        with patch("orchestration.tool_registry.invoke_tool") as mock_invoke:
            # Side effect: first call fails, second call succeeds
            result_fail = MagicMock(success=False, error="Primary failed", execution_time_ms=10)
            result_success = MagicMock(success=True, output="Success", execution_time_ms=10)
            
            mock_invoke.side_effect = [result_fail, result_success]
            
            with patch("orchestration.tool_registry.get_tool") as mock_get_tool:
                mock_get_tool.return_value = True # Mock fallback tool exists
                
                self.planner.execute(plan, self.context)
                
                # Verify Step 1 succeeded eventually
                self.assertEqual(plan.steps[0].status, PlanStatus.SUCCESS)
                self.assertEqual(plan.steps[0].metadata["fallback_used"], "fallback_tool")
                
                # Verify invoke called twice
                self.assertEqual(mock_invoke.call_count, 2)
                args_list = mock_invoke.call_args_list
                self.assertEqual(args_list[0][0][0], "primary_tool")
                self.assertEqual(args_list[1][0][0], "fallback_tool")

    @patch("llm_manager.llm_interface.get_llm_completion")
    def test_intent_llm_fallback(self, mock_llm):
        # Mock LLM to return "visualize"
        mock_llm.return_value = "visualize"
        
        from orchestration.cascade_planner import classify_intent
        
        # Query that fails regex
        query = "Can you make a picture of the sales?"
        # Assuming regex doesn't catch "picture" or "sales" combined this way? 
        # Actually "picture" is not in regex list (chart, graph, plot).
        
        intent, confidence = classify_intent(query)
        
        self.assertEqual(intent, Intent.VISUALIZE)
        self.assertEqual(confidence, 0.6)
        mock_llm.assert_called_once()

    def test_semantic_auto_correction(self):
        # Planner has _validate_semantic_inputs
        inputs = {"x": "revenu", "y": "cst", "df": self.df}
        
        self.planner._validate_semantic_inputs("tool", inputs, self.context)
        
        # Verify correction
        self.assertEqual(inputs["x"], "revenue")
        self.assertEqual(inputs["y"], "cost")

if __name__ == "__main__":
    unittest.main()

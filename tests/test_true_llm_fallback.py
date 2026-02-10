import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import sys
import os
import json

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestration.cascade_planner import CascadePlanner, PlanStep

class TestTrueLLMFallback(unittest.TestCase):
    def setUp(self):
        self.planner = CascadePlanner()
        self.df = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
        self.context = {"df": self.df}

    @patch("llm_manager.llm_interface.get_llm_completion")
    def test_llm_generate_plan_success(self, mock_completion):
        # Mock LLM response
        mock_response = json.dumps([
            {
                "tool": "filter_rows",
                "inputs": {"column": "value", "operator": ">", "value": 15, "df": "__context.df__"},
                "reasoning": "Filter for high values"
            },
            {
                "tool": "table_display",
                "inputs": {"df": "__context.df__", "max_rows": 10},
                "reasoning": "Show result"
            }
        ])
        mock_completion.return_value = mock_response
        
        # Execute
        steps = self.planner._llm_generate_plan("Show me values greater than 15", self.context)
        
        # Verify
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].tool, "filter_rows")
        self.assertEqual(steps[0].inputs["value"], 15)
        self.assertEqual(steps[1].tool, "table_display")
        
        # Verify prompt contained schema
        args, _ = mock_completion.call_args
        prompt = args[0]
        self.assertIn("AVAILABLE TOOLS:", prompt)
        self.assertIn("DATA CONTEXT:", prompt)
        self.assertIn("value", prompt) # column name

    @patch("llm_manager.llm_interface.get_llm_completion")
    def test_llm_generate_plan_invalid_json(self, mock_completion):
        # Mock invalid response
        mock_completion.return_value = "This is not JSON"
        
        # Execute
        steps = self.planner._llm_generate_plan("Show me something", self.context)
        
        # Verify fallback
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].action, "describe_data") # Default fallback

if __name__ == "__main__":
    unittest.main()

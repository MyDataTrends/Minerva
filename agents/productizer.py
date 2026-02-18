"""
Productizer Agent — Vertical Use Case Generator.

Analyzes a dataset + run outputs to create a "Best Fit" Vertical MVP Plan.
This helps monetize the app by showing specific value propositions to different user bases.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from agents.base import AgentConfig, AgentResult, BaseAgent, Priority, TriggerType
from agents.memory.operational import OperationalMemory
from llm_manager.llm_interface import get_llm_completion
from preprocessing.metadata_parser import parse_metadata
from utils.logging import get_logger

logger = get_logger(__name__)

# Directory where user data is stored
USER_DATA_DIR = Path(__file__).resolve().parents[1] / "User_Data"
# Directory to save generated plans
KB_PRODUCT_DIR = Path(__file__).resolve().parents[1] / "agents" / "knowledge_base" / "product"


from mcp_server.discovery_agent import APIDiscoveryAgent

class ProductizerAgent(BaseAgent):
    """
    Vertical Sales & Use Case Generator.

    Objective: Create compelling "Sales Kits" for different verticals based on data samples.
    
    Workflow:
    1. Sources data (Concept -> Data):
       - User provides a topic (e.g. "Crypto", "Weather").
       - Agent uses APIDiscoveryAgent to fetch public data.
    2. Analyzes the dataset to understand the domain.
    3. Generates a "Vertical Sales Kit" (Persona, Pitch, Script, ROI).
    4. Saves the kit to `knowledge_base/product/sales_kits/`.
    """

    name = "productizer"
    trigger_type = TriggerType.MANUAL

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config)
        self.memory = OperationalMemory("productizer")
        
        # New dedicated directory for sales assets
        self.sales_kit_dir = KB_PRODUCT_DIR / "sales_kits"
        self.sales_kit_dir.mkdir(parents=True, exist_ok=True)

    def run(self, **kwargs) -> AgentResult:
        """Execute the productizer workflow."""
        result = self._make_result()
        
        # 1. Acquire Data (File or Topic)
        target_file = kwargs.get("file")
        topic = kwargs.get("topic")
        
        if topic and not target_file:
            # Source data automatically
            result.add_action(f"sourcing_data_for_topic:{topic}")
            target_file = self._source_data(topic, result)
            
        if not target_file:
            # Fallback to latest local file
            target_file = self._find_latest_dataset()
            result.add_action("using_latest_local_file")
        
        if not target_file:
            result.error = "No data found. Provide 'file' or 'topic'."
            result.success = False
            return result
            
        result.add_action(f"selected_dataset:{target_file.name}")
        
        # 2. Profile Data
        try:
            profile = self._profile_dataset(target_file)
            result.add_action("profiled_dataset")
        except Exception as e:
            result.error = f"Failed to profile dataset: {e}"
            result.success = False
            return result

        # 3. Generate Sales Kit via LLM
        try:
            sales_kit = self._generate_sales_kit(profile, target_file.name)
            result.add_action("generated_sales_kit")
        except Exception as e:
            result.error = f"LLM generation failed: {e}"
            result.success = False
            return result

        # 4. Save Sales Kit
        try:
            output_path = self._save_sales_kit(sales_kit, target_file.name)
            result.add_action(f"saved_kit:{output_path.name}")
            result.summary = f"Generated Sales Kit for {sales_kit.get('vertical', 'General')} (based on {target_file.name})"
            
            result.metrics["vertical"] = sales_kit.get("vertical")
            result.metrics["kit_file"] = str(output_path)
            
        except Exception as e:
            result.error = f"Failed to save kit: {e}"
            result.success = False
            return result

        return result

    def _source_data(self, topic: str, result: AgentResult) -> Optional[Path]:
        """Use DiscoveryAgent to fetch data for a topic."""
        try:
            discoverer = APIDiscoveryAgent()
            query = f"Recent high quality dataset for {topic} in CSV or JSON format"
            
            # Use one_click_fetch to get data
            data, status = discoverer.one_click_fetch(query)
            
            if data is None:
                result.add_escalation(Priority.FYI, "Data Sourcing Failed", status)
                logger.warning(f"Data sourcing failed: {status}")
                return None
            
            # Save to User_Data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_topic = topic.replace(" ", "_").lower()
            filename = f"sourced_{sanitized_topic}_{timestamp}.csv"
            
            save_path = USER_DATA_DIR / filename
            USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
            
            # Convert to DataFrame if it's a dict/list
            if isinstance(data, (dict, list)):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame(data) # Attempt direct conversion
                
            df.to_csv(save_path, index=False)
            return save_path
            
        except Exception as e:
            logger.error(f"Error sourcing data: {e}")
            return None

    def _find_latest_dataset(self) -> Optional[Path]:
        """Find the most recently modified CSV/Excel in User_Data."""
        if not USER_DATA_DIR.exists():
            return None
        candidates = []
        for ext in ["*.csv", "*.xlsx", "*.xls", "*.parquet"]:
            candidates.extend(USER_DATA_DIR.rglob(ext))
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def _profile_dataset(self, file_path: Path) -> Dict[str, Any]:
        """Load and profile the dataset."""
        if file_path.suffix == ".csv":
            df = pd.read_csv(file_path, nrows=100)
        elif file_path.suffix in (".xls", ".xlsx"):
            df = pd.read_excel(file_path, nrows=100)
        elif file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path) # Parquet usually manages without nrows
            df = df.head(100)
        else:
            raise ValueError(f"Unsupported format: {file_path.suffix}")

        meta = parse_metadata(df)
        return {
            "columns": meta["columns"],
            "dtypes": meta["dtypes"],
            "sample": df.head(3).to_dict(orient="records"),
        }

    def _generate_sales_kit(self, profile: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """generate a compelling Sales Kit for this data's vertical."""
        
        prompt = f"""
        You are an Elite Solutions Engineer and Product Marketer for Minerva.
        Minerva is an AI Data Analyst that replaces manual dashboarding with autonomous insights.
        
        Your Goal: Create a "Vertical Sales Kit" based on this dataset. 
        We want to sell Minerva to a specific industry by showing them a concrete use case using this data.

        Dataset Context:
        File: {filename}
        Columns: {profile['columns']}
        Sample: {json.dumps(profile['sample'], default=str)}
        
        Step 1: Identify the Vertical (e.g., Retail, Healthcare, Fintech, Logistics).
        Step 2: Identify the Buyer Persona (e.g., VP of Sales, Supply Chain Lead).
        Step 3: Define 3 "Wake Up in a Sweat" Pain Points this persona has.
        Step 4: Script a "Minerva Demo" that solves these pains using the provided columns.
        
        Output JSON:
        {{
            "vertical": "Industry Name",
            "title": "Use Case Title (e.g., 'Automated Churn Prediction for Telecom')",
            "persona": "Target Buyer",
            "pain_points": [
                "Pain 1",
                "Pain 2", 
                "Pain 3"
            ],
            "elevator_pitch": "2-3 sentences selling Minerva to this persona.",
            "demo_script": [
                {{
                    "step": "1. Ask this question...",
                    "action": "Minerva will generate...",
                    "wow_factor": "Why this impresses them"
                }},
                ... (3-4 steps)
            ],
            "roi_metrics": [
                "Metric 1 (e.g. Save 20 hours/week)",
                "Metric 2"
            ]
        }}
        """
        
        response = get_llm_completion(prompt, max_tokens=2500, temperature=0.7)
        
        try:
            clean_resp = response.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_resp)
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse Sales Kit JSON")
            return {
                "vertical": "General",
                "title": "Data Analysis Use Case",
                "raw_content": response
            }

    def _save_sales_kit(self, kit: Dict[str, Any], filename: str) -> Path:
        """Save the Sales Kit as a polished Markdown document."""
        timestamp = datetime.now().strftime("%Y%m%d")
        vertical_slug = kit.get("vertical", "general").lower().replace(" ", "_")
        output_file = self.sales_kit_dir / f"sales_kit_{vertical_slug}_{timestamp}.md"
        
        if "raw_content" in kit:
            content = kit["raw_content"]
        else:
            demo_steps = ""
            for i, step in enumerate(kit.get('demo_script', []), 1):
                demo_steps += f"### Step {i}: {step.get('step')}\n"
                demo_steps += f"- **Action**: {step.get('action')}\n"
                demo_steps += f"- **✨ The 'Wow' Factor**: {step.get('wow_factor')}\n\n"

            content = f"""# 🚀 Minerva Sales Kit: {kit.get('vertical')}

> **Use Case**: {kit.get('title')}
> **Target Persona**: {kit.get('persona')}

## 1. The Executive Pitch
{kit.get('elevator_pitch')}

## 2. The Pain Points (Why they buy)
{chr(10).join([f"* 🔴 **{p}**" for p in kit.get('pain_points', [])])}

## 3. The Solution: Minerva Autonomous Analyst
We don't just "show data". We solve the problem.

### 💰 ROI & Business Impact
{chr(10).join([f"* 🟢 {m}" for m in kit.get('roi_metrics', [])])}

---

## 4. The Golden Demo Script
*Use the dataset `{filename}` for this demo.*

{demo_steps}

---
*Generated by Minerva Productizer Agent on {datetime.now().strftime('%Y-%m-%d')}*
"""
            
        output_file.write_text(content, encoding="utf-8")
        return output_file

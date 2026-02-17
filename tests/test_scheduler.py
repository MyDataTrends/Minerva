
import sys
import os
import time
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.getcwd())

from agents.scheduler import SchedulerAgent

def test_scheduler():
    print("Testing Scheduler Agent...")
    
    # 1. Setup Dummy Data
    data_dir = Path("User_Data/test_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "test_sales.csv"
    
    df = pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=10),
        "Sales": [100, 120, 110, 130, 140, 160, 150, 170, 180, 190],
        "Category": ["A", "B"] * 5
    })
    df.to_csv(csv_path, index=False)
    print(f"Created test data at {csv_path}")

    # 2. Initialize Agent
    agent = SchedulerAgent(storage_path="User_Data/test_schedules.json")
    
    # 3. Add Job (Immediate execution via interval)
    job = {
        "id": "test_job_1",
        "name": "Test Sales Report",
        "frequency": "interval",
        "interval_minutes": 1, # Should run potentially next minute, but we can force run
        "dataset_path": str(csv_path),
        "prompt": "Calculate the average sales and max sales."
    }
    
    agent.add_job(job, save=False)
    print("Job added.")

    # 4. Force Execute (to avoid waiting)
    print("Force executing job...")
    agent._execute_job(job)
    
    # 5. Check Report
    reports_dir = Path("reports")
    if reports_dir.exists():
        files = list(reports_dir.glob("*.md"))
        if files:
            print(f"✅ Report created: {files[-1]}")
            print("Content preview:")
            print(files[-1].read_text()[:200])
        else:
            print("❌ No report file found.")
    else:
        print("❌ Reports directory not created.")

if __name__ == "__main__":
    test_scheduler()

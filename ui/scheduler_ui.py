"""
Scheduler UI - Frontend for the Scheduler Agent.
"""
import streamlit as st
from datetime import datetime
from agents.scheduler import SchedulerAgent
import pandas as pd

def render_scheduler_ui():
    st.header("📅 Analysis Scheduler")
    st.markdown("Automate your analysis workflows. Schedules run in the background.")
    
    # Initialize Agent
    if "scheduler_agent" not in st.session_state:
        st.session_state["scheduler_agent"] = SchedulerAgent()
        st.session_state["scheduler_agent"].start()
    
    agent = st.session_state["scheduler_agent"]
    
    # --- Create New Job ---
    with st.expander("➕ Schedule New Job", expanded=True):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            job_name = st.text_input("Job Name", placeholder="Weekly Sales Report")
            
            # Dataset Selection (Only those with paths)
            paths = st.session_state.get("dataset_paths", {})
            if not paths:
                st.warning("Upload a CSV to schedule analysis (API sources not yet supported for scheduling).")
                dataset_name = None
            else:
                dataset_name = st.selectbox("Select Dataset", list(paths.keys()))
        
        with col2:
            frequency = st.selectbox("Frequency", ["Daily", "Weekly", "Interval (Minutes)"])
            
            if frequency == "Weekly":
                day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
                time = st.time_input("Time", value=datetime.strptime("09:00", "%H:%M").time())
            elif frequency == "Daily":
                 time = st.time_input("Time", value=datetime.strptime("09:00", "%H:%M").time())
            else:
                interval = st.number_input("Every X Minutes", min_value=1, value=60)

        prompt = st.text_area("Analysis Prompt", placeholder="Analyze the sales trends for the last month...")
        
        if st.button("Schedule Job", type="primary", disabled=not (job_name and dataset_name and prompt)):
            job_config = {
                "id": f"job_{int(datetime.now().timestamp())}",
                "name": job_name,
                "dataset_path": paths[dataset_name],
                "prompt": prompt,
                "frequency": frequency.lower().split()[0] if "Interval" not in frequency else "interval",
            }
            
            if frequency == "Weekly":
                job_config["day_of_week"] = day
                job_config["time"] = time.strftime("%H:%M")
            elif frequency == "Daily":
                job_config["time"] = time.strftime("%H:%M")
            else:
                job_config["interval_minutes"] = interval
            
            agent.add_job(job_config)
            st.success(f"Scheduled '{job_name}'!")
            st.rerun()

    # --- Active Schedules ---
    st.subheader("Active Schedules")
    jobs = agent.list_jobs()
    
    if not jobs:
        st.info("No active schedules.")
    else:
        for job in jobs:
            with st.container():
                c1, c2, c3 = st.columns([2, 2, 1])
                c1.markdown(f"**{job['name']}**")
                
                freq_str = job.get("frequency", "daily").title()
                if freq_str == "Weekly":
                    timing = f"{job.get('day_of_week')} at {job.get('time')}"
                elif freq_str == "Daily":
                     timing = f"at {job.get('time')}"
                else:
                    timing = f"Every {job.get('interval_minutes')} mins"
                    
                c2.caption(f"{freq_str} • {timing}")
                c2.caption(f"Dataset: {job.get('dataset_path')}")
                
                if c3.button("Run Now", key=f"run_{job['id']}"):
                    agent._execute_job(job)
                    st.toast(f"Executed {job['name']}")
                
                st.markdown("---")
                
    # --- Reports ---
    st.subheader("📄 Generated Reports")
    import os
    if os.path.exists("reports"):
        reports = sorted(os.listdir("reports"), reverse=True)
        for r in reports:
            if r.endswith(".md"):
                with st.expander(r):
                    with open(f"reports/{r}", "r") as f:
                        st.markdown(f.read())

import os
import sqlite3
import json
import streamlit as st
import pandas as pd
from ui import redaction_banner

DB_PATH = os.getenv("SESSION_DB_PATH", "session_history.db")


def load_history(db_path: str = DB_PATH) -> pd.DataFrame:
    """Load session history from SQLite DB sorted by timestamp desc."""
    if not os.path.exists(db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(
            "SELECT timestamp, file_name, model_name, output FROM sessions ORDER BY timestamp DESC",
            conn,
        )
    except Exception as exc:  # pragma: no cover - best effort
        st.error(f"Failed to read {db_path}: {exc}")
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


st.title("Session History")
redaction_banner()

history = load_history()
if history.empty:
    st.info("No session history found.")
else:
    st.write("Recent sessions")
    for i, row in history.iterrows():
        cols = st.columns([3, 3, 3, 2])
        cols[0].write(row["timestamp"])
        cols[1].write(row["file_name"])
        cols[2].write(row["model_name"])
        output = row.get("output")
        if isinstance(output, bytes):
            data = output
        else:
            data = json.dumps(output, indent=2).encode()
        cols[3].download_button(
            "Download", data, file_name=f"{row['file_name']}_output.json", mime="application/json", key=f"dl_{i}"
        )


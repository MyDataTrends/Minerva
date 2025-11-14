import streamlit as st
import pandas as pd
from typing import Dict, List

from preprocessing.column_meta import ColumnMeta
from preprocessing.sanitize import scrub_df
from utils.role_mapper import map_description_to_role
from ui import redaction_banner


def column_review(df: pd.DataFrame, meta: List[ColumnMeta]) -> Dict[str, str]:
    """Render a column review UI and return updated roles when submitted."""
    st.header("Column Review")
    redaction_banner()
    st.write("Verify detected column roles. Edit any values then click Submit.")
    updated = {}
    sample = scrub_df(df.head(50))
    for m in meta:
        st.subheader(m.name)
        st.dataframe(sample[[m.name]])
        desc_key = f"desc_{m.name}"
        role_key = f"role_{m.name}"
        desc_val = st.text_input("Description", value=m.description or "", key=desc_key)
        if desc_val:
            suggested = map_description_to_role(desc_val)
            if role_key not in st.session_state or st.session_state[role_key] == m.role:
                st.session_state[role_key] = suggested
        updated[m.name] = st.text_input(
            "Role", value=st.session_state.get(role_key, m.role), key=role_key
        )
    if st.button("Submit Roles"):
        st.session_state["roles_submitted"] = True
    if st.session_state.get("roles_submitted"):
        return updated
    return {}

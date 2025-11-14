import json
from pathlib import Path
import sys
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def test_dashboard_recommendations_and_report(tmp_path):
    dash_path = Path(__file__).resolve().parents[1] / "ui" / "dashboard.py"
    at = AppTest.from_file(str(dash_path))
    at.session_state["result"] = {
        "analysis_type": "regression",
        "predictions": [1, 2],
        "anomalies": [],
        "recommended_models": {"semantic_merge": ["lr", "rf"]},
        "model_info": {"merge_report": {"chosen_table": "dummy.csv"}},
    }
    at.run(timeout=20)

    exp = at.sidebar.expander[0]
    assert exp.label == "Recommended Models"
    rec = json.loads(exp.json[0].value)
    assert rec["semantic_merge"] == ["lr", "rf"]

    report_exp = at.expander[0]
    assert report_exp.label == "Merge Report"
    dl = report_exp.get("download_button")[0]
    assert dl.proto.label == "Download merge_report.json"


def test_run_history_display(tmp_path, monkeypatch):
    run_history = [
        {"run_id": "abc123", "score_ok": True, "needs_role_review": False}
    ]
    (tmp_path / "run_history.json").write_text(json.dumps(run_history))
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))

    dash_path = Path(__file__).resolve().parents[1] / "ui" / "dashboard.py"
    at = AppTest.from_file(str(dash_path))
    at.run(timeout=20)

    assert any(s.value == "Run History" for s in at.sidebar.subheader)
    markdowns = [m.value for m in at.sidebar.markdown]
    assert "abc123" in "".join(markdowns)

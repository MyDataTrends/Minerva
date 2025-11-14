import pandas as pd
from analysis_selector import select_analyzer


def test_select_and_run_simple():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "sales": [5, 4, 3, 2, 1]})
    analyzer = select_analyzer(df)
    result = analyzer.run(df)
    assert "summary" in result
    assert result["alternatives"]

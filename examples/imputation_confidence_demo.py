import pandas as pd
import numpy as np
from imputation_confidence import score_imputations


def main():
    data = {
        "A": [1.0, 2.0, np.nan, 4.0, 5.0],
        "B": [10.0, np.nan, 30.0, 40.0, 50.0],
        "C": [100.0, 200.0, 300.0, np.nan, 500.0],
        "D": [7.0, 8.0, np.nan, 10.0, 11.0],
    }
    df_original = pd.DataFrame(data)

    df = df_original.copy()
    df["A"].fillna(df["A"].mean(), inplace=True)
    df["B"].fillna(df["B"].median(), inplace=True)
    df["C"].fillna(df["C"].mean(), inplace=True)
    df["D"].fillna(df["D"].mean(), inplace=True)

    imputed_mask = pd.DataFrame({
        "A": df_original["A"].isna(),
        "B": df_original["B"].isna(),
        "C": df_original["C"].isna(),
        "D": df_original["D"].isna(),
    })

    method_map = {"A": "mean", "B": "median", "C": "knn", "D": "model"}
    result = score_imputations(df, imputed_mask, method_map, df_original=df_original)
    print("Summary:\n", result["summary"])


if __name__ == "__main__":
    main()

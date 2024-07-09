"""Clean raw data asn save into csv
"""

import numpy as np
import pandas as pd


# Add code to load in the data.
def load_data(path: str) -> pd.DataFrame:
    """Load Pandas CSV file given path"""
    return pd.read_csv(path)


_df = load_data("./data/census.csv")

df = _df.copy()
df.columns = [col.strip(" ").replace("-", "_") for col in df.columns]


for col in df.select_dtypes("O").columns:
    df[col] = df[col].str.strip(" ")
    df[col] = df[col].replace("?", "Missing")


df.to_csv("./data/cleaned_census.csv", index=False)

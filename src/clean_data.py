"""Clean raw data asn save into csv
"""

import pandas as pd
import yaml

with open("config.yaml", "rb") as f:
    config = yaml.safe_load(f)


# Add code to load in the data.
def load_data(path: str) -> pd.DataFrame:
    """Load Pandas CSV file given path"""
    return pd.read_csv(path)


def main():
    _df = load_data(config["DATA_PATH"])

    df = _df.copy()
    df.columns = [col.strip(" ").replace("-", "_") for col in df.columns]

    for col in df.select_dtypes("O").columns:
        df[col] = df[col].str.strip(" ")
        df[col] = df[col].replace("?", "Missing")

    df.to_csv(config["CLEANED_DATA_PATH"], index=False)


if __name__ == "__main__":
    main()

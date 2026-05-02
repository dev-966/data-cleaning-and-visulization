from pathlib import Path

import pandas as pd


def load_sales_data(path: Path) -> pd.DataFrame:
    """Load the sales CSV with the encoding used by this dataset."""
    df = pd.read_csv(path, encoding="latin-1")
    df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"], errors="coerce")
    df["ORDER_MONTH"] = df["ORDERDATE"].dt.to_period("M").astype(str)
    df["ORDER_YEAR_MONTH"] = df["ORDERDATE"].dt.to_period("M")
    return df

"""Data loading and preparation."""

from pathlib import Path

import pandas as pd


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_data(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, encoding="latin1")
    df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"], errors="coerce")
    df["ORDER_MONTH"] = df["ORDERDATE"].dt.to_period("M").astype(str)
    df["ORDER_YEAR"] = df["ORDERDATE"].dt.year
    df["ORDER_MONTH_NUM"] = df["ORDERDATE"].dt.month
    df["PROFIT_PROXY"] = df["SALES"] - (df["QUANTITYORDERED"] * df["MSRP"])
    return df

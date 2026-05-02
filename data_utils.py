"""Data loading and feature selection helpers."""

from pathlib import Path

import pandas as pd

from config import DEFAULT_FEATURES


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def select_modeling_data(df: pd.DataFrame, target: str, include_sales: bool) -> tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found in the dataset.")

    features = [column for column in DEFAULT_FEATURES if column in df.columns]
    if include_sales and "SALES" in df.columns:
        features.append("SALES")

    if not features:
        raise ValueError("No usable feature columns were found in the dataset.")

    modeling_df = df[features + [target]].dropna(subset=[target]).copy()
    return modeling_df[features], modeling_df[target].astype(str)

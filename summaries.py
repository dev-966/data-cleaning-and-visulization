"""Summary table generation for the EDA project."""

from pathlib import Path

import pandas as pd


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def analyze_data_quality(df: pd.DataFrame, output_dir: Path) -> None:
    quality = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
            "missing_count": df.isna().sum().values,
            "missing_percent": (df.isna().mean().values * 100).round(2),
            "unique_values": df.nunique(dropna=True).values,
        }
    )
    quality.to_csv(output_dir / "data_quality.csv", index=False)


def save_summary_statistics(df: pd.DataFrame, output_dir: Path) -> None:
    df.describe(include="all").transpose().to_csv(output_dir / "summary_statistics.csv")


def build_grouped_tables(df: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {
        "productline_sales": df.groupby("PRODUCTLINE")["SALES"]
        .agg(["count", "sum", "mean", "median"])
        .sort_values("sum", ascending=False),
        "country_sales": df.groupby("COUNTRY")["SALES"]
        .agg(["count", "sum", "mean", "median"])
        .sort_values("sum", ascending=False),
        "dealsize_sales": df.groupby("DEALSIZE")["SALES"]
        .agg(["count", "sum", "mean", "median"])
        .sort_values("mean", ascending=False),
        "year_sales": df.groupby("YEAR_ID")["SALES"]
        .agg(["count", "sum", "mean", "median"])
        .sort_values("sum", ascending=False),
        "status_sales": df.groupby("STATUS")["SALES"]
        .agg(["count", "sum", "mean", "median"])
        .sort_values("sum", ascending=False),
    }

    for name, table in tables.items():
        table.round(2).to_csv(output_dir / f"{name}.csv")

    return tables

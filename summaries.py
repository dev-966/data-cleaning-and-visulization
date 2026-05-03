"""Summary tables for business analysis."""

from pathlib import Path

import pandas as pd


def save_data_quality(df: pd.DataFrame, output_dir: Path) -> None:
    quality = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[column].dtype) for column in df.columns],
            "missing_values": [df[column].isna().sum() for column in df.columns],
            "missing_percent": [round(df[column].isna().mean() * 100, 2) for column in df.columns],
            "unique_values": [df[column].nunique(dropna=True) for column in df.columns],
        }
    )
    quality.to_csv(output_dir / "data_quality.csv", index=False)
    df.describe(include="all").transpose().to_csv(output_dir / "summary_statistics.csv")


def save_business_tables(df: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {
        "productline_performance": df.groupby("PRODUCTLINE", as_index=False)
        .agg(total_sales=("SALES", "sum"), orders=("ORDERNUMBER", "nunique"), units=("QUANTITYORDERED", "sum"))
        .sort_values("total_sales", ascending=False),
        "country_performance": df.groupby("COUNTRY", as_index=False)
        .agg(total_sales=("SALES", "sum"), orders=("ORDERNUMBER", "nunique"), avg_sale=("SALES", "mean"))
        .sort_values("total_sales", ascending=False),
        "monthly_sales": df.groupby("ORDER_MONTH", as_index=False)
        .agg(total_sales=("SALES", "sum"), orders=("ORDERNUMBER", "nunique"))
        .sort_values("ORDER_MONTH"),
        "deal_size_mix": df.groupby("DEALSIZE", as_index=False)
        .agg(total_sales=("SALES", "sum"), line_items=("ORDERNUMBER", "count"))
        .sort_values("total_sales", ascending=False),
        "status_sales": df.groupby("STATUS", as_index=False)
        .agg(total_sales=("SALES", "sum"), line_items=("ORDERNUMBER", "count"))
        .sort_values("total_sales", ascending=False),
    }

    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    return tables

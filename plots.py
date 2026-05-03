"""Visualization functions."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def create_plots(df: pd.DataFrame, output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_productline_revenue(output_dir, tables["productline_performance"])
    plot_top_countries(output_dir, tables["country_performance"])
    plot_monthly_sales(output_dir, tables["monthly_sales"])
    plot_deal_size_boxplot(df, output_dir)
    plot_correlation_heatmap(df, output_dir)


def plot_productline_revenue(output_dir: Path, product_table: pd.DataFrame) -> None:
    product_table = product_table.sort_values("total_sales")
    plt.figure(figsize=(10, 5))
    plt.barh(product_table["PRODUCTLINE"], product_table["total_sales"], color="#2a9d8f")
    plt.title("Revenue by Product Line")
    plt.xlabel("Total Sales")
    plt.ylabel("Product Line")
    plt.tight_layout()
    plt.savefig(output_dir / "revenue_by_productline.png", dpi=160)
    plt.close()


def plot_top_countries(output_dir: Path, country_table: pd.DataFrame) -> None:
    top_countries = country_table.head(10).sort_values("total_sales")
    plt.figure(figsize=(10, 5))
    plt.barh(top_countries["COUNTRY"], top_countries["total_sales"], color="#457b9d")
    plt.title("Top 10 Countries by Revenue")
    plt.xlabel("Total Sales")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.savefig(output_dir / "top_countries_by_revenue.png", dpi=160)
    plt.close()


def plot_monthly_sales(output_dir: Path, monthly_table: pd.DataFrame) -> None:
    monthly = monthly_table.copy()
    monthly["ORDER_MONTH"] = pd.to_datetime(monthly["ORDER_MONTH"])
    plt.figure(figsize=(11, 5))
    plt.plot(monthly["ORDER_MONTH"], monthly["total_sales"], marker="o", color="#e76f51")
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_sales_trend.png", dpi=160)
    plt.close()


def plot_deal_size_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    labels = ["Small", "Medium", "Large"]
    box_data = [df.loc[df["DEALSIZE"] == label, "SALES"] for label in labels]
    plt.boxplot(box_data, labels=labels, patch_artist=True)
    plt.title("Sales Distribution by Deal Size")
    plt.xlabel("Deal Size")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.savefig(output_dir / "sales_by_dealsize_boxplot.png", dpi=160)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    numeric_columns = ["QUANTITYORDERED", "PRICEEACH", "SALES", "MSRP", "ORDER_YEAR", "ORDER_MONTH_NUM"]
    corr = df[numeric_columns].corr(numeric_only=True)
    corr.to_csv(output_dir / "correlation_matrix.csv")

    plt.figure(figsize=(8, 6))
    image = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    for row_idx, row_name in enumerate(corr.index):
        for col_idx, col_name in enumerate(corr.columns):
            plt.text(col_idx, row_idx, f"{corr.loc[row_name, col_name]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=160)
    plt.close()

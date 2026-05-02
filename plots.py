"""Visualization functions for the EDA project."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.style.use("ggplot")


def plot_sales_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["SALES"], bins=35, color="#2a9d8f", edgecolor="white")
    ax.set_title("Sales Order Value Distribution")
    ax.set_xlabel("Sales")
    ax.set_ylabel("Order lines")
    plt.tight_layout()
    plt.savefig(output_dir / "sales_distribution.png", dpi=160)
    plt.close()


def plot_sales_by_productline(df: pd.DataFrame, output_dir: Path) -> None:
    product_sales = (
        df.groupby("PRODUCTLINE")["SALES"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(product_sales["PRODUCTLINE"], product_sales["SALES"], color="#457b9d")
    ax.invert_yaxis()
    ax.set_title("Revenue by Product Line")
    ax.set_xlabel("Total sales")
    ax.set_ylabel("Product line")
    plt.tight_layout()
    plt.savefig(output_dir / "sales_by_productline.png", dpi=160)
    plt.close()


def plot_sales_by_country(df: pd.DataFrame, output_dir: Path) -> None:
    country_sales = (
        df.groupby("COUNTRY")["SALES"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(country_sales["COUNTRY"], country_sales["SALES"], color="#e76f51")
    ax.invert_yaxis()
    ax.set_title("Top 10 Countries by Revenue")
    ax.set_xlabel("Total sales")
    ax.set_ylabel("Country")
    plt.tight_layout()
    plt.savefig(output_dir / "top_countries_by_sales.png", dpi=160)
    plt.close()


def plot_monthly_trend(df: pd.DataFrame, output_dir: Path) -> None:
    monthly = (
        df.dropna(subset=["ORDER_YEAR_MONTH"])
        .groupby("ORDER_YEAR_MONTH")["SALES"]
        .sum()
        .reset_index()
    )
    monthly["ORDER_YEAR_MONTH"] = monthly["ORDER_YEAR_MONTH"].astype(str)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly["ORDER_YEAR_MONTH"], monthly["SALES"], marker="o", color="#264653")
    ax.set_title("Monthly Sales Trend")
    ax.set_xlabel("Order month")
    ax.set_ylabel("Total sales")
    ax.tick_params(axis="x", rotation=60)
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_sales_trend.png", dpi=160)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr().round(3)
    corr.to_csv(output_dir / "correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)

    for row_index, row_name in enumerate(corr.index):
        for col_index, col_name in enumerate(corr.columns):
            ax.text(
                col_index,
                row_index,
                f"{corr.loc[row_name, col_name]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Numeric Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=160)
    plt.close()
    return corr


def plot_dealsize_mix(df: pd.DataFrame, output_dir: Path) -> None:
    deal_counts = df["DEALSIZE"].value_counts().reset_index()
    deal_counts.columns = ["DEALSIZE", "count"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(deal_counts["DEALSIZE"], deal_counts["count"], color="#f4a261")
    ax.set_title("Deal Size Mix")
    ax.set_xlabel("Deal size")
    ax.set_ylabel("Order lines")
    plt.tight_layout()
    plt.savefig(output_dir / "deal_size_mix.png", dpi=160)
    plt.close()


def plot_sales_vs_quantity(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Small": "#2a9d8f", "Medium": "#457b9d", "Large": "#e76f51"}

    for deal_size, group in df.groupby("DEALSIZE"):
        ax.scatter(
            group["QUANTITYORDERED"],
            group["SALES"],
            label=deal_size,
            alpha=0.7,
            color=colors.get(deal_size, "#6c757d"),
        )

    ax.set_title("Sales vs Quantity Ordered")
    ax.set_xlabel("Quantity ordered")
    ax.set_ylabel("Sales")
    ax.legend(title="Deal size")
    plt.tight_layout()
    plt.savefig(output_dir / "sales_vs_quantity.png", dpi=160)
    plt.close()


def create_all_plots(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    plot_sales_distribution(df, output_dir)
    plot_sales_by_productline(df, output_dir)
    plot_sales_by_country(df, output_dir)
    plot_monthly_trend(df, output_dir)
    corr = plot_correlation_heatmap(df, output_dir)
    plot_dealsize_mix(df, output_dir)
    plot_sales_vs_quantity(df, output_dir)
    return corr

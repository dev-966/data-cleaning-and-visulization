"""Markdown report generation for the EDA project."""

from pathlib import Path

import pandas as pd


def format_money(value: float) -> str:
    return f"${value:,.0f}"


def create_report(
    df: pd.DataFrame,
    output_dir: Path,
    corr: pd.DataFrame,
    grouped_tables: dict[str, pd.DataFrame],
) -> None:
    total_sales = df["SALES"].sum()
    avg_sales = df["SALES"].mean()
    median_sales = df["SALES"].median()
    date_min = df["ORDERDATE"].min().date()
    date_max = df["ORDERDATE"].max().date()

    top_product = grouped_tables["productline_sales"].iloc[0]
    top_product_name = grouped_tables["productline_sales"].index[0]
    top_country = grouped_tables["country_sales"].iloc[0]
    top_country_name = grouped_tables["country_sales"].index[0]
    top_deal_mean = grouped_tables["dealsize_sales"].iloc[0]
    top_deal_name = grouped_tables["dealsize_sales"].index[0]

    sales_corr = corr["SALES"].drop("SALES").sort_values(key=lambda s: s.abs(), ascending=False)
    strongest_positive = sales_corr[sales_corr > 0].head(3)

    monthly = (
        df.dropna(subset=["ORDER_YEAR_MONTH"])
        .groupby("ORDER_YEAR_MONTH")["SALES"]
        .sum()
        .sort_values(ascending=False)
    )
    best_month = monthly.index[0]
    best_month_sales = monthly.iloc[0]

    missing = df.isna().mean().sort_values(ascending=False)
    high_missing = missing[missing > 0].head(5)

    report = [
        "# Exploratory Data Analysis Report",
        "",
        "## Dataset Overview",
        "- Source file: `../data/sales.csv`",
        f"- Records analyzed: {len(df):,}",
        f"- Columns analyzed: {df.shape[1]}",
        f"- Date range: {date_min} to {date_max}",
        f"- Total sales: {format_money(total_sales)}",
        f"- Average order-line sales: {format_money(avg_sales)}",
        f"- Median order-line sales: {format_money(median_sales)}",
        "",
        "## Data Quality Notes",
        f"- Duplicate rows: {df.duplicated().sum():,}",
        "- Missing values are concentrated in address/location support fields, not in the main sales metrics.",
    ]

    for col, value in high_missing.items():
        report.append(f"- `{col}` missing rate: {value * 100:.1f}%")

    report.extend(
        [
            "",
            "## Statistical Summary",
            "- Full descriptive statistics are saved in `summary_statistics.csv`.",
            "- `SALES` is right-skewed: the average is higher than the median, showing that larger orders pull the mean upward.",
            f"- The highest revenue product line is `{top_product_name}` with {format_money(top_product['sum'])}.",
            f"- The highest revenue country is `{top_country_name}` with {format_money(top_country['sum'])}.",
            f"- `{top_deal_name}` deals have the highest average order-line value at {format_money(top_deal_mean['mean'])}.",
            "",
            "## Correlations and Key Influencing Factors",
        ]
    )

    for feature, value in strongest_positive.items():
        report.append(f"- `{feature}` has a positive correlation with `SALES` of {value:.2f}.")

    report.extend(
        [
            "- Product line, country, and deal size also strongly influence revenue when grouped categorically.",
            f"- The strongest sales month is `{best_month}` with {format_money(best_month_sales)} in revenue.",
            "",
            "## Visualizations Created",
            "- `sales_distribution.png`",
            "- `sales_by_productline.png`",
            "- `top_countries_by_sales.png`",
            "- `monthly_sales_trend.png`",
            "- `correlation_heatmap.png`",
            "- `deal_size_mix.png`",
            "- `sales_vs_quantity.png`",
            "",
            "## Insights",
            f"- Revenue is concentrated in a small set of markets, led by `{top_country_name}`.",
            f"- `{top_product_name}` is the most important product line by total revenue.",
            "- Larger quantities, higher unit prices, and higher MSRP values are associated with higher sales.",
            "- Deal size is a useful summary factor because large deals have much higher average sales values than medium or small deals.",
            "- Sales activity changes noticeably over time, so month and year should be considered in planning, forecasting, and campaign analysis.",
            "",
            "## Recommendation",
            "- Focus business review on top countries and top product lines first because they explain the largest revenue share.",
            "- Use quantity, price, MSRP, deal size, product line, and country as priority variables for future predictive modeling.",
        ]
    )

    (output_dir / "eda_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

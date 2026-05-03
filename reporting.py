"""Markdown report generation."""

from pathlib import Path

import pandas as pd


def money(value: float) -> str:
    return f"${value:,.0f}"


def create_report(
    df: pd.DataFrame,
    output_dir: Path,
    tables: dict[str, pd.DataFrame],
    metrics: pd.DataFrame,
    best_model: str,
) -> None:
    productline = tables["productline_performance"].iloc[0]
    country = tables["country_performance"].iloc[0]
    deal = tables["deal_size_mix"].iloc[0]
    total_sales = df["SALES"].sum()
    total_orders = df["ORDERNUMBER"].nunique()
    avg_sale = df["SALES"].mean()
    best_metric = metrics.iloc[0]

    report = f"""# Real-World Data Project: Retail Sales Analysis and Prediction

## Domain and Dataset

This project uses a retail sales order dataset from `../data/sales.csv`. The data contains {len(df):,} order line items across {total_orders:,} orders, including product line, country, order status, date, price, quantity, sales value, and deal size.

## Objective

- Analyze real-world retail sales performance.
- Identify product, country, time, and deal-size patterns.
- Build a prediction model for `DEALSIZE` using order attributes.
- Provide a Streamlit app where a user can enter order details and predict deal size.
- Present findings with visualizations and conclusions.

## Data Preparation

- Parsed `ORDERDATE` into year and month fields.
- Checked missing values, data types, and unique values.
- Created grouped business summaries by product line, country, month, status, and deal size.
- Excluded direct customer identity fields from the model to keep prediction focused on business attributes.

## Key Findings

- Total sales in the dataset are {money(total_sales)}.
- The strongest product line by revenue is **{productline["PRODUCTLINE"]}** with {money(productline["total_sales"])}.
- The top country by revenue is **{country["COUNTRY"]}** with {money(country["total_sales"])}.
- The largest revenue share comes from **{deal["DEALSIZE"]}** deals with {money(deal["total_sales"])}.
- Average line-item sales value is {money(avg_sale)}.

## Prediction Result

The best model is **{best_model}** and is saved as `best_dealsize_model.pkl`.

| Model | Accuracy | Weighted F1 |
|---|---:|---:|
"""

    for _, row in metrics.iterrows():
        report += f'| {row["model"]} | {row["accuracy"]:.3f} | {row["weighted_f1"]:.3f} |\n'

    report += f"""
The strongest model achieved a weighted F1 score of **{best_metric["weighted_f1"]:.3f}**, which shows the selected order attributes are useful for estimating whether a sale is Small, Medium, or Large.

## Visualizations Generated

- `revenue_by_productline.png`
- `top_countries_by_revenue.png`
- `monthly_sales_trend.png`
- `sales_by_dealsize_boxplot.png`
- `correlation_heatmap.png`
- `confusion_matrix.png`

## Streamlit Prediction App

Run the app with:

```bash
streamlit run streamlit_app.py
```

The app loads `outputs/best_dealsize_model.pkl`, accepts order details from the user, and predicts deal size.

## Conclusion

This project applies an end-to-end data science workflow to a retail dataset: loading, cleaning, exploration, visualization, modeling, evaluation, saving the model, and building a small prediction interface. The analysis shows that revenue is concentrated in specific product lines and countries, while the prediction model demonstrates how order attributes can support practical retail decision-making such as deal segmentation and sales planning.
"""
    (output_dir / "project_report.md").write_text(report, encoding="utf-8")

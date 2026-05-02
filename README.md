# Exploratory Data Analysis Project

This project analyzes the sales order dataset in `../data/sales.csv` to uncover patterns, correlations, and key factors that influence revenue.

## Objective

- Use statistical summaries and visualizations.
- Identify correlations and important business factors.
- Present findings in a structured report.

## How To Run

From this folder:

```bash
python3 eda_analysis.py
```

The script creates the `outputs/` folder with the full submission artifacts.

## Generated Outputs

- `eda_report.md` - structured EDA report with findings and recommendations.
- `summary_statistics.csv` - descriptive statistics for all columns.
- `data_quality.csv` - data types, missing values, and unique-value counts.
- `correlation_matrix.csv` - numeric correlation table.
- `productline_sales.csv`, `country_sales.csv`, `dealsize_sales.csv`, `year_sales.csv`, `status_sales.csv` - grouped summaries.
- `sales_distribution.png`
- `sales_by_productline.png`
- `top_countries_by_sales.png`
- `monthly_sales_trend.png`
- `correlation_heatmap.png`
- `deal_size_mix.png`
- `sales_vs_quantity.png`

## Project Files

- `eda_analysis.py` - main script that runs the full EDA workflow.
- `data_loader.py` - loads the CSV and prepares date columns.
- `summaries.py` - creates statistical, quality, and grouped summary tables.
- `plots.py` - creates all visualizations and the correlation matrix.
- `reporting.py` - builds the final Markdown report.

## Main Questions Explored

- Which product lines and countries generate the most revenue?
- How does sales performance change over time?
- Which numeric features correlate most with sales?
- How does deal size relate to sales value?
- Are there data quality issues that should be considered before modeling?

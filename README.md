# Real-World Data Project: Retail Sales

This folder contains an end-to-end data science project for the retail domain. It uses the sales order dataset at `../data/sales.csv` to analyze business performance and predict deal size.

## Assignment Fit

- **Dataset:** Retail sales order data.
- **Analysis:** Sales by product line, country, month, order status, and deal size.
- **Prediction:** Classification model to predict `DEALSIZE`.
- **Visualizations:** Revenue charts, trend line, correlation heatmap, boxplot, and confusion matrix.
- **Conclusion:** Generated in `outputs/project_report.md`.

## How To Run

From this folder:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python real_world_retail_project.py
```

If the required packages are already installed globally, this also works:

```bash
python3 real_world_retail_project.py
```

## Run The Prediction App

After generating the model, start the Streamlit app:

```bash
.venv/bin/streamlit run streamlit_app.py
```

The app lets a user enter quantity, price, MSRP, date fields, product line, country, territory, and order status. It loads `outputs/best_dealsize_model.pkl` and predicts whether the deal is `Small`, `Medium`, or `Large`.

## Generated Outputs

The script creates an `outputs/` folder containing:

- `project_report.md` - final write-up with findings and conclusions.
- `best_dealsize_model.pkl` - saved trained model pipeline for prediction.
- `data_quality.csv` - missing values, data types, and unique-value counts.
- `summary_statistics.csv` - descriptive statistics.
- `productline_performance.csv`
- `country_performance.csv`
- `monthly_sales.csv`
- `deal_size_mix.csv`
- `status_sales.csv`
- `model_metrics.csv`
- `classification_report.txt`
- `revenue_by_productline.png`
- `top_countries_by_revenue.png`
- `monthly_sales_trend.png`
- `sales_by_dealsize_boxplot.png`
- `correlation_heatmap.png`
- `confusion_matrix.png`

## Project Files

- `real_world_retail_project.py` - main runner for the complete workflow.
- `config.py` - shared constants and feature names.
- `data_loader.py` - dataset loading and date feature preparation.
- `summaries.py` - data quality, descriptive, and business summary tables.
- `plots.py` - visualization generation.
- `modeling.py` - model training, evaluation, saving, loading, and prediction.
- `reporting.py` - Markdown report generation.
- `streamlit_app.py` - user input app for deal-size prediction.

## Main Questions

- Which product lines generate the most revenue?
- Which countries are the strongest markets?
- How does revenue change over time?
- How does deal size relate to sales value?
- Can order attributes predict whether a deal is Small, Medium, or Large?

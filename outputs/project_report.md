# Real-World Data Project: Retail Sales Analysis and Prediction

## Domain and Dataset

This project uses a retail sales order dataset from `../data/sales.csv`. The data contains 2,823 order line items across 307 orders, including product line, country, order status, date, price, quantity, sales value, and deal size.

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

- Total sales in the dataset are $10,032,629.
- The strongest product line by revenue is **Classic Cars** with $3,919,616.
- The top country by revenue is **USA** with $3,627,983.
- The largest revenue share comes from **Medium** deals with $6,087,432.
- Average line-item sales value is $3,554.

## Prediction Result

The best model is **Random Forest** and is saved as `best_dealsize_model.pkl`.

| Model | Accuracy | Weighted F1 |
|---|---:|---:|
| Random Forest | 0.885 | 0.886 |
| Logistic Regression | 0.857 | 0.868 |

The strongest model achieved a weighted F1 score of **0.886**, which shows the selected order attributes are useful for estimating whether a sale is Small, Medium, or Large.

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

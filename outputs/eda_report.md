# Exploratory Data Analysis Report

## Dataset Overview
- Source file: `../data/sales.csv`
- Records analyzed: 2,823
- Columns analyzed: 27
- Date range: 2003-01-06 to 2005-05-31
- Total sales: $10,032,629
- Average order-line sales: $3,554
- Median order-line sales: $3,185

## Data Quality Notes
- Duplicate rows: 0
- Missing values are concentrated in address/location support fields, not in the main sales metrics.
- `ADDRESSLINE2` missing rate: 89.3%
- `STATE` missing rate: 52.6%
- `TERRITORY` missing rate: 38.0%
- `POSTALCODE` missing rate: 2.7%

## Statistical Summary
- Full descriptive statistics are saved in `summary_statistics.csv`.
- `SALES` is right-skewed: the average is higher than the median, showing that larger orders pull the mean upward.
- The highest revenue product line is `Classic Cars` with $3,919,616.
- The highest revenue country is `USA` with $3,627,983.
- `Large` deals have the highest average order-line value at $8,294.

## Correlations and Key Influencing Factors
- `PRICEEACH` has a positive correlation with `SALES` of 0.66.
- `MSRP` has a positive correlation with `SALES` of 0.64.
- `QUANTITYORDERED` has a positive correlation with `SALES` of 0.55.
- Product line, country, and deal size also strongly influence revenue when grouped categorically.
- The strongest sales month is `2004-11` with $1,089,048 in revenue.

## Visualizations Created
- `sales_distribution.png`
- `sales_by_productline.png`
- `top_countries_by_sales.png`
- `monthly_sales_trend.png`
- `correlation_heatmap.png`
- `deal_size_mix.png`
- `sales_vs_quantity.png`

## Insights
- Revenue is concentrated in a small set of markets, led by `USA`.
- `Classic Cars` is the most important product line by total revenue.
- Larger quantities, higher unit prices, and higher MSRP values are associated with higher sales.
- Deal size is a useful summary factor because large deals have much higher average sales values than medium or small deals.
- Sales activity changes noticeably over time, so month and year should be considered in planning, forecasting, and campaign analysis.

## Recommendation
- Focus business review on top countries and top product lines first because they explain the largest revenue share.
- Use quantity, price, MSRP, deal size, product line, and country as priority variables for future predictive modeling.

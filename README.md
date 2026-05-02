# Predictive Modeling Using Machine Learning

This folder contains a supervised machine learning project for predicting `DEALSIZE` from the sales order dataset in `../data/sales.csv`.

The project trains and compares:

- Logistic Regression
- Decision Tree
- Random Forest

It evaluates the models with accuracy, weighted F1 score, macro ROC-AUC, a confusion matrix, and one-vs-rest ROC curves.

## Quick Start

From this folder:

```bash
python predictive_model.py
```

The script writes results to `p2/outputs/`:

- `metrics.csv` - model comparison table
- `model_report.md` - classification report and summary
- `best_model.pkl` - dumped best model pipeline
- `confusion_matrix.png` - confusion matrix for the best model
- `roc_curves.png` - multiclass ROC curves for the best model
- `feature_importance.png` - top feature importances when supported

## Project Files

- `predictive_model.py` - main training script
- `config.py` - shared feature list and random seed
- `data_utils.py` - CSV loading and target/feature selection
- `models.py` - preprocessing, model creation, evaluation, and model dump/load helpers
- `visualize.py` - confusion matrix, ROC curve, and feature importance plots
- `reporting.py` - markdown report generation
- `predict_saved_model.py` - loads `best_model.pkl` and predicts on a CSV

## Use The Dumped Model

After training, run:

```bash
python predict_saved_model.py
```

This loads `outputs/best_model.pkl` and writes `outputs/predictions.csv` with a new `PREDICTED_DEALSIZE` column.

## Notes

By default, the script excludes the `SALES` column. `DEALSIZE` is closely related to sales value, so including `SALES` would make the result look unrealistically strong.

To include it for demonstration:

```bash
python predictive_model.py --include-sales
```

You can also choose another CSV or target column:

```bash
python predictive_model.py --input ../data/sales.csv --target DEALSIZE --output outputs
```

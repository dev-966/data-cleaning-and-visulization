# Predictive Modeling Report

Best model: Decision Tree

## Model Metrics

```text
              model  accuracy  weighted_f1  macro_roc_auc_ovr
      Decision Tree    0.8980       0.8931             0.9410
      Random Forest    0.8810       0.8820             0.9714
Logistic Regression    0.8569       0.8675             0.9643
```

## Classification Report

```text
              precision    recall  f1-score   support

       Large       0.75      0.38      0.51        39
      Medium       0.86      0.95      0.90       346
       Small       0.95      0.91      0.93       321

    accuracy                           0.90       706
   macro avg       0.85      0.75      0.78       706
weighted avg       0.90      0.90      0.89       706

```

Generated files:

- `metrics.csv`
- `best_model.pkl`
- `confusion_matrix.png`
- `roc_curves.png`
- `feature_importance.png` when the best model supports it
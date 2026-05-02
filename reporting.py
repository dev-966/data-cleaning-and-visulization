"""Report generation for model results."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


def save_report(model: Pipeline, metrics: pd.DataFrame, best_name: str, x_test, y_test, output_dir: Path) -> None:
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions)
    metrics_table = metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}")
    report_text = [
        "# Predictive Modeling Report",
        "",
        f"Best model: {best_name}",
        "",
        "## Model Metrics",
        "",
        "```text",
        metrics_table,
        "```",
        "",
        "## Classification Report",
        "",
        "```text",
        report,
        "```",
        "",
        "Generated files:",
        "",
        "- `metrics.csv`",
        "- `best_model.pkl`",
        "- `confusion_matrix.png`",
        "- `roc_curves.png`",
        "- `feature_importance.png` when the best model supports it",
    ]
    (output_dir / "model_report.md").write_text("\n".join(report_text), encoding="utf-8")

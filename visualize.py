"""Plots for model evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize


plt.style.use("seaborn-v0_8-whitegrid")


def plot_confusion_matrix(model: Pipeline, x_test, y_test, output_dir: Path) -> None:
    predictions = model.predict(x_test)
    labels = list(model.classes_)
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - Best Model")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def plot_roc_curves(model: Pipeline, x_test, y_test, output_dir: Path) -> None:
    probabilities = model.predict_proba(x_test)
    classes = list(model.classes_)
    encoded_y = label_binarize(y_test, classes=classes)

    fig, ax = plt.subplots(figsize=(7, 5))
    for index, label in enumerate(classes):
        RocCurveDisplay.from_predictions(
            encoded_y[:, index],
            probabilities[:, index],
            name=f"{label}",
            ax=ax,
        )
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_title("One-vs-Rest ROC Curves - Best Model")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=160)
    plt.close(fig)


def plot_feature_importance(model: Pipeline, output_dir: Path) -> None:
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return

    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    importance = (
        pd.DataFrame({"feature": feature_names, "importance": estimator.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(importance["feature"], importance["importance"], color="#4C78A8")
    ax.invert_yaxis()
    ax.set_title("Top Feature Importances - Best Model")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=160)
    plt.close(fig)

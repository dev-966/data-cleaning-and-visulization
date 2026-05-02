#!/usr/bin/env python3
"""Train, evaluate, and save the best supervised model."""

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from config import RANDOM_STATE
from data_utils import ensure_dir, read_csv_with_fallback, select_modeling_data
from models import dump_model, evaluate_models, model_candidates
from reporting import save_report
from visualize import plot_confusion_matrix, plot_feature_importance, plot_roc_curves


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate supervised models for deal size prediction.")
    parser.add_argument("--input", default="../data/sales.csv", help="Input CSV path.")
    parser.add_argument("--target", default="DEALSIZE", help="Target column to predict.")
    parser.add_argument("--output", default="outputs", help="Directory for metrics, plots, and saved model.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Test split size.")
    parser.add_argument(
        "--include-sales",
        action="store_true",
        help="Include SALES as a feature. This is useful for demos but can leak the answer.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_path = (script_dir / args.input).resolve()
    output_dir = (script_dir / args.output).resolve()
    ensure_dir(output_dir)

    df = read_csv_with_fallback(input_path)
    x, y = select_modeling_data(df, args.target, include_sales=args.include_sales)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = model_candidates(x_train)
    metrics, best_name = evaluate_models(models, x_train, x_test, y_train, y_test)
    best_model = models[best_name]

    metrics.to_csv(output_dir / "metrics.csv", index=False)
    dump_model(best_model, output_dir / "best_model.pkl")
    plot_confusion_matrix(best_model, x_test, y_test, output_dir)
    plot_roc_curves(best_model, x_test, y_test, output_dir)
    plot_feature_importance(best_model, output_dir)
    save_report(best_model, metrics, best_name, x_test, y_test, output_dir)

    print(f"Trained {len(models)} models on {len(x_train)} rows and tested on {len(x_test)} rows.")
    print(f"Best model: {best_name}")
    print(metrics.to_string(index=False))
    print(f"Saved best model to: {output_dir / 'best_model.pkl'}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Load the dumped model and make predictions on a CSV file."""

import argparse
from pathlib import Path

from data_utils import read_csv_with_fallback
from models import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict deal sizes with the saved model.")
    parser.add_argument("--model", default="outputs/best_model.pkl", help="Saved pickle model path.")
    parser.add_argument("--input", default="../data/sales.csv", help="CSV file to predict.")
    parser.add_argument("--output", default="outputs/predictions.csv", help="Output CSV with predictions.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    model_path = (script_dir / args.model).resolve()
    input_path = (script_dir / args.input).resolve()
    output_path = (script_dir / args.output).resolve()

    model = load_model(model_path)
    df = read_csv_with_fallback(input_path)
    predictions = model.predict(df)

    df_out = df.copy()
    df_out["PREDICTED_DEALSIZE"] = predictions
    df_out.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()

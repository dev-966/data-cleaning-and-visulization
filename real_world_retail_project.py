#!/usr/bin/env python3
"""Run the complete retail sales analysis and prediction workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import DEFAULT_INPUT, DEFAULT_OUTPUT
from data_loader import ensure_output_dir, load_data
from modeling import train_models
from plots import create_plots
from reporting import create_report
from summaries import save_business_tables, save_data_quality


def run_project(input_path: Path, output_dir: Path) -> None:
    ensure_output_dir(output_dir)
    df = load_data(input_path)
    save_data_quality(df, output_dir)
    tables = save_business_tables(df, output_dir)
    create_plots(df, output_dir, tables)
    metrics, best_model, _ = train_models(df, output_dir)
    create_report(df, output_dir, tables, metrics, best_model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a retail sales analysis and prediction project.")
    parser.add_argument("--input", default=DEFAULT_INPUT, type=Path, help="Path to the retail sales CSV.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path, help="Folder for generated artifacts.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_path = (script_dir / args.input).resolve()
    output_dir = (script_dir / args.output).resolve()

    run_project(input_path, output_dir)
    print(f"Project complete. Outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()

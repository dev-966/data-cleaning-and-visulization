from __future__ import annotations

import argparse
from pathlib import Path

from data_loader import load_sales_data
from plots import create_all_plots
from reporting import create_report
from summaries import (
    analyze_data_quality,
    build_grouped_tables,
    ensure_output_dir,
    save_summary_statistics,
)


def run_eda(input_path: Path, output_dir: Path) -> None:
    ensure_output_dir(output_dir)

    df = load_sales_data(input_path)
    save_summary_statistics(df, output_dir)
    analyze_data_quality(df, output_dir)

    corr = create_all_plots(df, output_dir)
    grouped_tables = build_grouped_tables(df, output_dir)
    create_report(df, output_dir, corr, grouped_tables)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA on the sales order dataset.")
    parser.add_argument(
        "--input",
        default="../data/sales.csv",
        type=Path,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        type=Path,
        help="Folder where reports, tables, and charts will be saved.",
    )
    args = parser.parse_args()

    run_eda(args.input, args.output)
    print(f"EDA complete. Results saved in: {args.output}")


if __name__ == "__main__":
    main()

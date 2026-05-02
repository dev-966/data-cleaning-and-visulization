#!/usr/bin/env python3
"""Clean and visualize a CSV dataset."""
import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_data(path, encoding=None):
    if encoding:
        return pd.read_csv(path, encoding=encoding)
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        for enc in ("latin-1", "cp1252"):
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                continue
        return pd.read_csv(path, encoding="latin-1", engine="python")


def handle_missing(df, strategy="drop"):
    df_out = df.copy()
    if strategy == "drop":
        return df_out.dropna()
    num_cols = df_out.select_dtypes(include=[np.number]).columns
    if strategy == "mean":
        df_out[num_cols] = df_out[num_cols].fillna(df_out[num_cols].mean())
    elif strategy == "median":
        df_out[num_cols] = df_out[num_cols].fillna(df_out[num_cols].median())
    else:
        df_out = df_out.fillna(method="ffill").fillna(method="bfill")
    cat_cols = df_out.select_dtypes(include=[object]).columns
    for c in cat_cols:
        if df_out[c].isnull().any():
            try:
                df_out[c] = df_out[c].fillna(df_out[c].mode().iloc[0])
            except Exception:
                df_out[c] = df_out[c].fillna("")
    return df_out


def remove_duplicates(df):
    return df.drop_duplicates()


def handle_outliers_iqr(df, method="clip", factor=1.5):
    df_out = df.copy()
    num_cols = df_out.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return df_out
    Q1 = df_out[num_cols].quantile(0.25)
    Q3 = df_out[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    if method == "remove":
        mask = ~((df_out[num_cols] < lower) | (df_out[num_cols] > upper)).any(axis=1)
        return df_out[mask]
    else:
        return df_out.clip(lower=lower, upper=upper, axis=1)


def save_clean(df, output_dir, filename="cleaned.csv"):
    ensure_dir(output_dir)
    out_path = Path(output_dir) / filename
    df.to_csv(out_path, index=False)
    return out_path


def plot_histograms(df, output_dir):
    ensure_dir(output_dir)
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Histogram: {col}")
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"hist_{col}.png")
        plt.close()


def plot_boxplots(df, output_dir):
    ensure_dir(output_dir)
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        plt.figure(figsize=(6, 3))
        sns.boxplot(x=df[col].dropna())
        plt.title(f"Boxplot: {col}")
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"box_{col}.png")
        plt.close()


def plot_corr_heatmap(df, output_dir):
    ensure_dir(output_dir)
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) < 2:
        return
    corr = df[num_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "corr_heatmap.png")
    plt.close()


def summary_stats(df, output_dir):
    ensure_dir(output_dir)
    stats = df.describe(include='all')
    stats.to_csv(Path(output_dir) / "summary_stats.csv")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", default="outputs", help="Output folder to save cleaned data and plots")
    parser.add_argument("--missing", choices=["drop", "mean", "median", "ffill"], default="mean", help="Missing value strategy")
    parser.add_argument("--outliers", choices=["clip", "remove", "none"], default="clip", help="Outlier handling method")
    parser.add_argument("--encoding", default=None, help="Optional file encoding (e.g. latin-1, cp1252)")
    args = parser.parse_args()

    df = load_data(args.input, encoding=args.encoding)
    df = remove_duplicates(df)
    df = handle_missing(df, strategy=args.missing)
    if args.outliers != "none":
        df = handle_outliers_iqr(df, method=args.outliers)

    out = Path(args.output)
    ensure_dir(out)
    cleaned_path = save_clean(df, out, filename="cleaned.csv")
    summary_stats(df, out)
    plot_histograms(df, out)
    plot_boxplots(df, out)
    plot_corr_heatmap(df, out)

    print(f"Cleaned data saved to: {cleaned_path}")
    print(f"Plots and stats saved to folder: {out}")


if __name__ == "__main__":
    main()

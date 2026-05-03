"""Shared configuration for the retail project."""

from pathlib import Path

RANDOM_STATE = 42
TARGET = "DEALSIZE"
MODEL_FILENAME = "best_dealsize_model.pkl"
DEFAULT_INPUT = Path("../data/sales.csv")
DEFAULT_OUTPUT = Path("outputs")

FEATURE_COLUMNS = [
    "QUANTITYORDERED",
    "PRICEEACH",
    "MSRP",
    "QTR_ID",
    "MONTH_ID",
    "YEAR_ID",
    "PRODUCTLINE",
    "COUNTRY",
    "TERRITORY",
    "STATUS",
]

DEAL_SIZE_LABELS = ["Small", "Medium", "Large"]

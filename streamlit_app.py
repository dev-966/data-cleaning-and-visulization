"""Streamlit app for predicting retail deal size."""

from pathlib import Path

import pandas as pd
import streamlit as st

from config import DEFAULT_INPUT, MODEL_FILENAME
from data_loader import load_data
from modeling import load_model, predict_deal_size


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / MODEL_FILENAME
DATA_PATH = (BASE_DIR / DEFAULT_INPUT).resolve()


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    return load_data(DATA_PATH)


@st.cache_resource
def load_saved_model():
    return load_model(MODEL_PATH)


def sorted_options(df: pd.DataFrame, column: str) -> list[str]:
    return sorted(df[column].dropna().astype(str).unique().tolist())


def main() -> None:
    st.set_page_config(page_title="Retail Deal Size Predictor", layout="centered")
    st.title("Retail Deal Size Predictor")

    if not MODEL_PATH.exists():
        st.error("Model file not found. Run `python3 real_world_retail_project.py` first.")
        st.stop()

    df = load_reference_data()
    model = load_saved_model()

    st.subheader("Enter Order Details")

    col1, col2 = st.columns(2)
    with col1:
        quantity = st.number_input("Quantity Ordered", min_value=1, max_value=1000, value=35, step=1)
        price_each = st.number_input("Price Each", min_value=1.0, max_value=500.0, value=95.0, step=1.0)
        msrp = st.number_input("MSRP", min_value=1.0, max_value=500.0, value=100.0, step=1.0)
        year = st.selectbox("Year", sorted(df["YEAR_ID"].dropna().unique().tolist()), index=0)
        quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=0)

    with col2:
        month = st.selectbox("Month", list(range(1, 13)), index=0)
        productline = st.selectbox("Product Line", sorted_options(df, "PRODUCTLINE"))
        country = st.selectbox("Country", sorted_options(df, "COUNTRY"))
        territory = st.selectbox("Territory", sorted_options(df, "TERRITORY"))
        status = st.selectbox("Order Status", sorted_options(df, "STATUS"))

    input_data = {
        "QUANTITYORDERED": quantity,
        "PRICEEACH": price_each,
        "MSRP": msrp,
        "QTR_ID": quarter,
        "MONTH_ID": month,
        "YEAR_ID": year,
        "PRODUCTLINE": productline,
        "COUNTRY": country,
        "TERRITORY": territory,
        "STATUS": status,
    }

    if st.button("Predict Deal Size", type="primary"):
        prediction, probabilities = predict_deal_size(model, input_data)
        st.success(f"Predicted Deal Size: {prediction}")

        if probabilities is not None:
            probabilities["probability"] = probabilities["probability"].round(3)
            st.dataframe(probabilities, use_container_width=True, hide_index=True)
            st.bar_chart(probabilities.set_index("deal_size")["probability"])

        st.caption("Model uses quantity, price, MSRP, date fields, product line, country, territory, and status.")


if __name__ == "__main__":
    main()

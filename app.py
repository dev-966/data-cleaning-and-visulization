import io
import streamlit as st
import pandas as pd
import numpy as np
from clean_and_visualize import handle_missing, remove_duplicates, handle_outliers_iqr, summary_stats
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Data Cleaning & Visualization — Interactive Dashboard")

def read_uploaded_csv(uploaded_file):
    data = uploaded_file.read()
    try:
        return pd.read_csv(io.BytesIO(data))
    except Exception:
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                text = data.decode(enc)
                return pd.read_csv(io.StringIO(text))
            except Exception:
                continue
        return pd.read_csv(io.StringIO(data.decode("latin-1", errors="ignore")))

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        df = read_uploaded_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        st.stop()
    st.dataframe(df.head())
    missing = st.sidebar.selectbox("Missing value strategy", ["drop", "mean", "median", "ffill"], index=1)
    outliers = st.sidebar.selectbox("Outlier handling", ["clip", "remove", "none"], index=0)
    if st.sidebar.button("Run cleaning"):
        df2 = remove_duplicates(df)
        df2 = handle_missing(df2, strategy=missing)
        if outliers != "none":
            df2 = handle_outliers_iqr(df2, method=outliers)
        st.dataframe(df2.head())
        st.dataframe(summary_stats(df2, output_dir=".").T)
        num_cols = df2.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.histplot(df2[col].dropna(), kde=True, ax=ax)
            st.pyplot(fig)

# Data Cleaning & Visualization Project

This project scaffold helps you clean, preprocess, and visualize a raw dataset using Python.

Contents

- `clean_and_visualize.py` — command-line script to clean data and produce plots.
- `app.py` — simple Streamlit dashboard to interactively upload and explore a dataset.
- `requirements.txt` — Python dependencies.
- `notebooks/analysis.ipynb` — starter Jupyter notebook for step-by-step exploration.
- `data/sample_raw.csv` — tiny sample dataset to test the pipeline.

Quick start

1. Create and activate a virtualenv (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the CLI script on your CSV:

```bash
python clean_and_visualize.py --input data/sample_raw.csv --output outputs
```

4. Or run the interactive dashboard:

```bash
streamlit run app.py
```

Next steps

- Replace `data/sample_raw.csv` with your raw CSV file and run the script.
- Tell me if you want me to run the script on your dataset or customize cleaning rules.

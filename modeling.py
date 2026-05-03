"""Model training, evaluation, saving, and loading."""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DEAL_SIZE_LABELS, FEATURE_COLUMNS, MODEL_FILENAME, RANDOM_STATE, TARGET


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLUMNS], df[TARGET]


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_features = x.select_dtypes(include="number").columns.tolist()
    categorical_features = x.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def model_candidates(x: pd.DataFrame) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(x)),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(x)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=10,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def train_models(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, str, str]:
    x, y = select_features(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    rows = []
    best_name = ""
    best_score = -1.0
    best_model = None
    best_predictions = None

    for name, model in model_candidates(x).items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        weighted_f1 = f1_score(y_test, predictions, average="weighted")
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "weighted_f1": weighted_f1,
            }
        )
        if weighted_f1 > best_score:
            best_score = weighted_f1
            best_name = name
            best_model = model
            best_predictions = predictions

    metrics = pd.DataFrame(rows).sort_values("weighted_f1", ascending=False)
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    save_model(best_model, output_dir / MODEL_FILENAME)

    report = classification_report(y_test, best_predictions, labels=DEAL_SIZE_LABELS, zero_division=0)
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    plot_confusion_matrix(y_test, best_predictions, best_name, output_dir)

    return metrics, best_name, report


def save_model(model: Pipeline, output_path: Path) -> None:
    with output_path.open("wb") as file:
        pickle.dump(model, file)


def load_model(model_path: Path) -> Pipeline:
    with model_path.open("rb") as file:
        return pickle.load(file)


def predict_deal_size(model: Pipeline, input_data: dict[str, object]) -> tuple[str, pd.DataFrame | None]:
    row = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
    prediction = model.predict(row)[0]
    probabilities = None

    if hasattr(model.named_steps["model"], "predict_proba"):
        probabilities = pd.DataFrame(
            {
                "deal_size": model.classes_,
                "probability": model.predict_proba(row)[0],
            }
        ).sort_values("probability", ascending=False)

    return prediction, probabilities


def plot_confusion_matrix(y_test: pd.Series, predictions, model_name: str, output_dir: Path) -> None:
    matrix = confusion_matrix(y_test, predictions, labels=DEAL_SIZE_LABELS)
    plt.figure(figsize=(6, 5))
    image = plt.imshow(matrix, cmap="Blues")
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(DEAL_SIZE_LABELS)), DEAL_SIZE_LABELS)
    plt.yticks(range(len(DEAL_SIZE_LABELS)), DEAL_SIZE_LABELS)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            plt.text(col_idx, row_idx, str(matrix[row_idx, col_idx]), ha="center", va="center")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

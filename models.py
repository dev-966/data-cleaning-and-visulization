"""Model pipelines and training logic."""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier

from config import RANDOM_STATE


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_features = x.select_dtypes(include="number").columns.tolist()
    categorical_features = x.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_features),
            ("categorical", categorical_pipe, categorical_features),
        ]
    )


def model_candidates(x: pd.DataFrame) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(x)),
                (
                    "model",
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(x)),
                ("model", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(x)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=10,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate_models(models: dict[str, Pipeline], x_train, x_test, y_train, y_test) -> tuple[pd.DataFrame, str]:
    rows = []
    best_name = None
    best_f1 = -1.0

    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test) if hasattr(model.named_steps["model"], "predict_proba") else None
        roc_auc = None
        if probabilities is not None:
            encoded_y = label_binarize(y_test, classes=model.classes_)
            roc_auc = roc_auc_score(encoded_y, probabilities, average="macro", multi_class="ovr")

        weighted_f1 = f1_score(y_test, predictions, average="weighted")
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "weighted_f1": weighted_f1,
                "macro_roc_auc_ovr": roc_auc,
            }
        )
        if weighted_f1 > best_f1:
            best_f1 = weighted_f1
            best_name = name

    return pd.DataFrame(rows).sort_values("weighted_f1", ascending=False), best_name


def dump_model(model: Pipeline, output_path: Path) -> None:
    with output_path.open("wb") as file:
        pickle.dump(model, file)


def load_model(model_path: Path) -> Pipeline:
    with model_path.open("rb") as file:
        return pickle.load(file)

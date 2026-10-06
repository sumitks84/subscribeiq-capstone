"""Load, normalize, and prepare customer data for the models."""

from pathlib import Path
from typing import IO

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

RAW_DATA_PATH = "data/telco_churn.csv"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
TARGET = "Churn"

_EXPECTED_COLUMNS = {
    column.casefold(): column
    for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["customerID", TARGET]
}

_ENCODED_CATEGORIES = {
    "gender": {0: "Female", 1: "Male"},
    "partner": {0: "No", 1: "Yes"},
    "dependents": {0: "No", 1: "Yes"},
    "phoneservice": {0: "No", 1: "Yes"},
    "multiplelines": {0: "No", 1: "Yes", 2: "No phone service"},
    "internetservice": {0: "No", 1: "DSL", 2: "Fiber optic"},
    "onlinesecurity": {0: "No", 1: "Yes", 2: "No internet service"},
    "onlinebackup": {0: "No", 1: "Yes", 2: "No internet service"},
    "deviceprotection": {0: "No", 1: "Yes", 2: "No internet service"},
    "techsupport": {0: "No", 1: "Yes", 2: "No internet service"},
    "streamingtv": {0: "No", 1: "Yes", 2: "No internet service"},
    "streamingmovies": {0: "No", 1: "Yes", 2: "No internet service"},
    "contract": {0: "Month-to-month", 1: "One year", 2: "Two year"},
    "paperlessbilling": {0: "No", 1: "Yes"},
    "paymentmethod": {
        0: "Electronic check",
        1: "Mailed check",
        2: "Bank transfer (automatic)",
        3: "Credit card (automatic)",
    },
}


def load_raw_data(path: str | Path | IO[bytes] | IO[str] = RAW_DATA_PATH) -> pd.DataFrame:
    """Load a CSV-like file from a path or Streamlit upload."""
    name = str(getattr(path, "name", path)).casefold()
    if name.endswith((".tsv", ".tab")):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Match expected columns without making callers care about casing."""
    normalized = {}
    for column in df.columns:
        expected = _EXPECTED_COLUMNS.get(str(column).strip().casefold())
        if expected is not None:
            if expected in normalized.values():
                raise ValueError(f"Duplicate column after case normalization: {expected}")
            normalized[column] = expected
    return df.rename(columns=normalized)


def _normalize_parameter_values(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize known categorical values, including common numeric encodings."""
    for column, encoded_values in _ENCODED_CATEGORIES.items():
        actual_column = next(
            (name for name in df.columns if str(name).casefold() == column),
            None,
        )
        if actual_column is None:
            continue

        def normalize(value):
            if pd.isna(value):
                return value
            try:
                numeric_value = float(value)
                if numeric_value.is_integer() and int(numeric_value) in encoded_values:
                    return encoded_values[int(numeric_value)]
            except (TypeError, ValueError):
                pass
            text = str(value).strip()
            canonical = {
                str(option).casefold(): option
                for option in encoded_values.values()
            }
            return canonical.get(text.casefold(), text)

        df[actual_column] = df[actual_column].map(normalize)

    if TARGET in df.columns:
        target_values = {"yes": 1, "no": 0, "1": 1, "0": 0}
        df[TARGET] = df[TARGET].map(
            lambda value: target_values.get(str(value).strip().casefold(), value)
            if not pd.isna(value)
            else value
        )
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix known quirks in the Telco dataset:
    - TotalCharges is read as a string with some blank entries for
      customers with 0 tenure; convert to numeric and fill with 0.
    - Drop the customerID column (identifier, not a feature).
    - Encode the target as 0/1.
    """
    df = _normalize_columns(df.copy())
    df = _normalize_parameter_values(df)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build a reusable sklearn ColumnTransformer:
    - scales numeric features (required for K-Means, helps most models)
    - one-hot encodes categorical features
    Fit this once on the training set, then reuse it for inference.
    """
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def compute_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple RFM-style columns used for segmentation and LTV:
    - Recency proxy: inverse of tenure (newer customers = higher "recency risk")
    - Frequency/Monetary proxy: MonthlyCharges and TotalCharges
    - A simple estimated LTV: MonthlyCharges * tenure (what they've
      already been worth) as a baseline before any churn adjustment.
    """
    df = df.copy()
    df["EstimatedLTV"] = df["MonthlyCharges"] * df["tenure"]
    return df


def get_feature_target_split(df: pd.DataFrame):
    """Return (X, y) ready for train/test split."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET] if TARGET in df.columns else None
    return X, y


if __name__ == "__main__":
    raw = load_raw_data()
    cleaned = clean_data(raw)
    cleaned = compute_rfm_features(cleaned)
    print(cleaned.head())
    print(f"Rows: {len(cleaned)}, Churn rate: {cleaned[TARGET].mean():.2%}")

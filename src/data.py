"""
data.py
Loads and cleans the Telco Customer Churn dataset, and builds the
features used by both the segmentation model and the churn model.

Dataset: "Telco Customer Churn" (Kaggle / IBM sample dataset)
Expected columns include: customerID, gender, SeniorCitizen, Partner,
Dependents, tenure, PhoneService, MultipleLines, InternetService,
OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
StreamingTV, StreamingMovies, Contract, PaperlessBilling,
PaymentMethod, MonthlyCharges, TotalCharges, Churn
"""

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


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix known quirks in the Telco dataset:
    - TotalCharges is read as a string with some blank entries for
      customers with 0 tenure; convert to numeric and fill with 0.
    - Drop the customerID column (identifier, not a feature).
    - Encode the target as 0/1.
    """
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if TARGET in df.columns:
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

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

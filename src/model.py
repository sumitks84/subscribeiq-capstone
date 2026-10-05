"""
model.py
Trains and saves:
1. A K-Means segmentation model (customer clusters)
2. A churn prediction model (Logistic Regression baseline + Random
   Forest / XGBoost as the stronger comparison model)
3. SHAP explainability for the chosen churn model

Also provides the inference functions the Streamlit app calls.
"""

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split

try:
    # Works when model.py is imported as part of the src package
    # (e.g. Streamlit running app.py, which does `from src.model import ...`)
    from .data import (
        load_raw_data,
        clean_data,
        compute_rfm_features,
        build_preprocessor,
        get_feature_target_split,
        NUMERIC_FEATURES,
    )
except ImportError:
    # Works when model.py is run directly, e.g. `python src/model.py`
    from data import (
        load_raw_data,
        clean_data,
        compute_rfm_features,
        build_preprocessor,
        get_feature_target_split,
        NUMERIC_FEATURES,
    )

MODEL_DIR = "models"
N_CLUSTERS = 4  # chosen via elbow method / silhouette score — see notebook


def train_segmentation(df: pd.DataFrame, preprocessor):
    """
    Fit K-Means on the preprocessed numeric features only (tenure,
    MonthlyCharges, TotalCharges) — clustering on scaled continuous
    features avoids letting one-hot categorical columns dominate
    the distance calculation.
    """
    numeric_data = preprocessor.named_transformers_["num"].transform(
        df[NUMERIC_FEATURES]
    )
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    kmeans.fit(numeric_data)
    return kmeans


def label_segments(df: pd.DataFrame, cluster_labels: np.ndarray) -> dict:
    """
    Map numeric cluster IDs to business-friendly names based on each
    cluster's average tenure and spend. Recompute this after training
    since cluster order isn't guaranteed to be stable.
    """
    df = df.copy()
    df["cluster"] = cluster_labels

    summary = df.groupby("cluster")[["tenure", "MonthlyCharges"]].mean()
    summary = summary.sort_values("tenure")

    names = [
        "At-risk new customer",
        "Budget month-to-month",
        "Growing / mid-tenure",
        "High-value loyal",
    ]
    # Assign names in tenure order; adjust N_CLUSTERS and this list
    # together if you change cluster count.
    label_map = {cluster_id: names[i] for i, cluster_id in enumerate(summary.index)}
    return label_map


def train_churn_model(X_train, y_train, model_type: str = "random_forest"):
    """
    Train the churn classifier. Class weighting handles the churn
    class imbalance (~27% positive) without needing synthetic
    oversampling — simpler to explain in Q&A than SMOTE.
    """
    if model_type == "logistic_regression":
        model = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )
    else:
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
        )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Report precision/recall/F1/ROC-AUC rather than plain accuracy,
    since accuracy is misleading on an imbalanced churn label.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)

    return {"classification_report": report, "roc_auc": auc}


def get_shap_explainer(model, X_background):
    """
    Build a SHAP explainer for the trained model. TreeExplainer is
    used for Random Forest / XGBoost; swap to LinearExplainer if the
    final model is Logistic Regression.
    """
    explainer = shap.TreeExplainer(model, X_background)
    return explainer


def explain_prediction(explainer, X_row):
    """Return SHAP values for a single customer's prediction."""
    shap_values = explainer.shap_values(X_row)
    return shap_values


def estimate_ltv(row: pd.Series, churn_probability: float) -> float:
    """
    Simple LTV estimate: current monthly value projected forward,
    discounted by churn risk. Not a full survival-analysis model —
    this is intentionally simple so it's easy to explain and defend.

    LTV = MonthlyCharges * expected_remaining_months
    expected_remaining_months is approximated as 1 / churn_probability,
    capped at 60 months to avoid unrealistic values for very low-risk
    customers.
    """
    churn_probability = max(churn_probability, 1 / 60)  # avoid div-by-zero
    expected_remaining_months = min(1 / churn_probability, 60)
    return round(row["MonthlyCharges"] * expected_remaining_months, 2)


def save_artifacts(preprocessor, kmeans, churn_model, segment_labels):
    import os

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(preprocessor, f"{MODEL_DIR}/preprocessor.joblib")
    joblib.dump(kmeans, f"{MODEL_DIR}/kmeans.joblib")
    joblib.dump(churn_model, f"{MODEL_DIR}/churn_model.joblib")
    joblib.dump(segment_labels, f"{MODEL_DIR}/segment_labels.joblib")


def load_artifacts():
    preprocessor = joblib.load(f"{MODEL_DIR}/preprocessor.joblib")
    kmeans = joblib.load(f"{MODEL_DIR}/kmeans.joblib")
    churn_model = joblib.load(f"{MODEL_DIR}/churn_model.joblib")
    segment_labels = joblib.load(f"{MODEL_DIR}/segment_labels.joblib")
    return preprocessor, kmeans, churn_model, segment_labels


if __name__ == "__main__":
    raw = load_raw_data()
    df = clean_data(raw)
    df = compute_rfm_features(df)

    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    kmeans = train_segmentation(X_train, preprocessor)
    cluster_labels = kmeans.predict(
        preprocessor.named_transformers_["num"].transform(X_train[NUMERIC_FEATURES])
    )
    segment_labels = label_segments(X_train, cluster_labels)

    churn_model = train_churn_model(X_train_enc, y_train, model_type="random_forest")
    results = evaluate_model(churn_model, X_test_enc, y_test)

    print("ROC-AUC:", results["roc_auc"])
    print(pd.DataFrame(results["classification_report"]).T)

    save_artifacts(preprocessor, kmeans, churn_model, segment_labels)
    print("Artifacts saved to", MODEL_DIR)

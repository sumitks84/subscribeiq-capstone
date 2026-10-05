"""
app.py
SubscribeIQ — Streamlit app for churn risk + customer lifetime value.

Run with:
    streamlit run app.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.data import (
    clean_data,
    compute_rfm_features,
    get_feature_target_split,
    NUMERIC_FEATURES as NUM_COLS,
    CATEGORICAL_FEATURES,
)
from src.model import (
    load_artifacts,
    get_shap_explainer,
    estimate_ltv,
    NUMERIC_FEATURES,
)

st.set_page_config(page_title="SubscribeIQ", layout="wide")

st.title("SubscribeIQ")
st.caption("Customer churn risk and lifetime value, for non-technical reviewers.")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


@st.cache_resource
def get_explainer(_churn_model, _background_enc):
    # Leading underscore on params tells Streamlit's cache not to hash
    # these large/unhashable objects — safe here since both come from
    # already-cached, immutable artifacts.
    return get_shap_explainer(_churn_model, _background_enc)


def get_encoded_feature_names(preprocessor) -> list:
    """Human-readable names for the one-hot encoded columns, used to
    label the SHAP chart (otherwise bars show 'x12', 'x47', ...)."""
    return list(preprocessor.get_feature_names_out())


@st.cache_data
def load_and_score(uploaded_file):
    raw = pd.read_csv(uploaded_file)
    df = clean_data(raw)
    df = compute_rfm_features(df)

    preprocessor, kmeans, churn_model, segment_labels = get_artifacts()

    X, _ = get_feature_target_split(df)
    X_enc = preprocessor.transform(X)

    churn_proba = churn_model.predict_proba(X_enc)[:, 1]
    df["ChurnProbability"] = churn_proba
    df["RiskTier"] = pd.cut(
        churn_proba, bins=[0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"]
    )

    numeric_scaled = preprocessor.named_transformers_["num"].transform(
        df[NUMERIC_FEATURES]
    )
    cluster_ids = kmeans.predict(numeric_scaled)
    df["Segment"] = [segment_labels.get(c, f"Cluster {c}") for c in cluster_ids]

    df["EstimatedLTV"] = [
        estimate_ltv(row, proba) for (_, row), proba in zip(df.iterrows(), churn_proba)
    ]

    return df


uploaded_file = st.file_uploader("Upload a customer CSV (Telco churn format)", type="csv")

if uploaded_file is None:
    st.info("Upload a customer list to see churn risk, segments, and LTV estimates.")
    st.stop()

scored_df = load_and_score(uploaded_file)

col1, col2, col3 = st.columns(3)
col1.metric("Customers scanned", len(scored_df))
col2.metric("High-risk customers", int((scored_df["RiskTier"] == "High").sum()))
col3.metric(
    "Revenue at risk (monthly)",
    f"${scored_df.loc[scored_df['RiskTier'] == 'High', 'MonthlyCharges'].sum():,.0f}",
)

st.subheader("Customer list")

risk_filter = st.multiselect(
    "Filter by risk tier", options=["Low", "Medium", "High"], default=["High", "Medium"]
)
filtered = scored_df[scored_df["RiskTier"].isin(risk_filter)]

display_cols = [
    "tenure", "MonthlyCharges", "Segment", "ChurnProbability", "RiskTier", "EstimatedLTV"
]
st.dataframe(
    filtered[display_cols].sort_values("ChurnProbability", ascending=False),
    use_container_width=True,
)

st.subheader("Customer detail")
row_index = st.number_input(
    "Row index to inspect", min_value=0, max_value=len(filtered) - 1, value=0, step=1
)

if len(filtered) > 0:
    selected_row = filtered.iloc[int(row_index)]
    st.write(f"**Segment:** {selected_row['Segment']}")
    st.write(f"**Churn probability:** {selected_row['ChurnProbability']:.1%}")
    st.write(f"**Estimated LTV:** ${selected_row['EstimatedLTV']:,.2f}")

    if selected_row["RiskTier"] == "High" and selected_row["EstimatedLTV"] > scored_df["EstimatedLTV"].median():
        st.warning("Priority retention target: high value, high churn risk.")

    st.subheader("Why this score? (SHAP feature contributions)")

    preprocessor, kmeans, churn_model, segment_labels = get_artifacts()
    feature_cols = NUM_COLS + CATEGORICAL_FEATURES

    # Encode a small background sample once (SHAP needs a reference
    # distribution) and the single selected row for explanation.
    background_enc = preprocessor.transform(filtered[feature_cols].sample(
        min(50, len(filtered)), random_state=42
    ))
    row_enc = preprocessor.transform(selected_row[feature_cols].to_frame().T)

    explainer = get_explainer(churn_model, background_enc)
    raw_shap = explainer.shap_values(row_enc)

    # Different SHAP versions return this in different shapes for a
    # binary classifier. Handle all three so this doesn't break again
    # if the library updates:
    #   - list [class0_values, class1_values], each (n_samples, n_features)
    #   - ndarray (n_samples, n_features, n_classes)
    #   - ndarray (n_samples, n_features)  (single-output case)
    if isinstance(raw_shap, list):
        values = raw_shap[1][0]
    else:
        raw_shap = np.asarray(raw_shap)
        if raw_shap.ndim == 3:
            values = raw_shap[0, :, 1]  # row 0, all features, "churn=yes" class
        else:
            values = raw_shap[0]

    feature_names = get_encoded_feature_names(preprocessor)

    contrib = pd.Series(values, index=feature_names).sort_values(key=abs, ascending=False).head(8)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#D85A30" if v > 0 else "#1D9E75" for v in contrib.values]
    ax.barh(contrib.index[::-1], contrib.values[::-1], color=colors[::-1])
    ax.set_xlabel("Impact on churn probability")
    ax.set_title("Top factors for this customer")
    st.pyplot(fig)
    st.caption("Orange bars push churn risk up, green bars push it down.")

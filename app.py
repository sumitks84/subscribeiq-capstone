"""
app.py
SubscribeIQ — Streamlit dashboard for churn risk + customer lifetime
value. Auto-trains models on first run if none exist yet (so a fresh
deploy works without a manual training step).

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import (
    clean_data,
    compute_rfm_features,
    get_feature_target_split,
    NUMERIC_FEATURES as NUM_COLS,
    CATEGORICAL_FEATURES,
)
from src.model import (
    load_or_train_artifacts,
    get_shap_explainer,
    estimate_ltv,
    NUMERIC_FEATURES,
)

SAMPLE_DATA_PATH = "data/telco_churn.csv"

st.set_page_config(page_title="SubscribeIQ", page_icon="📡", layout="wide")

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #161B22 0%, #1C2430 100%);
        border: 1px solid #2A313C;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 6px;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #9AA4B2;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #FAFAFA;
    }
    .metric-sub {
        font-size: 0.8rem;
        margin-top: 2px;
    }
    .up { color: #E0634F; }
    .down { color: #1D9E75; }
    .app-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .app-subtitle {
        color: #9AA4B2;
        margin-top: 0;
        margin-bottom: 1.4rem;
    }
    .priority-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .pill-high { background: rgba(224,99,79,0.18); color: #E0634F; }
    .pill-medium { background: rgba(224,180,79,0.18); color: #E0B44F; }
    .pill-low { background: rgba(29,158,117,0.18); color: #1D9E75; }
    </style>
    """,
    unsafe_allow_html=True,
)

RISK_COLORS = {"High": "#E0634F", "Medium": "#E0B44F", "Low": "#1D9E75"}


def metric_card(label: str, value: str, sub: str = "", sub_class: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub {sub_class}">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_artifacts():
    with st.spinner("Preparing models (first run only — this takes a moment)..."):
        return load_or_train_artifacts()


@st.cache_resource
def get_explainer(_churn_model, _background_enc):
    return get_shap_explainer(_churn_model, _background_enc)


def get_encoded_feature_names(preprocessor) -> list:
    return list(preprocessor.get_feature_names_out())


@st.cache_data
def load_and_score(file_or_path):
    raw = pd.read_csv(file_or_path)
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

    numeric_scaled = preprocessor.named_transformers_["num"].transform(df[NUMERIC_FEATURES])
    cluster_ids = kmeans.predict(numeric_scaled)
    df["Segment"] = [segment_labels.get(c, f"Cluster {c}") for c in cluster_ids]

    df["EstimatedLTV"] = [
        estimate_ltv(row, proba) for (_, row), proba in zip(df.iterrows(), churn_proba)
    ]
    df["CustomerLabel"] = [f"Customer {i}" for i in df.index]

    return df


# ---------- Header ----------
st.markdown('<p class="app-title">📡 SubscribeIQ</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Churn risk, customer segments, and lifetime value — built for non-technical reviewers.</p>',
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Data")
    uploaded_file = st.file_uploader("Upload a customer CSV", type="csv")
    st.caption("No file? The bundled Telco sample dataset loads automatically.")

    st.divider()
    st.header("Filters")
    risk_filter = st.multiselect(
        "Risk tier", options=["Low", "Medium", "High"], default=["High", "Medium", "Low"]
    )

data_source = uploaded_file if uploaded_file is not None else SAMPLE_DATA_PATH

try:
    scored_df = load_and_score(data_source)
except FileNotFoundError:
    st.error("No dataset found. Upload a CSV to get started.")
    st.stop()

filtered = scored_df[scored_df["RiskTier"].isin(risk_filter)] if risk_filter else scored_df.iloc[0:0]

with st.sidebar:
    segment_options = sorted(scored_df["Segment"].unique())
    segment_filter = st.multiselect("Segment", options=segment_options, default=segment_options)
    filtered = filtered[filtered["Segment"].isin(segment_filter)]

    st.divider()
    st.caption(f"{len(filtered):,} of {len(scored_df):,} customers match your filters.")

# ---------- KPI row ----------
k1, k2, k3, k4 = st.columns(4)
with k1:
    metric_card("Customers scanned", f"{len(scored_df):,}")
with k2:
    high_risk_n = int((scored_df["RiskTier"] == "High").sum())
    high_risk_pct = high_risk_n / len(scored_df) * 100 if len(scored_df) else 0
    metric_card("High-risk customers", f"{high_risk_n:,}", f"{high_risk_pct:.1f}% of base", "up")
with k3:
    revenue_at_risk = scored_df.loc[scored_df["RiskTier"] == "High", "MonthlyCharges"].sum()
    metric_card("Revenue at risk / mo", f"${revenue_at_risk:,.0f}")
with k4:
    avg_ltv = scored_df["EstimatedLTV"].mean()
    metric_card("Avg. estimated LTV", f"${avg_ltv:,.0f}")

st.write("")

# ---------- Tabs ----------
tab_overview, tab_explore, tab_detail = st.tabs(["📊 Overview", "🔍 Customer Explorer", "🧑 Customer Detail"])

with tab_overview:
    col_a, col_b = st.columns([1, 1.4])

    with col_a:
        st.subheader("Risk distribution")
        risk_counts = scored_df["RiskTier"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    hole=0.55,
                    marker=dict(colors=[RISK_COLORS[r] for r in risk_counts.index]),
                    textinfo="label+percent",
                )
            ]
        )
        fig_donut.update_layout(
            showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=320,
            paper_bgcolor="rgba(0,0,0,0)", font_color="#FAFAFA",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        st.subheader("Segment breakdown")
        seg_counts = scored_df["Segment"].value_counts()
        fig_seg = px.bar(
            x=seg_counts.values, y=seg_counts.index, orientation="h",
            labels={"x": "Customers", "y": ""}, color=seg_counts.values,
            color_continuous_scale=["#1D9E75", "#E0B44F"],
        )
        fig_seg.update_layout(
            showlegend=False, coloraxis_showscale=False, height=260,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#FAFAFA",
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    with col_b:
        st.subheader("Tenure vs. monthly charges")
        fig_scatter = px.scatter(
            scored_df, x="tenure", y="MonthlyCharges", color="RiskTier",
            color_discrete_map=RISK_COLORS, opacity=0.6,
            hover_data=["Segment", "ChurnProbability", "EstimatedLTV"],
            labels={"tenure": "Tenure (months)", "MonthlyCharges": "Monthly charges ($)"},
        )
        fig_scatter.update_layout(
            height=320, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#FAFAFA",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Churn probability distribution")
        fig_hist = px.histogram(
            scored_df, x="ChurnProbability", nbins=30, color="RiskTier",
            color_discrete_map=RISK_COLORS,
            labels={"ChurnProbability": "Predicted churn probability"},
        )
        fig_hist.update_layout(
            height=260, margin=dict(t=10, b=10, l=10, r=10), barmode="stack",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#FAFAFA",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

with tab_explore:
    st.subheader(f"Customer list ({len(filtered):,} shown)")

    display_cols = [
        "CustomerLabel", "tenure", "MonthlyCharges", "Segment",
        "ChurnProbability", "RiskTier", "EstimatedLTV",
    ]
    st.dataframe(
        filtered[display_cols].sort_values("ChurnProbability", ascending=False),
        use_container_width=True,
        height=460,
        column_config={
            "ChurnProbability": st.column_config.ProgressColumn(
                "Churn probability", min_value=0, max_value=1, format="%.0f%%"
            ),
            "EstimatedLTV": st.column_config.NumberColumn("Est. LTV", format="$%.0f"),
            "MonthlyCharges": st.column_config.NumberColumn("Monthly charges", format="$%.2f"),
            "CustomerLabel": "Customer",
        },
        hide_index=True,
    )

    st.download_button(
        "Download filtered list as CSV",
        data=filtered[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="subscribeiq_filtered_customers.csv",
        mime="text/csv",
    )

with tab_detail:
    if len(filtered) == 0:
        st.info("No customers match the current filters. Adjust filters in the sidebar.")
    else:
        selected_label = st.selectbox(
            "Choose a customer",
            options=filtered.sort_values("ChurnProbability", ascending=False)["CustomerLabel"],
        )
        selected_row = filtered[filtered["CustomerLabel"] == selected_label].iloc[0]

        risk_tier = selected_row["RiskTier"]
        pill_class = {"High": "pill-high", "Medium": "pill-medium", "Low": "pill-low"}[risk_tier]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Segment", selected_row["Segment"])
        with c2:
            metric_card("Churn probability", f"{selected_row['ChurnProbability']:.1%}")
        with c3:
            metric_card("Estimated LTV", f"${selected_row['EstimatedLTV']:,.0f}")
        with c4:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Risk tier</div>'
                f'<span class="priority-pill {pill_class}">{risk_tier}</span></div>',
                unsafe_allow_html=True,
            )

        is_priority = risk_tier == "High" and selected_row["EstimatedLTV"] > scored_df["EstimatedLTV"].median()
        if is_priority:
            st.warning("⚠️ Priority retention target — high value, high churn risk.")
            if st.button("Simulate: send retention offer"):
                st.success(f"Retention offer queued for {selected_label}. (Demo action — no email actually sent.)")

        st.subheader("Why this score? (SHAP feature contributions)")

        preprocessor, kmeans, churn_model, segment_labels = get_artifacts()
        feature_cols = NUM_COLS + CATEGORICAL_FEATURES

        background_enc = preprocessor.transform(
            filtered[feature_cols].sample(min(50, len(filtered)), random_state=42)
        )
        row_enc = preprocessor.transform(selected_row[feature_cols].to_frame().T)

        explainer = get_explainer(churn_model, background_enc)
        raw_shap = explainer.shap_values(row_enc)

        if isinstance(raw_shap, list):
            values = raw_shap[1][0]
        else:
            raw_shap = np.asarray(raw_shap)
            values = raw_shap[0, :, 1] if raw_shap.ndim == 3 else raw_shap[0]

        feature_names = get_encoded_feature_names(preprocessor)
        contrib = pd.Series(values, index=feature_names).sort_values(key=abs, ascending=False).head(8)
        contrib = contrib.sort_values()

        fig_shap = go.Figure(
            go.Bar(
                x=contrib.values, y=contrib.index, orientation="h",
                marker_color=["#E0634F" if v > 0 else "#1D9E75" for v in contrib.values],
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            )
        )
        fig_shap.update_layout(
            height=340, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#FAFAFA",
            xaxis_title="Impact on churn probability",
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption("Orange bars push churn risk up, green bars push it down.")

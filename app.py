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
    :root {
        --ink: #f4f7fb;
        --muted: #8b9bb2;
        --line: rgba(148, 163, 184, 0.16);
        --panel: rgba(17, 27, 46, 0.78);
        --cyan: #58d6e8;
        --violet: #9d8cff;
    }
    .stApp {
        font-family: "Times New Roman", Times, serif;
        background:
            radial-gradient(circle at 78% -10%, rgba(88, 214, 232, 0.12), transparent 32rem),
            radial-gradient(circle at 8% 12%, rgba(157, 140, 255, 0.10), transparent 28rem),
            #08111f;
    }
    [data-testid="stHeader"] {
        background: rgba(8, 17, 31, 0.8);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1a2d 0%, #0a1425 100%);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--ink);
        letter-spacing: -0.02em;
    }
    [data-testid="stSidebar"] .stCaption {
        color: var(--muted);
    }
    .block-container {
        max-width: 1520px;
        padding-top: 3.5rem;
        padding-bottom: 4rem;
    }
    h1, h2, h3, h4 {
        color: var(--ink);
        letter-spacing: -0.035em;
    }
    .section-heading {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        color: var(--ink);
        font-size: 1.28rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin: 1.05rem 0 0.55rem;
    }
    .section-heading::before {
        content: "";
        display: inline-block;
        width: 4px;
        height: 1.25rem;
        border-radius: 99px;
        background: linear-gradient(180deg, var(--cyan), var(--violet));
        box-shadow: 0 0 12px rgba(88, 214, 232, 0.45);
    }
    .section-heading.amber::before {
        background: linear-gradient(180deg, #ffd27d, #ff7e6b);
    }
    .metric-card {
        cursor: pointer;
        transition: transform 180ms ease, box-shadow 180ms ease;
    }
    .metric-card:hover,
    .metric-card:has(+ div .card-action button:hover) {
        transform: scale(1.025);
        box-shadow: 0 18px 42px rgba(0, 0, 0, 0.28), 0 0 0 1px rgba(88, 214, 232, 0.16);
    }
    .card-action {
        height: 0;
        position: relative;
        z-index: 4;
    }
    [class*="st-key-kpi_"],
    [class*="st-key-detail_"] {
        height: 0 !important;
        min-height: 0 !important;
        position: relative;
        z-index: 4;
    }
    [class*="st-key-kpi_"] button,
    [class*="st-key-detail_"] button,
    .card-action button {
        opacity: 0;
        display: block;
        width: 100%;
        height: 112px;
        margin-top: -112px;
        padding: 0;
        cursor: pointer;
    }
    .dialog-insight-card {
        background: linear-gradient(145deg, rgba(28, 52, 82, 0.88), rgba(14, 27, 48, 0.92));
        border: 1px solid rgba(88, 214, 232, 0.28);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.9rem;
    }
    h2 {
        margin-top: 0.75rem;
    }
    [data-testid="stTabs"] {
        margin-top: 1.5rem;
    }
    [data-baseweb="tab-list"] {
        gap: 0.35rem;
        border-bottom: 1px solid var(--line);
        background: rgba(11, 24, 42, 0.62);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.28rem;
    }
    [data-baseweb="tab"] {
        color: var(--muted);
        padding: 0.62rem 1rem;
        font-weight: 600;
        border-radius: 10px;
        transition: background 160ms ease, color 160ms ease;
    }
    [data-baseweb="tab"]:nth-child(1)[aria-selected="true"] {
        color: var(--cyan) !important;
        background: rgba(88, 214, 232, 0.11);
    }
    [data-baseweb="tab"]:nth-child(2)[aria-selected="true"] {
        color: var(--violet) !important;
        background: rgba(157, 140, 255, 0.12);
    }
    [data-baseweb="tab"]:nth-child(3)[aria-selected="true"] {
        color: #ffd27d !important;
        background: rgba(255, 202, 105, 0.12);
    }
    [data-testid="stPlotlyChart"] {
        position: relative;
        border: 1px solid transparent;
        border-radius: 16px;
        padding: 0.25rem;
        background:
            linear-gradient(rgba(10, 21, 38, 0.88), rgba(10, 21, 38, 0.88)) padding-box,
            linear-gradient(120deg, rgba(88, 214, 232, 0.58), rgba(157, 140, 255, 0.25), rgba(88, 214, 232, 0.05)) border-box;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.12);
        transition: box-shadow 180ms ease, transform 180ms ease;
    }
    [data-testid="stPlotlyChart"]:hover {
        box-shadow: 0 0 0 1px rgba(88, 214, 232, 0.12), 0 16px 38px rgba(0, 0, 0, 0.2);
        transform: translateY(-1px);
    }
    .interaction-hint {
        display: inline-block;
        color: #9edfea;
        background: linear-gradient(100deg, rgba(88, 214, 232, 0.12), rgba(157, 140, 255, 0.1));
        border: 1px solid rgba(88, 214, 232, 0.22);
        border-radius: 999px;
        font-size: 0.74rem;
        padding: 0.38rem 0.72rem;
        margin: 0.2rem 0 0.7rem;
    }
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px dashed rgba(88, 214, 232, 0.38);
        border-radius: 12px;
        padding: 0.35rem;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 14px;
        overflow: hidden;
    }
    .metric-card {
        background: linear-gradient(145deg, rgba(22, 39, 65, 0.9), rgba(13, 25, 44, 0.86));
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 20px 21px;
        min-height: 104px;
        box-shadow: 0 14px 35px rgba(0, 0, 0, 0.16);
        margin-bottom: 8px;
    }
    .metric-card.neutral {
        background: linear-gradient(145deg, rgba(35, 55, 91, 0.9), rgba(17, 30, 55, 0.88));
        border-color: rgba(88, 214, 232, 0.28);
    }
    .metric-card.danger {
        background: linear-gradient(145deg, rgba(102, 48, 58, 0.76), rgba(40, 28, 48, 0.9));
        border-color: rgba(255, 126, 107, 0.32);
    }
    .metric-card.warning {
        background: linear-gradient(145deg, rgba(93, 72, 45, 0.78), rgba(39, 34, 40, 0.9));
        border-color: rgba(255, 202, 105, 0.3);
    }
    .metric-card.success {
        background: linear-gradient(145deg, rgba(25, 83, 79, 0.76), rgba(16, 38, 53, 0.9));
        border-color: rgba(100, 223, 187, 0.3);
    }
    .metric-label {
        font-size: 0.7rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 700;
        margin-bottom: 7px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--ink);
        letter-spacing: -0.04em;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: var(--muted);
        margin-top: 4px;
    }
    .up { color: #ff9b86; }
    .down { color: #64dfbb; }
    .app-title {
        font-size: 2.75rem;
        font-weight: 800;
        letter-spacing: -0.06em;
        margin: 0;
        color: var(--ink);
    }
    .app-subtitle {
        color: var(--muted);
        font-size: 1rem;
        margin: 0.35rem 0 1.8rem;
    }
    .eyebrow {
        color: var(--cyan);
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .status-badge {
        display: inline-block;
        background: rgba(100, 223, 187, 0.1);
        border: 1px solid rgba(100, 223, 187, 0.28);
        border-radius: 999px;
        color: #64dfbb;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.35rem 0.7rem;
        margin-top: 0.35rem;
    }
    .priority-pill {
        display: inline-block;
        padding: 5px 13px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .pill-high { background: rgba(255, 126, 107, 0.16); color: #ff9b86; }
    .pill-medium { background: rgba(255, 202, 105, 0.16); color: #ffd27d; }
    .pill-low { background: rgba(100, 223, 187, 0.16); color: #64dfbb; }
    </style>
    """,
    unsafe_allow_html=True,
)

RISK_COLORS = {"High": "#ff7e6b", "Medium": "#ffca69", "Low": "#64dfbb"}


def metric_card(
    label: str,
    value: str,
    sub: str = "",
    sub_class: str = "",
    tone: str = "neutral",
    action_key: str | None = None,
    details: list[str] | None = None,
):
    st.markdown(
        f"""
        <div class="metric-card {tone}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub {sub_class}">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if action_key and details:
        st.markdown('<div class="card-action">', unsafe_allow_html=True)
        if st.button("View insight ↗", key=action_key, use_container_width=False):
            show_card_insight(label, value, details)
        st.markdown("</div>", unsafe_allow_html=True)


@st.cache_resource
def get_artifacts():
    with st.spinner("Preparing models (first run only — this takes a moment)..."):
        return load_or_train_artifacts()


@st.cache_resource
def get_explainer(_churn_model, _background_enc):
    return get_shap_explainer(_churn_model, _background_enc)


def get_encoded_feature_names(preprocessor) -> list:
    return list(preprocessor.get_feature_names_out())


def selected_point(event):
    """Return the first Plotly point selected by a chart click."""
    if event is None:
        return None
    try:
        points = event.selection.points
    except AttributeError:
        points = event.get("selection", {}).get("points", []) if isinstance(event, dict) else []
    return points[0] if points else None


@st.dialog("Chart insight")
def show_chart_insight(title: str, summary: str, details: list[str]):
    st.markdown('<div class="dialog-insight-card">', unsafe_allow_html=True)
    st.markdown(f"### {title}")
    st.write(summary)
    for detail in details:
        st.markdown(f"- {detail}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Click another point or slice to explore a different insight.")


@st.dialog("Card insight")
def show_card_insight(title: str, value: str, details: list[str]):
    st.markdown('<div class="dialog-insight-card">', unsafe_allow_html=True)
    st.markdown(f"### {title}")
    st.markdown(f"## {value}")
    for detail in details:
        st.markdown(f"- {detail}")
    st.markdown("</div>", unsafe_allow_html=True)


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
st.markdown('<div class="eyebrow">Customer intelligence platform</div>', unsafe_allow_html=True)
st.markdown('<p class="app-title">📡 SubscribeIQ</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Churn risk, customer segments, and lifetime value — built for non-technical reviewers.</p>',
    unsafe_allow_html=True,
)
st.markdown('<span class="status-badge">● Models online &nbsp;·&nbsp; Analysis ready</span>', unsafe_allow_html=True)

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
    metric_card(
        "Customers scanned", f"{len(scored_df):,}", tone="neutral",
        action_key="kpi_customers",
        details=["Total customer records currently scored by the churn and segmentation models.",
                 "Use the sidebar filters to narrow the visible customer population."],
    )
with k2:
    high_risk_n = int((scored_df["RiskTier"] == "High").sum())
    high_risk_pct = high_risk_n / len(scored_df) * 100 if len(scored_df) else 0
    metric_card(
        "High-risk customers", f"{high_risk_n:,}", f"{high_risk_pct:.1f}% of base", "up", "danger",
        action_key="kpi_high_risk",
        details=["Customers with a predicted churn probability above 60%.",
                 "This is the primary retention-priority population in the dashboard."],
    )
with k3:
    revenue_at_risk = scored_df.loc[scored_df["RiskTier"] == "High", "MonthlyCharges"].sum()
    metric_card(
        "Revenue at risk / mo", f"${revenue_at_risk:,.0f}", tone="warning",
        action_key="kpi_revenue_risk",
        details=["Sum of monthly charges for customers in the High risk tier.",
                 "It estimates the monthly recurring revenue exposed if those customers churn."],
    )
with k4:
    avg_ltv = scored_df["EstimatedLTV"].mean()
    metric_card(
        "Avg. estimated LTV", f"${avg_ltv:,.0f}", tone="success",
        action_key="kpi_ltv",
        details=["Average model-estimated lifetime value across the scored customer base.",
                 "Use this alongside churn risk to prioritize high-value retention work."],
    )

st.write("")

# ---------- Tabs ----------
tab_overview, tab_explore, tab_detail = st.tabs(["📊 Overview", "🔍 Customer Explorer", "🧑 Customer Detail"])

with tab_overview:
    st.markdown(
        '<div class="interaction-hint">✦ Click any slice, bar, point, or histogram bin to open a focused explanation.</div>',
        unsafe_allow_html=True,
    )
    col_a, col_b = st.columns([1, 1.4])
    chart_insight = None

    with col_a:
        st.markdown('<div class="section-heading">Risk distribution</div>', unsafe_allow_html=True)
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
        donut_event = st.plotly_chart(
            fig_donut, use_container_width=True, key="risk_distribution",
            on_select="rerun", selection_mode=("points",),
        )
        donut_point = selected_point(donut_event)
        if donut_point:
            donut_value = int(donut_point.get("value", 0))
            donut_share = donut_value / len(scored_df) if len(scored_df) else 0
            chart_insight = (
                "Risk distribution",
                f"{donut_point.get('label', 'This tier')} includes "
                f"{donut_value:,} customers "
                f"({donut_share:.1%} of the customer base).",
                [
                    "Low risk customers are predicted to have less than 30% churn probability.",
                    "Medium risk customers fall between 30% and 60% predicted churn probability.",
                    "High risk customers are above 60% predicted churn probability and are the clearest retention priority.",
                ],
            )

        st.markdown('<div class="section-heading">Segment breakdown</div>', unsafe_allow_html=True)
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
        seg_event = st.plotly_chart(
            fig_seg, use_container_width=True, key="segment_breakdown",
            on_select="rerun", selection_mode=("points",),
        )
        seg_point = selected_point(seg_event)
        if seg_point:
            segment_name = seg_point.get("y", "This segment")
            segment_total = int(seg_point.get("x", 0))
            segment_rows = scored_df[scored_df["Segment"] == segment_name]
            segment_risk = segment_rows["ChurnProbability"].mean() if len(segment_rows) else 0
            chart_insight = (
                "Segment breakdown",
                f"{segment_name} contains {segment_total:,} customers, "
                f"with an average predicted churn probability of {segment_risk:.1%}.",
                [
                    "The bar length represents the number of customers assigned to the segment.",
                    "Use the segment filter in the sidebar to focus the rest of the dashboard on this group.",
                    "Pair segment size with churn probability to prioritize outreach efficiently.",
                ],
            )

    with col_b:
        st.markdown('<div class="section-heading">Tenure vs. monthly charges</div>', unsafe_allow_html=True)
        fig_scatter = px.scatter(
            scored_df, x="tenure", y="MonthlyCharges", color="RiskTier",
            color_discrete_map=RISK_COLORS, opacity=0.6,
            hover_data=["Segment", "ChurnProbability", "EstimatedLTV"],
            custom_data=["CustomerLabel", "Segment", "ChurnProbability", "EstimatedLTV", "RiskTier"],
            labels={"tenure": "Tenure (months)", "MonthlyCharges": "Monthly charges ($)"},
        )
        fig_scatter.update_layout(
            height=320, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#FAFAFA",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        scatter_event = st.plotly_chart(
            fig_scatter, use_container_width=True, key="tenure_charges",
            on_select="rerun", selection_mode=("points",),
        )
        scatter_point = selected_point(scatter_event)
        if scatter_point:
            custom = scatter_point.get("customdata", [])
            customer_label = custom[0] if len(custom) > 0 else "This customer"
            segment_name = custom[1] if len(custom) > 1 else "their segment"
            churn_probability = float(custom[2]) if len(custom) > 2 else 0
            estimated_ltv = float(custom[3]) if len(custom) > 3 else 0
            chart_insight = (
                "Tenure vs. monthly charges",
                f"{customer_label} has been subscribed for {float(scatter_point.get('x', 0)):.0f} months "
                f"at ${float(scatter_point.get('y', 0)):,.2f} per month.",
                [
                    f"This customer belongs to the {segment_name} segment.",
                    f"Their predicted churn probability is {churn_probability:.1%}.",
                    f"Their estimated lifetime value is ${estimated_ltv:,.0f}.",
                ],
            )

        st.markdown('<div class="section-heading">Churn probability distribution</div>', unsafe_allow_html=True)
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
        hist_event = st.plotly_chart(
            fig_hist, use_container_width=True, key="churn_distribution",
            on_select="rerun", selection_mode=("points",),
        )
        hist_point = selected_point(hist_event)
        if hist_point:
            bucket_value = float(hist_point.get("x", 0))
            risk_label = "Low" if bucket_value < 0.3 else "Medium" if bucket_value < 0.6 else "High"
            bucket_rows = scored_df[
                (scored_df["ChurnProbability"] >= bucket_value - 0.02)
                & (scored_df["ChurnProbability"] <= bucket_value + 0.02)
            ]
            chart_insight = (
                "Churn probability distribution",
                f"This area of the chart represents customers around {bucket_value:.0%} "
                f"predicted churn probability, which maps to the {risk_label} risk tier.",
                [
                    f"Approximately {len(bucket_rows):,} customers fall within this nearby probability range.",
                    "The x-axis shows model-predicted churn probability; the y-axis shows customer count.",
                    "A concentration toward the right indicates more customers need retention attention.",
                ],
            )

    if chart_insight:
        show_chart_insight(*chart_insight)

with tab_explore:
    st.markdown(
        f'<div class="section-heading">Customer list <span style="color:#8b9bb2;font-size:.85rem;">({len(filtered):,} shown)</span></div>',
        unsafe_allow_html=True,
    )

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
            metric_card(
                "Segment", selected_row["Segment"], tone="neutral",
                action_key="detail_segment",
                details=["Behavioral segment assigned from numeric customer patterns.",
                         "Compare this segment with its churn probability and lifetime value."],
            )
        with c2:
            detail_risk_tone = {"High": "danger", "Medium": "warning", "Low": "success"}[risk_tier]
            metric_card(
                "Churn probability", f"{selected_row['ChurnProbability']:.1%}", tone=detail_risk_tone,
                action_key="detail_churn",
                details=["Model-estimated probability that this customer will churn.",
                         "High risk begins above 60%; medium risk spans 30% to 60%."],
            )
        with c3:
            metric_card(
                "Estimated LTV", f"${selected_row['EstimatedLTV']:,.0f}", tone="success",
                action_key="detail_ltv",
                details=["Estimated customer lifetime value adjusted for the customer's churn probability.",
                         "Higher value and higher risk together indicate a stronger retention opportunity."],
            )
        with c4:
            st.markdown(
                f'<div class="metric-card {detail_risk_tone}"><div class="metric-label">Risk tier</div>'
                f'<span class="priority-pill {pill_class}">{risk_tier}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="card-action">', unsafe_allow_html=True)
            if st.button("View insight ↗", key="detail_risk", use_container_width=False):
                show_card_insight(
                    "Risk tier", str(risk_tier),
                    ["Low: predicted churn below 30%.",
                     "Medium: predicted churn between 30% and 60%.",
                     "High: predicted churn above 60%."],
                )
            st.markdown("</div>", unsafe_allow_html=True)

        is_priority = risk_tier == "High" and selected_row["EstimatedLTV"] > scored_df["EstimatedLTV"].median()
        if is_priority:
            st.warning("⚠️ Priority retention target — high value, high churn risk.")
            if st.button("Simulate: send retention offer"):
                st.success(f"Retention offer queued for {selected_label}. (Demo action — no email actually sent.)")

        st.markdown(
            '<div class="section-heading amber">Why this score? <span style="color:#8b9bb2;font-size:.85rem;">(SHAP feature contributions)</span></div>',
            unsafe_allow_html=True,
        )

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
                marker_color=["#ff7e6b" if v > 0 else "#64dfbb" for v in contrib.values],
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            )
        )
        fig_shap.update_layout(
            height=340, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#FAFAFA",
            xaxis_title="Impact on churn probability",
        )
        shap_event = st.plotly_chart(
            fig_shap, use_container_width=True, key="shap_contributions",
            on_select="rerun", selection_mode=("points",),
        )
        st.caption("Orange bars push churn risk up, green bars push it down. Click a bar to inspect it.")
        shap_point = selected_point(shap_event)
        if shap_point:
            impact = float(shap_point.get("x", 0))
            direction = "increases" if impact > 0 else "reduces"
            show_chart_insight(
                "SHAP feature contribution",
                f"{shap_point.get('y', 'This feature')} {direction} this customer's "
                f"predicted churn risk by {abs(impact):.3f} model units.",
                [
                    "Orange bars push the churn prediction higher; green bars push it lower.",
                    "Longer bars have a stronger influence on the individual customer's score.",
                    "This explains model behavior for the selected customer, not a general population trend.",
                ],
            )

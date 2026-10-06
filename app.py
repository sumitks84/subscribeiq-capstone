"""
app.py

SubscribeIQ - a light-themed Streamlit dashboard for subscription churn
risk, customer segments, and lifetime value. The models (Random Forest
churn classifier, K-Means segmentation, SHAP explanations) auto-train on
first run if no saved artifacts exist, so a fresh deploy works without a
manual training step.

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
    load_raw_data,
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

st.set_page_config(
    page_title="SubscribeIQ - Retention intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Design tokens
#   Premium light theme. One interactive accent (indigo). A separate
#   semantic scale is reserved for risk only, so colour always means the
#   same thing: green = safe, amber = watch, rose = act now.
# ----------------------------------------------------------------------
INK = "#16181D"          # off-black, never pure #000
MUTED = "#5B6472"
FAINT = "#8A92A1"
LINE = "#E8E9EC"
PAPER = "#FBFBFA"        # off-white base, never pure #FFF
SURFACE = "#FFFFFF"
ACCENT = "#4F46E5"       # indigo, the only interactive accent
ACCENT_SOFT = "#EEF0FE"

RISK_LOW = "#0E9F6E"     # emerald
RISK_MED = "#C77700"     # amber
RISK_HIGH = "#D0384E"    # rose
RISK_COLORS = {"High": RISK_HIGH, "Medium": RISK_MED, "Low": RISK_LOW}

PLOT_FONT = "Inter, -apple-system, Segoe UI, Roboto, sans-serif"

# ---------- Styling ----------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Spline+Sans+Mono:wght@500;600&display=swap');

:root {{
    --ink: {INK};
    --muted: {MUTED};
    --faint: {FAINT};
    --line: {LINE};
    --paper: {PAPER};
    --surface: {SURFACE};
    --accent: {ACCENT};
    --accent-soft: {ACCENT_SOFT};
    --low: {RISK_LOW};
    --med: {RISK_MED};
    --high: {RISK_HIGH};
    --radius: 14px;
    --shadow: 0 1px 2px rgba(22,24,29,0.04), 0 8px 24px rgba(22,24,29,0.05);
    --shadow-hover: 0 2px 4px rgba(22,24,29,0.06), 0 14px 38px rgba(22,24,29,0.09);
}}

.stApp {{
    background: var(--paper);
    color: var(--ink);
    font-family: Inter, -apple-system, "Segoe UI", Roboto, sans-serif;
}}

[data-testid="stHeader"] {{
    background: transparent;
    height: 0;
}}
[data-testid="stToolbar"] {{ display: none; }}
#MainMenu, footer {{ visibility: hidden; }}

.block-container {{
    max-width: 1320px;
    padding-top: 0.8rem;
    padding-bottom: 5rem;
}}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {{
    background: var(--surface);
    border-right: 1px solid var(--line);
}}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem; }}
[data-testid="stSidebar"] h2 {{
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--faint);
    font-weight: 700;
    margin-bottom: 0.4rem;
}}

/* ---- Typography ---- */
html, body, [class*="css"], .stApp, .stMarkdown,
h1, h2, h3, h4, h5, h6, p, span, div, button, input, label, a {{
    font-family: Inter, -apple-system, "Segoe UI", Roboto, sans-serif;
}}
h1, h2, h3, h4 {{ color: var(--ink); letter-spacing: -0.02em; font-weight: 700; }}
/* numbers keep the mono face explicitly set on their own elements */
.hero .stake .big, .kpi .val, .act .num {{ font-family: "Spline Sans Mono", monospace; }}

.sq-nav {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.8rem 0;
    margin-bottom: 1.15rem;
    border-bottom: 1px solid var(--line);
    position: sticky; top: 0; z-index: 30;
    background: rgba(251,251,250,0.86);
    backdrop-filter: blur(8px);
}}
.sq-brand {{ display: flex; align-items: center; gap: 0.7rem; font-weight: 700; font-size: 1.3rem; letter-spacing: -0.025em; }}
.sq-brand .mark {{
    width: 34px; height: 34px; border-radius: 9px;
    background: var(--accent);
    display: inline-flex; align-items: center; justify-content: center;
    color: #fff; font-size: 1rem;
}}
.sq-links {{ display: flex; gap: 1.25rem; font-size: 0.92rem; }}
.sq-links a {{ color: var(--muted); text-decoration: none; font-weight: 500; white-space: nowrap; transition: color .18s ease; }}
.sq-links a:hover {{ color: var(--accent); }}
@media (max-width: 1200px) {{
    .sq-nav {{ flex-wrap: wrap; }}
    .sq-links {{
        width: 100%;
        justify-content: flex-end;
        padding-top: 0.55rem;
    }}
}}

/* ---- Hero ---- */
.hero {{
    display: grid;
    grid-template-columns: 1.35fr 1fr;
    gap: 2.2rem;
    align-items: center;
    padding: 1.2rem 0 2.4rem;
    animation: rise .6s cubic-bezier(.16,1,.3,1) both;
}}
.hero h1 {{
    font-size: 2.65rem; font-weight: 800; line-height: 1.04;
    letter-spacing: -0.035em; margin: 0 0 0.9rem;
}}
.hero p {{
    font-size: 1.02rem; color: var(--muted); line-height: 1.55;
    max-width: 52ch; margin: 0;
}}
.hero .stake {{
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 1.5rem 1.6rem;
    box-shadow: var(--shadow);
}}
.hero .stake .lab {{ font-size: 0.78rem; color: var(--muted); font-weight: 600; }}
.hero .stake .big {{
    font-family: "Spline Sans Mono", monospace;
    font-size: 2.75rem; font-weight: 600; color: var(--high);
    letter-spacing: -0.03em; line-height: 1.1; margin: 0.25rem 0;
}}
.hero .stake .note {{ font-size: 0.85rem; color: var(--faint); }}

/* ---- KPI cards ---- */
.kpi {{
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 1.15rem 1.25rem;
    box-shadow: var(--shadow);
    transition: transform .2s cubic-bezier(.16,1,.3,1), box-shadow .2s ease;
    height: 100%;
}}
.kpi:hover {{ transform: translateY(-2px); box-shadow: var(--shadow-hover); }}
.kpi .lab {{
    font-size: 0.74rem; color: var(--muted); font-weight: 600;
    display: flex; align-items: center; gap: 0.4rem;
}}
.kpi .val {{
    font-family: "Spline Sans Mono", monospace;
    font-size: 1.95rem; font-weight: 600; letter-spacing: -0.03em;
    margin: 0.35rem 0 0.1rem;
}}
.kpi .sub {{ font-size: 0.8rem; color: var(--faint); }}
.kpi .dot {{ width: 7px; height: 7px; border-radius: 50%; display:inline-block; }}

/* ---- Section heading ---- */
.sec {{ margin: 2.6rem 0 0.4rem; }}
.sec h3 {{ font-size: 1.28rem; font-weight: 700; margin: 0; letter-spacing: -0.025em; }}
.sec p {{ font-size: 0.9rem; color: var(--muted); margin: 0.3rem 0 0; max-width: 70ch; }}

/* ---- Action rows ---- */
.act {{
    display: grid;
    grid-template-columns: 1.4fr 1fr 1fr 1fr auto;
    gap: 1rem; align-items: center;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.55rem;
    transition: border-color .18s ease, box-shadow .18s ease;
}}
.act:hover {{ border-color: #D5D7FB; box-shadow: var(--shadow); }}
.act .who {{ font-weight: 600; font-size: 0.95rem; }}
.act .seg {{ font-size: 0.8rem; color: var(--muted); }}
.act .num {{ font-family: "Spline Sans Mono", monospace; font-weight: 600; font-size: 0.98rem; }}
.act .k {{ font-size: 0.72rem; color: var(--faint); display:block; }}

/* ---- Risk pill ---- */
.pill {{ display:inline-block; padding: 3px 11px; border-radius: 999px; font-size: 0.76rem; font-weight: 600; }}
.pill-high {{ background: #FCEBEE; color: var(--high); }}
.pill-med  {{ background: #FBF1E0; color: var(--med); }}
.pill-low  {{ background: #E5F6EF; color: var(--low); }}

/* ---- Plotly frame ---- */
[data-testid="stPlotlyChart"] {{
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 0.6rem 0.7rem 0.3rem;
    box-shadow: var(--shadow);
}}

/* ---- Tabs ---- */
[data-baseweb="tab-list"] {{
    gap: 0.25rem; background: #F2F3F5; border-radius: 11px;
    padding: 0.25rem; border: 1px solid var(--line);
}}
[data-baseweb="tab"] {{
    color: var(--muted); font-weight: 600; border-radius: 8px;
    padding: 0.5rem 1rem; font-size: 0.9rem;
}}
[data-baseweb="tab"][aria-selected="true"] {{
    background: var(--surface); color: var(--accent);
    box-shadow: 0 1px 2px rgba(22,24,29,0.06);
}}
[data-testid="stTabs"] {{ margin-top: 1.4rem; }}

/* ---- Buttons ---- */
.stButton button, .stDownloadButton button {{
    border-radius: 9px; font-weight: 600; border: 1px solid var(--accent);
    background: var(--accent); color: #fff; transition: transform .12s ease, filter .18s ease;
}}
.stButton button:hover, .stDownloadButton button:hover {{ filter: brightness(1.07); border-color: var(--accent); }}
.stButton button:active {{ transform: scale(0.98); }}

/* ---- Dataframe ---- */
[data-testid="stDataFrame"] {{ border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }}

/* ---- File uploader ---- */
[data-testid="stFileUploader"] {{
    background: var(--surface); border: 1px dashed #C9CBD4;
    border-radius: 12px; padding: 0.4rem;
}}

/* ---- Insight dialog ---- */
.ins-card {{
    background: var(--accent-soft);
    border: 1px solid #DADCFB;
    border-radius: 12px; padding: 1rem 1.1rem;
}}

@keyframes rise {{ from {{ opacity: 0; transform: translateY(12px); }} to {{ opacity: 1; transform: none; }} }}

@media (prefers-reduced-motion: reduce) {{
    .hero {{ animation: none; }}
    .kpi, .act, .stButton button {{ transition: none; }}
}}

@media (max-width: 860px) {{
    .hero {{ grid-template-columns: 1fr; }}
    .act {{ grid-template-columns: 1fr 1fr; }}
    .sq-links {{ display: none; }}
}}
</style>
    """,
    unsafe_allow_html=True,
)


def plotly_base(fig, height):
    """Shared light-theme Plotly styling."""
    fig.update_layout(
        height=height,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, family=PLOT_FONT, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor="#EFF0F3", zerolinecolor="#E3E4E8", linecolor="#E3E4E8")
    fig.update_yaxes(gridcolor="#EFF0F3", zerolinecolor="#E3E4E8", linecolor="#E3E4E8")
    return fig


# ---------- Cached resources ----------
@st.cache_resource
def get_artifacts():
    with st.spinner("Preparing models (first run only)..."):
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


@st.dialog("Insight")
def show_chart_insight(title: str, summary: str, details: list[str]):
    st.markdown('<div class="ins-card">', unsafe_allow_html=True)
    st.markdown(f"#### {title}")
    st.write(summary)
    for detail in details:
        st.markdown(f"- {detail}")
    st.markdown("</div>", unsafe_allow_html=True)


@st.cache_data
def load_and_score(file_or_path):
    raw = load_raw_data(file_or_path)
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


# ======================================================================
# TOP NAVIGATION
# ======================================================================
st.markdown(
    """
<div class="sq-nav">
  <div class="sq-brand"><span class="mark">◆</span> SubscribeIQ</div>
  <div class="sq-links">
    <a href="#priorities">Who to call</a>
    <a href="#drivers">Why they leave</a>
    <a href="#explore">Explore</a>
    <a href="#detail">Customer</a>
  </div>
</div>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar (data + filters) ----------
with st.sidebar:
    st.markdown("## Data")
    uploaded_file = st.file_uploader("Upload a customer file", label_visibility="collapsed")
    st.caption(
        "CSV or TSV. No file? The sample telecom subscriber dataset loads automatically so "
        "you can explore right away."
    )
    st.divider()
    st.markdown("## Filters")
    risk_filter = st.multiselect(
        "Risk tier", options=["Low", "Medium", "High"], default=["High", "Medium", "Low"]
    )

data_source = uploaded_file if uploaded_file is not None else SAMPLE_DATA_PATH
try:
    scored_df = load_and_score(data_source)
except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
    st.error(f"That file could not be read. Check it is a CSV or TSV and try again. ({error})")
    st.stop()
except (KeyError, ValueError) as error:
    st.error(
        "That file does not match the customer model. It needs the same columns as the "
        f"sample dataset (tenure, MonthlyCharges, contract type, and so on). ({error})"
    )
    st.stop()

filtered = scored_df[scored_df["RiskTier"].isin(risk_filter)] if risk_filter else scored_df.iloc[0:0]

with st.sidebar:
    segment_options = sorted(scored_df["Segment"].unique())
    segment_filter = st.multiselect("Segment", options=segment_options, default=segment_options)
    filtered = filtered[filtered["Segment"].isin(segment_filter)]
    st.divider()
    st.caption(f"{len(filtered):,} of {len(scored_df):,} customers match your filters.")

# ---------- Derived headline numbers ----------
total_n = len(scored_df)
high_n = int((scored_df["RiskTier"] == "High").sum())
high_pct = high_n / total_n * 100 if total_n else 0
revenue_at_risk = scored_df.loc[scored_df["RiskTier"] == "High", "MonthlyCharges"].sum()
avg_ltv = scored_df["EstimatedLTV"].mean()

# ======================================================================
# HERO - plain-language orientation + the number that matters
# ======================================================================
st.markdown(
    f"""
<div class="hero">
  <div>
    <h1>Know which subscribers are about to leave,<br>and what it costs you.</h1>
    <p>SubscribeIQ scores every customer for churn risk, groups them into
    segments, and estimates what each one is worth. It turns a raw customer
    list into a ranked set of people worth saving this week.</p>
  </div>
  <div class="stake">
    <div class="lab">Monthly revenue in high-risk accounts</div>
    <div class="big">${revenue_at_risk:,.0f}</div>
    <div class="note">{high_n:,} customers ({high_pct:.0f}% of the base) are more likely than not to churn.</div>
  </div>
</div>
    """,
    unsafe_allow_html=True,
)

# ---------- KPI row ----------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"""<div class="kpi"><div class="lab">Customers scored</div>
        <div class="val">{total_n:,}</div>
        <div class="sub">Every record run through the model</div></div>""",
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""<div class="kpi"><div class="lab"><span class="dot" style="background:{RISK_HIGH}"></span>High-risk customers</div>
        <div class="val" style="color:{RISK_HIGH}">{high_n:,}</div>
        <div class="sub">{high_pct:.1f}% of the base, churn above 60%</div></div>""",
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""<div class="kpi"><div class="lab">Revenue at risk / month</div>
        <div class="val">${revenue_at_risk:,.0f}</div>
        <div class="sub">Monthly charges tied to high-risk accounts</div></div>""",
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""<div class="kpi"><div class="lab">Average estimated LTV</div>
        <div class="val">${avg_ltv:,.0f}</div>
        <div class="sub">Lifetime value, discounted by churn risk</div></div>""",
        unsafe_allow_html=True,
    )

# ======================================================================
# SECTION 1 - PRIORITIES: who to call first (new, highest business value)
# ======================================================================
st.markdown('<div id="priorities"></div>', unsafe_allow_html=True)
st.markdown(
    """
<div class="sec"><h3>Who to call first</h3>
<p>The customers where a save is both likely needed and worth the most:
high churn risk and above-median lifetime value. Start at the top.</p></div>
    """,
    unsafe_allow_html=True,
)

ltv_median = scored_df["EstimatedLTV"].median()
priorities = (
    filtered[(filtered["RiskTier"] == "High") & (filtered["EstimatedLTV"] >= ltv_median)]
    .sort_values(["EstimatedLTV", "ChurnProbability"], ascending=False)
    .head(6)
)

if len(priorities) == 0:
    st.info(
        "No high-value, high-risk customers in the current filter. Widen the risk or segment "
        "filters in the sidebar to see the full priority list."
    )
else:
    for _, r in priorities.iterrows():
        st.markdown(
            f"""
<div class="act">
  <div><div class="who">{r['CustomerLabel']}</div><div class="seg">{r['Segment']}</div></div>
  <div><span class="k">Churn risk</span><span class="num" style="color:{RISK_HIGH}">{r['ChurnProbability']:.0%}</span></div>
  <div><span class="k">Monthly</span><span class="num">${r['MonthlyCharges']:,.0f}</span></div>
  <div><span class="k">Est. LTV</span><span class="num">${r['EstimatedLTV']:,.0f}</span></div>
  <div><span class="pill pill-high">Act now</span></div>
</div>
            """,
            unsafe_allow_html=True,
        )
    saved = int(priorities["MonthlyCharges"].sum())
    st.caption(f"Saving just these {len(priorities)} accounts protects ${saved:,}/month in recurring revenue.")

# ======================================================================
# SECTION 2 - DRIVERS: why customers leave (new contract chart)
# ======================================================================
st.markdown('<div id="drivers"></div>', unsafe_allow_html=True)
st.markdown(
    """
<div class="sec"><h3>Why customers leave</h3>
<p>Churn is not spread evenly. Contract length is the single clearest signal
in the data, which points straight at what to change.</p></div>
    """,
    unsafe_allow_html=True,
)

dcol1, dcol2 = st.columns([1.25, 1])

with dcol1:
    churn_num = None
    if "Churn" in scored_df.columns:
        churn_num = (
            scored_df["Churn"].map({"Yes": 1, "No": 0})
            if scored_df["Churn"].dtype == object
            else scored_df["Churn"]
        )

    if churn_num is not None and "Contract" in scored_df.columns:
        tmp = scored_df.assign(_ch=churn_num)
        contract_rate = (
            tmp.groupby("Contract")["_ch"].mean().reindex(
                ["Month-to-month", "One year", "Two year"]
            ).dropna() * 100
        )
        fig_contract = go.Figure(
            go.Bar(
                x=contract_rate.values,
                y=contract_rate.index,
                orientation="h",
                marker_color=[RISK_HIGH, RISK_MED, RISK_LOW][: len(contract_rate)],
                text=[f"{v:.0f}%" for v in contract_rate.values],
                textposition="outside",
                hovertemplate="%{y}: %{x:.1f}% churn<extra></extra>",
            )
        )
        fig_contract.update_xaxes(title="Actual churn rate", ticksuffix="%",
                                  range=[0, max(contract_rate.values) * 1.25])
        plotly_base(fig_contract, 260)
        fig_contract.update_layout(showlegend=False)
        st.plotly_chart(fig_contract, use_container_width=True, key="contract_churn")
    elif "Contract" in scored_df.columns:
        pred = (
            scored_df.groupby("Contract")["ChurnProbability"].mean().reindex(
                ["Month-to-month", "One year", "Two year"]
            ).dropna() * 100
        )
        fig_contract = go.Figure(
            go.Bar(
                x=pred.values, y=pred.index, orientation="h",
                marker_color=[RISK_HIGH, RISK_MED, RISK_LOW][: len(pred)],
                text=[f"{v:.0f}%" for v in pred.values], textposition="outside",
                hovertemplate="%{y}: %{x:.1f}% predicted risk<extra></extra>",
            )
        )
        fig_contract.update_xaxes(title="Average predicted churn risk", ticksuffix="%")
        plotly_base(fig_contract, 260)
        fig_contract.update_layout(showlegend=False)
        st.plotly_chart(fig_contract, use_container_width=True, key="contract_pred")
    else:
        st.info("This file has no contract column, so the contract driver chart is hidden.")

with dcol2:
    st.markdown(
        """
<div class="kpi" style="height:100%;">
  <div class="lab">Read this chart</div>
  <p style="font-size:0.9rem;color:var(--muted);line-height:1.5;margin:0.6rem 0 0;">
  Month-to-month subscribers churn many times more often than customers on a
  one or two year contract. The clearest retention lever is moving at-risk
  month-to-month customers onto a longer commitment, with an incentive that
  costs less than the revenue it protects.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

# ======================================================================
# SECTION 3 - PORTFOLIO VIEW (restyled original charts)
# ======================================================================
st.markdown('<div id="explore"></div>', unsafe_allow_html=True)
st.markdown(
    """
<div class="sec"><h3>The whole base at a glance</h3>
<p>How risk, segments, and value are spread across every customer. Click any
point, bar, or slice for a plain-language explanation.</p></div>
    """,
    unsafe_allow_html=True,
)

chart_insight = None
col_a, col_b = st.columns([1, 1.4])

with col_a:
    st.markdown("**Risk distribution**")
    risk_counts = scored_df["RiskTier"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
    fig_donut = go.Figure(
        data=[
            go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.6,
                marker=dict(colors=[RISK_COLORS[r] for r in risk_counts.index],
                            line=dict(color=SURFACE, width=2)),
                textinfo="label+percent",
                textfont=dict(size=12),
            )
        ]
    )
    plotly_base(fig_donut, 300)
    fig_donut.update_layout(showlegend=False)
    donut_event = st.plotly_chart(
        fig_donut, use_container_width=True, key="risk_distribution",
        on_select="rerun", selection_mode=("points",),
    )
    donut_point = selected_point(donut_event)
    if donut_point:
        v = int(donut_point.get("value", 0))
        share = v / total_n if total_n else 0
        chart_insight = (
            "Risk distribution",
            f"{donut_point.get('label', 'This tier')} holds {v:,} customers ({share:.0%} of the base).",
            [
                "Low: predicted churn under 30%.",
                "Medium: predicted churn 30% to 60%.",
                "High: predicted churn above 60%, the clearest retention priority.",
            ],
        )

    st.markdown("**Segment sizes**")
    seg_counts = scored_df["Segment"].value_counts()
    fig_seg = px.bar(
        x=seg_counts.values, y=seg_counts.index, orientation="h",
        labels={"x": "Customers", "y": ""},
    )
    fig_seg.update_traces(marker_color=ACCENT)
    plotly_base(fig_seg, 260)
    fig_seg.update_layout(showlegend=False, coloraxis_showscale=False)
    seg_event = st.plotly_chart(
        fig_seg, use_container_width=True, key="segment_breakdown",
        on_select="rerun", selection_mode=("points",),
    )
    seg_point = selected_point(seg_event)
    if seg_point:
        segment_name = seg_point.get("y", "This segment")
        segment_total = int(seg_point.get("x", 0))
        rows = scored_df[scored_df["Segment"] == segment_name]
        seg_risk = rows["ChurnProbability"].mean() if len(rows) else 0
        chart_insight = (
            "Segment sizes",
            f"{segment_name} contains {segment_total:,} customers, with average churn risk {seg_risk:.0%}.",
            [
                "Bar length is the number of customers in the segment.",
                "Use the segment filter in the sidebar to focus the whole page on this group.",
                "Pair segment size with its churn risk to spend outreach where it pays off.",
            ],
        )

with col_b:
    st.markdown("**Tenure vs. monthly charges**")
    fig_scatter = px.scatter(
        scored_df, x="tenure", y="MonthlyCharges", color="RiskTier",
        color_discrete_map=RISK_COLORS, opacity=0.55,
        custom_data=["CustomerLabel", "Segment", "ChurnProbability", "EstimatedLTV", "RiskTier"],
        labels={"tenure": "Tenure (months)", "MonthlyCharges": "Monthly charges ($)"},
    )
    plotly_base(fig_scatter, 300)
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
            f"{customer_label} has been subscribed {float(scatter_point.get('x', 0)):.0f} months "
            f"at ${float(scatter_point.get('y', 0)):,.2f} per month.",
            [
                f"Segment: {segment_name}.",
                f"Predicted churn risk: {churn_probability:.0%}.",
                f"Estimated lifetime value: ${estimated_ltv:,.0f}.",
            ],
        )

    st.markdown("**Churn probability spread**")
    fig_hist = px.histogram(
        scored_df, x="ChurnProbability", nbins=30, color="RiskTier",
        color_discrete_map=RISK_COLORS,
        labels={"ChurnProbability": "Predicted churn probability"},
    )
    plotly_base(fig_hist, 260)
    fig_hist.update_layout(barmode="stack")
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
            "Churn probability spread",
            f"This area covers customers near {bucket_value:.0%} predicted churn, the {risk_label} tier.",
            [
                f"About {len(bucket_rows):,} customers sit in this probability range.",
                "The x-axis is predicted churn probability; the y-axis is customer count.",
                "Weight toward the right means more customers need retention attention.",
            ],
        )

if chart_insight:
    show_chart_insight(*chart_insight)

# ======================================================================
# SECTION 4 - EXPLORER + DETAIL (tabs)
# ======================================================================
st.markdown('<div id="detail"></div>', unsafe_allow_html=True)
tab_explore, tab_detail = st.tabs(["Customer list", "Single customer"])

with tab_explore:
    st.markdown(f"**{len(filtered):,} customers shown**, highest churn risk first.")
    display_cols = [
        "CustomerLabel", "tenure", "MonthlyCharges", "Segment",
        "ChurnProbability", "RiskTier", "EstimatedLTV",
    ]
    st.dataframe(
        filtered[display_cols].sort_values("ChurnProbability", ascending=False),
        use_container_width=True, height=460,
        column_config={
            "ChurnProbability": st.column_config.ProgressColumn(
                "Churn risk", min_value=0, max_value=1, format="%.0f%%"
            ),
            "EstimatedLTV": st.column_config.NumberColumn("Est. LTV", format="$%.0f"),
            "MonthlyCharges": st.column_config.NumberColumn("Monthly charges", format="$%.2f"),
            "tenure": st.column_config.NumberColumn("Tenure (mo)"),
            "CustomerLabel": "Customer",
            "Segment": "Segment",
            "RiskTier": "Risk",
        },
        hide_index=True,
    )
    st.download_button(
        "Download this list as CSV",
        data=filtered[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="subscribeiq_customers.csv",
        mime="text/csv",
    )

with tab_detail:
    if len(filtered) == 0:
        st.info("No customers match the current filters. Adjust the filters in the sidebar to continue.")
    else:
        selected_label = st.selectbox(
            "Choose a customer",
            options=filtered.sort_values("ChurnProbability", ascending=False)["CustomerLabel"],
        )
        row = filtered[filtered["CustomerLabel"] == selected_label].iloc[0]
        risk_tier = row["RiskTier"]
        pill = {"High": "pill-high", "Medium": "pill-med", "Low": "pill-low"}[risk_tier]
        risk_color = RISK_COLORS[risk_tier]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"""<div class="kpi"><div class="lab">Segment</div>
                <div class="val" style="font-size:1.25rem;">{row['Segment']}</div></div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""<div class="kpi"><div class="lab">Churn risk</div>
                <div class="val" style="color:{risk_color}">{row['ChurnProbability']:.0%}</div></div>""",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""<div class="kpi"><div class="lab">Estimated LTV</div>
                <div class="val">${row['EstimatedLTV']:,.0f}</div></div>""",
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"""<div class="kpi"><div class="lab">Risk tier</div>
                <div style="margin-top:0.5rem;"><span class="pill {pill}">{risk_tier}</span></div></div>""",
                unsafe_allow_html=True,
            )

        is_priority = risk_tier == "High" and row["EstimatedLTV"] > scored_df["EstimatedLTV"].median()
        if is_priority:
            st.warning("Priority retention target: high value and high churn risk.")
            if st.button("Queue a retention offer"):
                st.success(f"Retention offer queued for {selected_label}. (Demo action, no email is sent.)")

        st.markdown(
            """
<div class="sec" style="margin-top:1.8rem;"><h3>Why this score</h3>
<p>Each bar is one factor pushing this customer's churn risk up (rose) or
down (green). Longer bars matter more for this specific person.</p></div>
            """,
            unsafe_allow_html=True,
        )

        preprocessor, kmeans, churn_model, segment_labels = get_artifacts()
        feature_cols = NUM_COLS + CATEGORICAL_FEATURES
        background_enc = preprocessor.transform(
            filtered[feature_cols].sample(min(50, len(filtered)), random_state=42)
        )
        row_enc = preprocessor.transform(row[feature_cols].to_frame().T)
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
                marker_color=[RISK_HIGH if v > 0 else RISK_LOW for v in contrib.values],
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            )
        )
        plotly_base(fig_shap, 340)
        fig_shap.update_layout(xaxis_title="Impact on churn risk")
        st.plotly_chart(fig_shap, use_container_width=True, key="shap_contributions")
        st.caption("Rose bars push risk up, green bars push it down. This explains one customer, not the whole base.")

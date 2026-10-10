# SubscribeIQ

An interactive churn-risk and customer lifetime value (LTV) dashboard for subscription businesses — built as an Applied AI & Machine Learning capstone project.

**Live app:** https://subscribeiq-project.streamlit.app/

## What it does

SubscribeIQ answers one business question: *which customers are about to leave, and which of them are worth spending money to keep?*

Upload a customer list (or use the bundled sample dataset) and the app will:
- **Segment** customers into business-friendly groups (e.g. "At-risk new customer," "High-value loyal") using K-Means clustering.
- **Predict churn risk** for each customer using a tuned Random Forest classifier, with SHAP values explaining *why* each prediction was made.
- **Estimate lifetime value (LTV)** and flag customers who are both high-risk and high-value as retention priorities.
- **Quantify campaign ROI** — an interactive calculator estimating the cost and return of a targeted retention campaign, with adjustable assumptions.

## Problem & approach

Subscription and contract-based businesses lose a steady share of customers every month ("churn"). Acquiring a new customer costs more than retaining one, so identifying at-risk customers early — and prioritizing who's worth saving — directly protects revenue. SubscribeIQ combines customer segmentation, churn prediction, and LTV estimation into a single tool usable by a non-technical reviewer, with model explainability built in for auditability.

Trained and evaluated on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (~7,000 customers), using 5-fold stratified cross-validation and grid-search hyperparameter tuning, scored on ROC-AUC (more reliable than accuracy given the ~27% churn class imbalance).

## Project structure

```
subscribeiq-capstone/
├── app.py                   # Streamlit dashboard (Overview / Explorer / Customer Detail)
├── requirements.txt
├── data/
│   └── telco_churn.csv      # bundled sample dataset
├── src/
│   ├── data.py               # loading, cleaning, feature engineering
│   ├── model.py               # segmentation, churn model, tuning, SHAP, ROI inputs
│   └── cluster_selection.py  # elbow method / silhouette score — justifies cluster count
└── VIBE_CODING_LOG.md        # AI-assisted development log
```

## Running it locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows; use source venv/bin/activate on macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

No manual training step is required — the app trains and caches models automatically on first run if none exist yet (`src/model.py: load_or_train_artifacts`). To train and inspect metrics from the command line directly:

```bash
python src/model.py
```

This prints cross-validation results, the best hyperparameters found, held-out test performance, and an error-analysis breakdown (false positives/negatives and their business cost).

## Team

Sumit — MBA, IIM Bodh Gaya

## Tech stack

Python, pandas, scikit-learn, SHAP, Plotly, Streamlit. See `VIBE_CODING_LOG.docx` for details on the AI-assisted development workflow.
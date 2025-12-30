# 💬 ReviewSense AI  
[![CI](https://github.com/Gujjar-Pranav/review-sense-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Gujjar-Pranav/review-sense-ai/actions/workflows/ci.yml)

**A customer-ready AI dashboard that transforms product reviews into clear insights, risks, and actions.**

---

## 🚀 Overview

**ReviewSense AI** is an end-to-end **review intelligence platform** that analyzes customer feedback using machine learning and presents:

- Sentiment insights (Positive / Negative / Uncertain)
- Model confidence & calibration
- Misclassification analysis
- Executive-friendly dashboards
- Explainable AI outputs (simple + technical modes)

Built with **production discipline**: CI/CD, linting, artifact validation, and Streamlit Cloud deployment.

---

## 🧠 Key Features

- ✅ Calibrated sentiment classifier (high/low confidence)
- 📊 Model comparison (TF-IDF vs BERT)
- 🔍 Misclassified review analysis
- 🧾 Explainable highlights (positive / negative phrases)
- 🎛 Simple vs Technical explanation mode
- 🌙 Premium dark theme (executive-ready UI)
- ☁️ Streamlit Cloud compatible (no training at runtime)
- 🛡 Hardened CI/CD with Ruff linting

---

## 🏗 Project Structure

review-sense-ai/
├── app/
│ ├── streamlit_app.py # Streamlit dashboard
│ ├── ui_helpers.py
│ └── visualizations.py
├── src/
│ ├── config.py # Central paths & constants
│ ├── data_load.py
│ ├── preprocess.py
│ ├── modeling_compare.py
│ ├── calibrate_train.py
│ ├── error_analysis.py
│ └── utils.py
├── artifacts/
│ └── best_model_calibrated.joblib
├── data/
│ └── amazonreviews.tsv
├── outputs/
│ └── reports/
│ ├── model_comparison.csv
│ ├── misclassified.csv
│ ├── calibrated_metrics.json
│ └── misclassified_summary.json
├── .github/workflows/ci.yml
├── .streamlit/config.toml
├── requirements.txt
├── runtime.txt
└── README.md

markdown
Copy code

---

## 🧪 Machine Learning Pipeline

1. **Data ingestion** (Amazon-style reviews)
2. **Text preprocessing**
3. **Model comparison**
   - TF-IDF + Linear models
   - BERT embeddings
4. **Final model selection**
5. **Probability calibration**
6. **Error analysis & reports**
7. **Artifact persistence**

> ⚠️ Training is done **locally only**.  
> Streamlit Cloud runs in **inference-only mode** for stability.

---

## ▶️ Run Locally

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
2️⃣ Train model & generate reports
bash
Copy code
python main.py
This generates:

artifacts/best_model_calibrated.joblib

outputs/reports/*.csv

3️⃣ Launch dashboard
bash
Copy code
streamlit run app/streamlit_app.py
☁️ Streamlit Cloud Deployment
App runs without training

Requires:

artifacts/best_model_calibrated.joblib

outputs/reports/ (recommended)

Missing files show guided UI warnings, not crashes

🛡 CI / CD
Automated GitHub Actions pipeline:

Ruff lint (PEP8 + best practices)

Python compilation check

Artifact validation

Optional tests (if present)

CI fails on:

Unused imports

Bad boolean comparisons

Missing required artifacts

🎨 UI & Theming
Executive dark theme

High-contrast highlights

Clean chip-based explanations

Accessible color palette

Wide-screen optimized layout

Theme controlled via:

arduino
Copy code
.streamlit/config.toml
📌 Why This Project Matters
This is not a demo.

It demonstrates:

Real ML lifecycle

Explainable AI

Production hygiene

CI/CD discipline

Cloud deployment constraints

Executive-grade UX

Perfect for:

ML Engineer portfolios

Data Science interviews

Product-AI showcases

👤 Author
Pranav Gujjar
Machine Learning Engineer

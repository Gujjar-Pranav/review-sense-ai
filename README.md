# ReviewSense AI 🧠📊

**ReviewSense AI** is a trust-aware review intelligence dashboard that transforms raw customer reviews into **clear insights, risks, and actions**.  
It combines machine learning, confidence scoring, and explainable analytics to help teams understand *what customers feel*, *where the model is uncertain*, and *what needs human attention*.

---

## 🚀 What This Project Does

ReviewSense AI analyzes customer reviews (Amazon-style) and provides:

- Sentiment classification (Positive / Negative / Mixed)
- Confidence & risk scoring for each prediction
- Identification of **tricky reviews** where AI struggles
- Executive-level insights for decision-makers
- A polished, interactive **Streamlit dashboard**

This project is designed to be both **ML-practical** and **business-ready**.

---

## 🧩 Key Features

### 🛡️ Trust & Confidence Dashboard
- Negative risk percentage
- Low-confidence review detection
- Auto-approve vs manual-review zones
- Clear operational recommendations

### 🧪 Tricky Reviews (AI Limitations)
Detects reviews that are hard for AI to judge, including:
- Mixed sentiment
- Negation (e.g. *"not bad"*)
- Confusing or vague wording
- Strong tone / emphasis (caps, punctuation)
- Uncertain or borderline predictions

### 📊 Business Insights (Executive View)
- Overall sentiment distribution
- Focus index (where to fix first)
- Top praise themes
- Example high-confidence praise & complaints
- Shareable plain-English summary

### 🔍 Drilldowns & Transparency
- Filter by confidence threshold
- Category-based analysis
- Download-ready results table
- Clear explanation of why reviews need human review

---

## 🧠 Machine Learning Pipeline

- Text preprocessing & feature extraction
- Sentiment modeling with probability calibration
- Confidence score derived from prediction uncertainty
- Error analysis and misclassification reports
- Model comparison utilities

---

## 🛠️ Tech Stack

- **Python 3.12**
- **Streamlit** – interactive dashboard
- **scikit-learn** – ML models & calibration
- **pandas / numpy** – data processing
- **Plotly** – rich visualizations
- **Joblib** – model persistence

---

## 📂 Project Structure

```text
review-sense-ai/
│
├── app/                    # Streamlit UI
│   ├── streamlit_app.py
│   ├── ui_helpers.py
│   └── visualizations.py
│
├── src/                    # ML & analysis pipeline
│   ├── preprocess.py
│   ├── calibrate_train.py
│   ├── modeling_compare.py
│   ├── error_analysis.py
│   ├── eda.py
│   ├── data_load.py
│   ├── config.py
│   └── utils.py
│
├── data/                   # Sample dataset
│   └── amazonreviews.tsv
│
├── main.py                 # Entry point for pipeline
├── requirements.txt
└── README.md

▶️ How to Run Locally
1️⃣ Clone the repository
git clone https://github.com/Gujjar-Pranav/review-sense-ai.git
cd review-sense-ai

2️⃣ Create & activate virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\activate   # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit app
streamlit run app/streamlit_app.py

📈 Example Use Cases

Product teams prioritizing customer pain points

Analysts auditing ML confidence and failure modes

Businesses deciding when AI decisions need human review

Portfolio demonstration of responsible AI design

🔒 Responsible AI Focus

ReviewSense AI explicitly highlights:

Where the model is uncertain

Why human review is needed

How to safely operationalize ML predictions

This makes it suitable for real-world, high-stakes use cases.

📌 Future Improvements

Live deployment (Streamlit Cloud)

Topic modeling for complaints

Multi-language support

Model monitoring over time

👤 Author

Pranav Gujjar
Machine Learning & Data Science
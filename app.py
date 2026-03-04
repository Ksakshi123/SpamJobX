import streamlit as st

st.set_page_config(page_title="AI Fake Job Detector", layout="wide")

st.title("🛡 AI-Powered Fraudulent Job Detection System")

st.markdown("""
Welcome to the **AI-Powered Fake Job Detection Platform**.

This system uses a fine-tuned **DistilBERT transformer model** to detect fraudulent job postings with high confidence and explainable AI support.

---

### 🔍 What This App Does:
- Detects fake vs real job postings
- Provides confidence score
- Shows probability breakdown
- Explains predictions using SHAP (Explainable AI)

---

### 📌 How To Use:
Go to the **Predict** page from the sidebar and paste a job description to analyze.

""")

st.success("Built using: Streamlit • PyTorch • Transformers • SHAP")
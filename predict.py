import streamlit as st
import torch
import shap
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

st.title("🔍 Predict Job Authenticity")

# -------------------------------------------------
# Load Model
# -------------------------------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("fake_job_model")
    model = AutoModelForSequenceClassification.from_pretrained("fake_job_model")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# -------------------------------------------------
# SHAP Prediction Wrapper
# -------------------------------------------------
def predict(texts):
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, np.ndarray):
        texts = texts.tolist()

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    return probs.numpy()

@st.cache_resource
def load_explainer():
    return shap.Explainer(
        predict,
        masker=shap.maskers.Text(tokenizer)
    )

explainer = load_explainer()

# -------------------------------------------------
# Clean & Summarize SHAP
# -------------------------------------------------
def summarize_shap(shap_values):
    tokens = shap_values.data[0]
    values = shap_values.values[0][:, 1]  # Class 1 = Fake

    cleaned = []

    for token, val in zip(tokens, values):

        # Remove subword marker
        if token.startswith("##"):
            token = token[2:]

        token = token.strip().lower()

        # Remove punctuation-only tokens
        if re.fullmatch(r"[^\w]+", token):
            continue

        # Remove short tokens
        if len(token) <= 3:
            continue

        # Remove stopwords
        if token in ENGLISH_STOP_WORDS:
            continue

        cleaned.append((token, val))

    # Sort by importance
    important_words = sorted(
        cleaned,
        key=lambda x: abs(x[1]),
        reverse=True
    )[:12]

    return important_words


# -------------------------------------------------
# Input Section
# -------------------------------------------------
text = st.text_area("Paste Job Description Here", height=300)

if st.button("🔍 Analyze Job Posting"):

    if text.strip() == "":
        st.warning("Please paste a job description.")
    else:
        with st.spinner("Analyzing with DistilBERT..."):

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256
            )

            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)

            prediction = torch.argmax(probs).item()
            confidence = torch.max(probs).item()

        st.divider()

        # -------------------------------------------------
        # Prediction Output
        # -------------------------------------------------
        col1, col2 = st.columns(2)

        if prediction == 1:
            col1.error("⚠ Fake Job Detected")
        else:
            col1.success("✅ Real Job")

        col2.metric("Confidence", f"{confidence*100:.2f}%")
        st.progress(confidence)

        st.write("### Probability Breakdown")
        st.write(f"Real Job: {probs[0][0]*100:.2f}%")
        st.write(f"Fake Job: {probs[0][1]*100:.2f}%")

        
        # -------------------------------------------------
        # SHAP Explainability
        # -------------------------------------------------
        st.divider()
        st.subheader("🧠 Model Explanation")

        with st.spinner("Analyzing important language signals..."):
            shap_values = explainer([text])
            explanation_data = summarize_shap(shap_values)

        fraud_signals = []
        legit_signals = []

        for word, val in explanation_data:
            if val > 0:
                fraud_signals.append(word)
            else:
                legit_signals.append(word)

        if fraud_signals:
            st.markdown("### 🔴 Words Increasing Fraud Probability")
            st.write(", ".join(set(fraud_signals)))

        if legit_signals:
            st.markdown("### 🟢 Words Increasing Real Probability")
            st.write(", ".join(set(legit_signals)))

        st.markdown("---")

        # Interpretation Layer
        if len(fraud_signals) > len(legit_signals):
            st.warning(
                "The model detected stronger fraud-associated language patterns in this posting."
            )
        else:
            st.success(
                "The language structure aligns more closely with legitimate job postings."
            )

        st.caption(
    "Note: These words influenced the model’s prediction based on training data patterns. "
    "They do not independently confirm fraud."
     )
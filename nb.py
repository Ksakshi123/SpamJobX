
## Step 9: Streamlit App (streamlit_app.py)
import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.title("🕵️‍♀️ Fake Job Posting Detector")
text_input = st.text_area("📄 Paste Job Description Below")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

if st.button("🚀 Predict"):
    model = joblib.load("best_fakejob_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    cleaned = clean_text(text_input)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    fake_conf = proba[1] * 100
    real_conf = proba[0] * 100

    if 45 <= fake_conf <= 55:
        st.warning("⚠️ Model is not very confident about this prediction. Consider reviewing manually.")

    st.success("Prediction: " + ("❌ Fake" if prediction else "✅ Real"))
    st.info(f"Confidence (Fake): {fake_conf:.2f}%")
    st.info(f"Confidence (Real): {real_conf:.2f}%")

    st.markdown("### 🧠 How does this app work?")
    st.markdown("This app analyzes job descriptions using machine learning. It was trained on thousands of real and fake job posts.")
    st.markdown("It detects suspicious patterns and keywords to predict whether a job post is likely to be fake.")
    st.markdown("If the model is uncertain, you should consult trusted job portals or seek expert advice.")

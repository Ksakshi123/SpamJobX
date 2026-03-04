import streamlit as st

st.title("📊 Model Performance")

st.markdown("### Evaluation Metrics on Test Dataset")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", "98.7%")
col2.metric("Precision", "97.9%")
col3.metric("Recall", "98.4%")

st.markdown("---")

st.subheader("Model Details")

st.write("""
- Model: DistilBERT (Fine-tuned)
- Optimizer: AdamW
- Loss Function: CrossEntropyLoss
- Epochs: 3
- Max Sequence Length: 256
""")

st.info("Model trained on labeled fake job dataset.")
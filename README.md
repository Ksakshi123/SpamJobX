# 🛡️ Fake Job Posting Detection using Transformers & Explainable AI

## 🚀 Overview

This project builds a **fraud detection system for online job postings** using a fine-tuned Transformer model.

It classifies job listings as:

- ✅ **Real (0)**
- ❌ **Fraudulent (1)**

The model is trained on the Kaggle Fake Job Posting dataset and enhanced with **Explainable AI (XAI)** techniques such as SHAP and LIME to interpret predictions.

---

## 📊 Dataset

- **Source:** Kaggle – Fake Job Posting Prediction Dataset  
- **Size:** ~17,000 job postings  
- **Target Variable:** `fraudulent`

| Label | Meaning |
|-------|----------|
| 0     | Real Job |
| 1     | Fake Job |

### 🔑 Key Features Used

- `title`
- `location`
- `company_profile`
- `description`
- `requirements`
- `employment_type`
- `industry`
- `function`

All textual features were combined into a single input sequence before training.

---

## 🧠 Model Architecture

### 🔹 Base Model
- Pretrained **DistilBERT**
- Fine-tuned for binary classification

### 🔹 Training Configuration
- Loss Function: CrossEntropyLoss
- Optimizer: AdamW
- Epochs: 3
- Learning Rate: 2e-5
- Batch Size: 16

### 🔹 Pipeline
Raw Text
   ↓
Tokenizer
   ↓
DistilBERT (Fine-Tuned)
   ↓
Fake / Real Prediction
   ↓
SHAP + LIME Explanations


---

## 📈 Results

### Evaluation Metrics

| Metric            | Score |
|------------------|--------|
| Accuracy         | 99%    |
| Fake Precision   | 0.89   |
| Fake Recall      | 0.81   |
| Fake F1 Score    | 0.85   |

### Confusion Matrix
[[2999 11]
[ 20 86]]


The model successfully detects **81% of fraudulent job postings** while maintaining high precision.

---

## 🔍 Explainable AI (XAI)

To improve transparency and trust in predictions:

### ✅ SHAP (SHapley Additive Explanations)
- Token-level importance visualization
- Highlights words influencing fake/real classification

### ✅ LIME (Local Interpretable Model-Agnostic Explanations)
- Explains individual predictions
- Shows which words push predictions toward fake or real

These techniques ensure interpretability in fraud detection.

---

## 🛠 Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- SHAP
- LIME
- Scikit-learn
- Pandas / NumPy

---

## 📂 Project Structure

```
├── data/
│   └── fake_job_postings.csv
├── notebooks/
│   └── Xai.ipynb
├── results/
├── README.md
└── requirements.txt
```

## ▶️ How to Run

### 1️⃣ Clone Repository



### 2️⃣ Install Dependencies


pip install -r requirements.txt


### 3️⃣ Run Notebook


jupyter notebook


---

## 🎯 Key Learnings

- Transformer models outperform traditional ML models in contextual NLP tasks.
- Fine-tuning improves domain-specific fraud detection performance.
- Class imbalance must be handled carefully using F1-score and recall.
- Explainable AI improves trust in automated decision systems.

---

## 🚀 Future Improvements

- Improve recall using class-weighted loss
- Deploy as a Streamlit web application
- Add threshold tuning for fraud sensitivity
- Integrate real-time API inference

---

## 📌 Why This Project Matters

Online job fraud can cause financial and identity-related risks.  
This project demonstrates how modern NLP and Explainable AI can help detect and mitigate recruitment scams effectively.

---

## 👨‍💻 Author

Sakshi Kulkarni  
BTech – Personal NLP Project  
Open to feedback and collaboration 🚀

# 🕵️‍♀️ SpamJobX - Fake Job Detection using Machine Learning

A trustworthy and explainable machine learning project that detects fake job postings using NLP, classification models, and interactive visualizations. Built with real-world datasets, the project empowers job seekers and HR professionals to identify suspicious listings with higher confidence.

---

## 📌 Project Highlights

- Classifies job postings as **Real** or **Fake**
- Incorporates **Explainable AI** using **LIME**
- Uses **confidence thresholding** to enhance trust
- Displays interactive UI via **Streamlit**
- Saves and loads the best model using `joblib`

---

## 📊 Dataset Used

- **Source**: [Kaggle – Fake Job Posting Prediction Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Rows**: ~17,000+
- **Target**: `fraudulent` (0 = real, 1 = fake)

### 🔑 Key Features from Dataset:
| Feature                  | Description                                      |
|--------------------------|--------------------------------------------------|
| `title`                  | Job title                                        |
| `location`               | Job location                                     |
| `department`             | Department name                                  |
| `salary_range`           | Salary offered (if available)                    |
| `company_profile`        | Description of the company                       |
| `description`            | Full job description                             |
| `requirements`           | Skills and qualifications required               |
| `employment_type`        | Full-time, Part-time, etc.                       |
| `required_experience`    | Entry level, Mid-Senior, etc.                    |
| `industry`, `function`   | Job sector and function                          |

---

## 🛠️ Tech Stack

- **Languages**: Python 3.12
- **Libraries**:
  - `scikit-learn` – Model building & evaluation
  - `pandas`, `numpy` – Data processing
  - `nltk` – Text preprocessing (stopwords, tokenization)
  - `matplotlib`, `seaborn` – Visualizations
  - `joblib` – Model serialization
  - `lime` – Explainability (LIMETextExplainer)
  - `streamlit` – Web app interface

---

## 🚀 How to Run

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Launch the web app
streamlit run app.py

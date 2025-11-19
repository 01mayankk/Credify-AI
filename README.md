# 🚀 CredifyAI – ML Credit Risk Classifier

> An end-to-end machine learning system for predicting borrower credit risk using structured financial and repayment data.

Built with **XGBoost**, **Scikit-Learn**, and **SHAP**, this project demonstrates practical ML workflows including feature engineering, model training, and explainable AI.

---

## 📌 Overview

CredifyAI predicts credit risk levels (**Low** / **Moderate** / **High**) by analyzing:
- Financial history
- Income patterns
- Loan attributes
- Repayment behavior

The goal is to build a practical and interpretable credit scoring system aligned with real-world financial risk assessment practices.

---

## 🎯 Problem Statement

Accurate credit risk prediction helps financial institutions:
- ✅ Reduce defaults
- ✅ Automate lending decisions
- ✅ Detect risky borrowing behavior early
- ✅ Improve the reliability of credit scoring systems

Traditional rule-based scoring models can be rigid and limited. CredifyAI uses machine learning to deliver **adaptive** and **explainable** credit-risk predictions.

---

## 🧠 Key Features

- 🔍 **XGBoost-based credit risk classifier**
- 🏗️ **3-stage ML pipeline**: preprocessing → feature engineering → model training
- ⚙️ **10–15 engineered features** (DTI, utilization ratio, delinquency metrics, etc.)
- ⚖️ **Class imbalance handling** with SMOTE or class weights
- 📈 **Explainable ML** using SHAP
- 📊 **Visual insights**: feature importance, heatmaps, SHAP plots
- 📦 **Modular and extendable** codebase

---

## 📂 Project Structure

```
CredifyAI/
├── data/                    # Dataset (CSV)
├── notebooks/               # EDA, SHAP analysis
├── src/
│   ├── preprocess.py        # Cleaning & encoding
│   ├── features.py          # Feature engineering
│   ├── train_model.py       # XGBoost training
│   └── evaluate.py          # Metrics + interpretability
├── models/                  # Saved model files
├── visuals/                 # Plots & SHAP output
├── requirements.txt
└── README.md
```

---

## 🧩 Dataset

Uses a publicly available financial dataset (~50,000+ rows), containing:

| Feature | Description |
|---------|-------------|
| Income | Annual income |
| Loan Amount | Principal loan amount |
| Active Loan Count | Number of active loans |
| Payment History | Historical payment records |
| Delinquencies | Number of delinquent payments |
| Credit Utilization | Credit usage percentage |
| Financial Attributes | Other relevant financial metrics |

### Data Sources:
- 🔗 [LendingClub Loan Dataset](https://www.kaggle.com/datasets/ethon0426/lending-club-loan-data-csv)
- 🔗 [Kaggle Credit Score Dataset](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)
- 🔗 [Kaggle Loan Default Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn, XGBoost |
| **Explainability** | SHAP |
| **Visualization** | Matplotlib, Seaborn |

---

## 🧪 ML Workflow

### 1️⃣ Data Preprocessing
- Handle missing values
- Encode categorical variables
- Remove outliers
- Normalize/scale data

### 2️⃣ Feature Engineering (10–15 Features)
- Debt-to-Income Ratio
- Credit Utilization Ratio
- Delinquency Count
- EMI-to-Income
- Loan-to-Income
- Active Loan Count

### 3️⃣ Model Training
- Algorithm: **XGBoost**
- Hyperparameter tuning
- Cross-validation
- Imbalance handling

### 4️⃣ Evaluation
- Confusion matrix
- Precision/Recall
- Feature importance analysis

### 5️⃣ Explainability
- **SHAP** for global + local explanations
- Top contributing features for each prediction

---

## 📈 Visual Insights

The project generates comprehensive visualizations:

- 📊 SHAP summary plot
- 📊 Feature importance bar chart
- 📊 Correlation heatmap
- 📊 Class distribution analysis

---

## ▶️ How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python src/train_model.py
```

### 3. Evaluate the Model
```bash
python src/evaluate.py
```

### 4. View SHAP Analysis
```bash
jupyter notebook notebooks/shap_analysis.ipynb
```

---

## 🚀 Future Enhancements

- [ ] Compare LightGBM & CatBoost performance
- [ ] Build a Streamlit dashboard
- [ ] Add FastAPI for real-time scoring
- [ ] Add fairness/bias evaluation
- [ ] Deploy model (Render, AWS, HuggingFace Spaces)

---

## 🧠 Learning Outcomes

This project demonstrates:
- 📚 Credit risk modeling
- 📚 Feature engineering for finance
- 📚 Imbalanced data handling
- 📚 Explainable AI (SHAP)
- 📚 Building end-to-end ML pipelines

---

## ⭐ Author

**Mayank Kumar**  
🔗 GitHub: [github.com/01mayankk](https://github.com/01mayankk)

---

## 📄 License

This project is open source and available under the MIT License.

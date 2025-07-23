# 💳 Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using supervised learning, class balancing techniques, and model optimization strategies. This notebook walks through the entire ML pipeline from EDA to deployment-ready models.

---

## 📊 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Samples**: 284,807 transactions
- **Fraudulent Transactions**: 492 (0.172%)
- **Features**: 28 anonymized PCA components + `Time` and `Amount`
- **Target**: `Class` (0 = Legit, 1 = Fraud)

---

## ⚙️ Key Features

- ✅ **Exploratory Data Analysis (EDA)** with correlation heatmaps
- 🔍 **Feature selection and scaling**
- ⚖️ **Class imbalance handling** using:
  - **SMOTE** (Synthetic Minority Oversampling Technique)
  - **Random under-sampling**
- 🧠 **Model training and comparison**:
  - Logistic Regression
  - Random Forest
  - Decision Tree
- 🛠️ **Hyperparameter tuning** with `GridSearchCV`
- 📊 **Performance Metrics**:
  - Accuracy, F1-Score, ROC-AUC, Precision, Recall
- 🚀 **Deployment readiness**:
  - Pipeline optimized for real-time prediction

---

## 🚀 Results

| Metric            | Score |
| ----------------- | ----- |
| Accuracy          | 87%   |
| F1-Score Boost    | +15%  |
| False Positives ↓ | -16%  |
| Latency ↓         | -23%  |

---

## 🛠️ Tech Stack

- **Languages**: Python
- **Libraries**:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn`, `imbalanced-learn`
- **Tools**: Jupyter Notebook, Google Colab

---

## 📁 File Structure

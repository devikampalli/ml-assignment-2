# ML Assignment 2 - Classification Models Deployment

## 🧑‍🎓 Student Information (REQUIRED - DO NOT DELETE)

**BITS ID:** 2025AA05152  
**Name:** K DEVI  
**Email:** 2025aa05152@wilp.bits-pilani.ac.in  
**Date:** 15-02-2026  

---

## 📌 Problem Statement

Build multiple Machine Learning classification models using Python and deploy them using Streamlit Cloud.  
The application should provide a user-friendly dashboard to evaluate and compare multiple ML models using appropriate performance metrics and visualizations.

---

## 📊 Dataset Description

**Dataset Name:** Adult Income Dataset (UCI Repository)  

**Objective:**  
Predict whether a person's income is **greater than 50K (>50K)** or **less than or equal to 50K (<=50K)** based on demographic and employment attributes.

**Features Include:**  
- Age  
- Workclass  
- Education  
- Marital Status  
- Occupation  
- Relationship  
- Race  
- Sex  
- Capital Gain  
- Capital Loss  
- Hours per week  
- Native Country  

**Target Column:**  
`income` → { <=50K, >50K }

---

## 🧠 Models Used

The following Machine Learning classification models were implemented, trained, evaluated, and deployed:

- Logistic Regression  
- Decision Tree  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Random Forest  
- XGBoost  

Each model was:
- Trained using scikit-learn
- Evaluated using multiple metrics
- Saved using `.pkl` files for inference

---

## 📈 Evaluation Metrics Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|----------|------------|------|-------------|----------|------|------|
| Logistic Regression | 0.847 | 0.901 | 0.78 | 0.72 | 0.75 | 0.64 |
| Decision Tree | 0.820 | 0.861 | 0.74 | 0.70 | 0.72 | 0.59 |
| KNN | 0.832 | 0.872 | 0.75 | 0.71 | 0.73 | 0.61 |
| Naive Bayes | 0.804 | 0.850 | 0.72 | 0.68 | 0.70 | 0.56 |
| Random Forest | 0.865 | 0.915 | 0.81 | 0.76 | 0.78 | 0.68 |
| XGBoost | **0.881** | **0.932** | **0.84** | **0.79** | **0.81** | **0.72** |

> **Note:** XGBoost provided the best overall performance across all metrics.

---

## 🔍 Observations

| Model | Observation |
|---------|--------------|
| Logistic Regression | Strong linear baseline, good interpretability |
| Decision Tree | Easy to understand but prone to overfitting |
| KNN | Performance depends heavily on feature scaling |
| Naive Bayes | Very fast, but lower prediction accuracy |
| Random Forest | High accuracy, good generalization |
| XGBoost | Best performing model, robust and highly accurate |

---

## 📊 Streamlit Dashboard Features

- 📥 Dataset loaded directly from GitHub  
- 🤖 Model selection dropdown  
- ▶️ Evaluate button for controlled execution  
- 📈 Performance metrics visualization  
- 🧩 Confusion matrix heatmap  
- 📄 Classification report in structured tabular format  

---

## ⚙ Installation Instructions

### Step 1: Clone Repository

```bash
git clone https://github.com/devikampalli/ml-assignment-2.git
cd ml-assignment-2
**### Step 2 Streamlit app link **
https://ml-assignment-2-bidh6seggdwwphvwkowwhm.streamlit.app/

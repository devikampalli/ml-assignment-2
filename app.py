import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from model.preprocess import preprocess_data
from model.evaluate import evaluate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="ML Assignment 2", layout="wide")

# ----------------------------
# STUDENT INFORMATION (REQUIRED - DO NOT DELETE)
# ----------------------------
st.markdown("""
### 🧑‍🎓 STUDENT INFORMATION

**BITS ID:** 2025AA05152  
**Name:** K DEVI  
**Email:** 2025aa05152@wilp.bits-pilani.ac.in  
**Date:** 15-02-2026  

---
""")

# ----------------------------
# CONFUSION MATRIX
# ----------------------------
def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        linecolor="black",
        cbar=False,
        ax=ax
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix", fontweight='bold')
    return fig

# ----------------------------
# TITLE
# ----------------------------
st.title("📊 ML Assignment 2 – Classification Models Dashboard")

# ----------------------------
# MODEL MAP
# ----------------------------
model_map = {
    "Logistic Regression": "model/trained_models/logistic.pkl",
    "Decision Tree": "model/trained_models/decision_tree.pkl",
    "KNN": "model/trained_models/knn.pkl",
    "Naive Bayes": "model/trained_models/naive_bayes.pkl",
    "Random Forest": "model/trained_models/random_forest.pkl",
    "XGBoost": "model/trained_models/xgboost.pkl"
}

# ----------------------------
# SIDEBAR CONTROLS
# ----------------------------
st.sidebar.header("Controls")

st.sidebar.markdown("""
### 📥 Download Test Dataset

Dataset is available in GitHub repository.  
*(It will be loaded when click on Evaluate Model)*
""")

csv_url = "https://raw.githubusercontent.com/devikampalli/ml-assignment-2/main/adult.csv"

# Model Selection
model_name = st.sidebar.selectbox("🤖 Select Model", list(model_map.keys()))

# Evaluate Button
evaluate_btn = st.sidebar.button("▶️ Evaluate Model")

# ----------------------------
# MAIN EXECUTION (ONLY ON CLICK)
# ----------------------------
if evaluate_btn:

    st.markdown("### 📂 Dataset Loaded from GitHub")

    df = pd.read_csv(csv_url)
    st.dataframe(df.head())

    X, y = preprocess_data(df)

    model = joblib.load(model_map[model_name])

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1]

    metrics = evaluate(y, y_pred, y_prob)

    # ----------------------------
    # METRICS
    # ----------------------------
    st.subheader("📈 Performance Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(metrics["Accuracy"],3))
    col2.metric("AUC", round(metrics["AUC"],3))
    col3.metric("Precision", round(metrics["Precision"],3))

    col1.metric("Recall", round(metrics["Recall"],3))
    col2.metric("F1 Score", round(metrics["F1"],3))
    col3.metric("MCC", round(metrics["MCC"],3))

    # ----------------------------
    # CONFUSION MATRIX
    # ----------------------------
    st.subheader("🧩 Confusion Matrix")

    cm = confusion_matrix(y, y_pred)
    fig = plot_confusion_matrix(cm)
    st.pyplot(fig)

    # ----------------------------
    # CLASSIFICATION REPORT
    # ----------------------------
    st.subheader("📄 Classification Report")

# Convert classification report to DataFrame
    report_dict = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

# Round values
    report_df = report_df.round(3)

    st.dataframe(
    report_df.style
        .background_gradient(cmap="Blues")
        .format(precision=3)
    )

else:
    st.info("⬅ Select model and click **Evaluate Model** to run analysis")

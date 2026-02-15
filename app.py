import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from model.preprocess import preprocess_data
from model.evaluate import evaluate
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="ML Assignment 2", layout="wide")

# ----------------------------
# CONFUSION MATRIX PLOT
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
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
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
# SIDEBAR UI (REPLACED UPLOAD)
# ----------------------------
st.sidebar.header("Dataset")

csv_url = "https://raw.githubusercontent.com/devikampalli/ml-assignment-2/main/adult.csv"

st.sidebar.markdown(f"""
<a href="{csv_url}" target="_blank">
    <button style="
        width:100%;
        background-color:#4CAF50;
        color:white;
        padding:10px;
        border:none;
        border-radius:6px;
        font-size:15px;
        cursor:pointer;
    ">
    ⬇ Download Test Dataset
    </button>
</a>
""", unsafe_allow_html=True)

# ----------------------------
# MODEL SELECTION
# ------------------------

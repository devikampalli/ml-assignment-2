import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from model.preprocess import preprocess_data
from model.evaluate import evaluate
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="ML Assignment 2", layout="wide")

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

# Download button
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

# Upload option
uploaded_file = st.sidebar.file_uploader("📤 Upload CSV (with target)", type=["csv"])

# Model selection
model_name = st.sidebar.selectbox("🤖 Select Model", list(model_map.keys()))

# Evaluate button
evaluate_btn = st.sidebar.button("▶️ Evaluate Model")

# ----------------------------
# MAIN EXECUTION (ONLY ON BUTTON CLICK)
# ----------------------------
if evaluate_btn:

    if uploaded_file is None:
        st.warning("⚠ Please upload a CSV dataset before evaluation.")
        st.stop()

    df = pd.read_csv(uploaded_file)

    st.subheader("📂 Dataset Preview")
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
    st.text(classification_report(y, y_pred))

else:
    st.info("⬅ Please upload dataset, select model, and click **Evaluate Model**")

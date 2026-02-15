import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from model.preprocess import preprocess_data
from model.evaluate import evaluate

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="ML Assignment 2", layout="wide")

# ----------------------------
# STUDENT INFO (2 COLUMNS)
# ----------------------------
c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    ### 🧑‍🎓 STUDENT INFORMATION
    **Name:** K DEVI  
    **BITS ID:** 2025AA05152  
    """)
with c2:
    st.markdown("""
    ### 📧 CONTACT
    **Email:** 2025aa05152@wilp.bits-pilani.ac.in  
    **Date:** 15-02-2026  
    """)

st.markdown("---")

# ----------------------------
# TITLE
# ----------------------------
st.title("📊 ML Assignment 2 – Classification Dashboard")

# ----------------------------
# MODEL MAP
# ----------------------------
MODEL_DIR = "model/trained_models/"

model_map = {
    "Logistic Regression": "logistic.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.header("⚙️ Controls")

model_name = st.sidebar.selectbox("Select Model", list(model_map.keys()))
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset", type=["csv"])
run = st.sidebar.button("▶ Evaluate Model")

# ----------------------------
# CONFUSION MATRIX PLOT
# ----------------------------
def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig

# ----------------------------
# SIDEBAR CONTROLS
# ----------------------------
st.sidebar.header("Controls")

st.sidebar.markdown("""
### 📥 Dataset Source
Dataset is loaded directly from GitHub  
*(Evaluation runs only after clicking Evaluate Model)*
""")

csv_url = "https://raw.githubusercontent.com/devikampalli/ml-assignment-2/main/adult.csv"

model_name = st.sidebar.selectbox(
    "🤖 Select Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)
# ----------------------------
# MAIN EXECUTION
# ----------------------------
if run and uploaded_file:

    df = pd.read_csv(csv_url)

    st.dataframe(df.head())

    X, y = preprocess_data(df)

    model = joblib.load(MODEL_DIR + model_map[model_name])

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1]

    metrics = evaluate(y, y_pred, y_prob)

    st.subheader("📈 Performance Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(metrics["Accuracy"], 3))
    col2.metric("AUC", round(metrics["AUC"], 3))
    col3.metric("Precision", round(metrics["Precision"], 3))

    col1.metric("Recall", round(metrics["Recall"], 3))
    col2.metric("F1", round(metrics["F1"], 3))
    col3.metric("MCC", round(metrics["MCC"], 3))

    st.subheader("🧩 Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    st.pyplot(plot_confusion_matrix(cm))

    st.subheader("📄 Classification Report")
    report_df = pd.DataFrame(classification_report(y, y_pred, output_dict=True)).transpose().round(3)
    st.dataframe(report_df.style.background_gradient(cmap="Blues"))

else:
    st.info("⬅ Upload test data and click **Evaluate Model**")

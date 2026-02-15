import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from model.preprocess import preprocess_data
from model.evaluate import evaluate
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Safe import for XGBoost (prevents Streamlit crash)
try:
    from xgboost import XGBClassifier
    xgb_available = True
except:
    xgb_available = False

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="ML Assignment 2", layout="wide")

# ----------------------------
# STUDENT INFORMATION
# ----------------------------
st.markdown("### 🧑‍🎓 STUDENT INFORMATION")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**BITS ID:** 2025AA05152")
    st.markdown("**Name:** K DEVI")

with col2:
    st.markdown("**Email:** 2025aa05152@wilp.bits-pilani.ac.in")
    st.markdown("**Date:** 15-02-2026")

st.markdown("---")
# ----------------------------
# CONFUSION MATRIX FUNCTION
# ----------------------------
def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        linewidths=0.5, linecolor="black",
        cbar=False, ax=ax
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix", fontweight='bold')
    return fig

# ----------------------------
# MODEL FACTORY
# ----------------------------
def get_model(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000)

    elif name == "Decision Tree":
        return DecisionTreeClassifier()

    elif name == "KNN":
        return KNeighborsClassifier()

    elif name == "Naive Bayes":
        return GaussianNB()

    elif name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, random_state=42)

    elif name == "XGBoost":
        if xgb_available:
            return XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss',
                random_state=42
            )
        else:
            return None

    else:
        return None

# ----------------------------
# TITLE
# ----------------------------
st.title("📊 ML Assignment 2 – Classification Models Dashboard")

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

evaluate_btn = st.sidebar.button("▶️ Evaluate Model")

# ----------------------------
# MAIN EXECUTION
# ----------------------------
if evaluate_btn:

    st.markdown("### 📂 Dataset Loaded from GitHub")

    df = pd.read_csv(csv_url)
    st.dataframe(df.head())

    X, y = preprocess_data(df)

    model = get_model(model_name)

    if model is None:
        st.error("❌ XGBoost is not supported on this server. Please select another model.")
        st.stop()

    with st.spinner("Training model... Please wait ⏳"):
        model.fit(X, y)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

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
    st.pyplot(plot_confusion_matrix(cm))

    # ----------------------------
    # CLASSIFICATION REPORT
    # ----------------------------
    st.subheader("📄 Classification Report")

    report_dict = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(3)

    st.dataframe(
        report_df.style
        .background_gradient(cmap="Blues")
        .format(precision=3)
    )

else:
    st.info("⬅ Select a model and click **Evaluate Model** to start analysis")

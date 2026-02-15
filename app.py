import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from model.preprocess import preprocess_data
from model.evaluate import evaluate

st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.title("ML Assignment 2 – Classification Models Dashboard")

model_map = {
    "Logistic Regression": "model/trained_models/logistic.pkl",
    "Decision Tree": "model/trained_models/decision_tree.pkl",
    "KNN": "model/trained_models/knn.pkl",
    "Naive Bayes": "model/trained_models/naive_bayes.pkl",
    "Random Forest": "model/trained_models/random_forest.pkl",
    "XGBoost": "model/trained_models/xgboost.pkl"
}

st.sidebar.header("Upload Test Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV (with target)", type=["csv"])

model_name = st.sidebar.selectbox("Select Model", list(model_map.keys()))

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(df.head())

    X, y = preprocess_data(df)

    model = joblib.load(model_map[model_name])

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1]

    metrics = evaluate(y, y_pred, y_prob)

    st.subheader("Performance Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(metrics["Accuracy"],3))
    col2.metric("AUC", round(metrics["AUC"],3))
    col3.metric("Precision", round(metrics["Precision"],3))

    col1.metric("Recall", round(metrics["Recall"],3))
    col2.metric("F1 Score", round(metrics["F1"],3))
    col3.metric("MCC", round(metrics["MCC"],3))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y, y_pred))

    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))

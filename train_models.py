import os
import joblib
import pandas as pd
from model.preprocess import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

df = pd.read_csv("adult.csv")

X, y = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "logistic": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

os.makedirs("model/trained_models", exist_ok=True)

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)

    joblib.dump(model, f"model/trained_models/{name}.pkl")
    print(f"Saved model/trained_models/{name}.pkl")

print("\nAll models trained and saved successfully!")

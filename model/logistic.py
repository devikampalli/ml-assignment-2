import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from model.preprocess import preprocess_data

df = pd.read_csv("adult.csv")
X, y = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

joblib.dump(model, "model/trained_models/logistic.pkl")
print("Logistic model saved")

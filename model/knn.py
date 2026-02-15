import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from model.preprocess import preprocess_data

df = pd.read_csv("adult.csv")

X, y = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

joblib.dump(model, "model/trained_models/knn.pkl")

print("KNN Model trained & saved successfully")

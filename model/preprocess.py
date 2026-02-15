import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    df = df.copy()

    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    X = df.drop("income", axis=1)
    y = df["income"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

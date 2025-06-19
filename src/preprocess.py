import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(path, return_feature_names=False):
    df = pd.read_csv(path)

    def get_label(rating):
        if rating >= 288:
            return 0  # green
        elif rating >= 208:
            return 1  # yellow
        else:
            return 2  # red

    df["label"] = df["CO2 Emissions(g/km)"].apply(get_label)

    features = [
        "Engine Size(L)",
        "Cylinders",
        "Fuel Consumption City (L/100 km)",
        "Fuel Consumption Hwy (L/100 km)",
        "Fuel Type",
        "Transmission"
    ]

    X = df[features]
    y = df["label"]

    # One-hot encoding
    X = pd.get_dummies(X, columns=["Fuel Type", "Transmission"])

    # Özellik isimlerini kaydet (Streamlit'te input oluşturmak için)
    feature_names = X.columns.tolist()

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y.values, test_size=0.2, random_state=42
    )

    if return_feature_names:
        return feature_names, X_train, X_test, y_train, y_test
    else:
        return X_train, X_test, y_train, y_test

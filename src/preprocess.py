import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(path):
    df = pd.read_csv(path)

    # Etiket sınıflandırması
    def get_label(rating):
        if rating >= 288:
            return 0  # green
        elif rating >= 208:
            return 1  # yellow
        else:
            return 2  # red

    df["label"] = df["CO2 Emissions(g/km)"].apply(get_label)


    # Kullanacağımız sütunlar
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

    # Kategorik sütunları sayısala çevir
    X = pd.get_dummies(X, columns=["Fuel Type", "Transmission"])

    # Normalizasyon
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Eğitim / test ayırma
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y.values, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test

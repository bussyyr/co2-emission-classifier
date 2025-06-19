import numpy as np
from src.predict import predict_class
from src.preprocess import load_and_preprocess_data
import sys
import os

##sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_prediction_shape():
    X, _, _, _ = load_and_preprocess_data("data/CO2 Emissions_Canada.csv")
    input_data = X[:1]  # Özellik sayısı 11
    label, probs = predict_class(input_data)
    assert probs.shape == (1, 3)
    assert 0 <= label[0] <= 2
    assert np.allclose(np.sum(probs), 1.0, atol=1e-5)

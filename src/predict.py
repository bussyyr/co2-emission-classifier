import numpy as np
from src.model import NeuralNetwork
from src.preprocess import load_and_preprocess_data

def predict_class(input_data):
    # Eğitimde kullandığın yapıya göre model mimarisi aynı olmalı
    input_size = input_data.shape[1]
    nn = NeuralNetwork(input_size, 64, 32, 16, 3)
    
    # Ağırlıkları yükle
    weights = np.load("best_model.npz")
    nn.W1 = weights["W1"]
    nn.b1 = weights["b1"]
    nn.W2 = weights["W2"]
    nn.b2 = weights["b2"]
    nn.W3 = weights["W3"]
    nn.b3 = weights["b3"]
    nn.W4 = weights["W4"]
    nn.b4 = weights["b4"]

    probs = nn.forward(input_data)
    label = np.argmax(probs, axis=1)
    return label, probs

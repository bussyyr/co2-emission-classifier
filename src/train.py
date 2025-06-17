import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from src.preprocess import load_and_preprocess_data
from src.model import NeuralNetwork

# 1. Veriyi al
X_train, X_test, y_train, y_test = load_and_preprocess_data("data/CO2 Emissions_Canada.csv")


# 2. Modeli başlat
input_size = X_train.shape[1]
hidden_size1 = 32
hidden_size2 = 16
output_size = 3

# Class weights hesapla
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(enumerate(weights))


nn = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size, learning_rate=0.01, class_weights=class_weights)

# Kaydedilmiş model varsa yükle
import os
if os.path.exists("model_weights.npz"):
    weights = np.load("model_weights.npz")
    nn.W1 = weights["W1"]
    nn.b1 = weights["b1"]
    nn.W2 = weights["W2"]
    nn.b2 = weights["b2"]
    nn.W3 = weights["W3"]
    nn.b3 = weights["b3"]
    print("Kaydedilmiş model yüklendi.")

y_train = y_train.astype(int).flatten()
y_test = y_test.astype(int).flatten()


# 3. Eğitim döngüsü
epochs = 1000
for epoch in range(epochs):
    # İleri besleme
    y_pred = nn.forward(X_train)
    y_pred_copy = y_pred.copy()  # Softmax çıktısını koru

    loss = nn.compute_loss(y_pred_copy, y_train)
    nn.backward(X_train, y_train)

    predictions = np.argmax(y_pred_copy, axis=1)

    y_true = y_train.astype(int)  # Emin olmak için
    acc = np.mean(predictions == y_true)

    if (epoch + 1) % 100 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {acc:.4f}")

# 4. Test doğruluğu
y_test_pred = nn.predict(X_test)
# Confusion matrix hesapla
cm = confusion_matrix(y_test, y_test_pred)

test_acc = np.mean(y_test_pred == y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

print(classification_report(y_test, y_test_pred, target_names=["Low", "Medium", "High"]))


# 5. Modeli kaydet
np.savez("model_weights.npz", W1=nn.W1, b1=nn.b1, W2=nn.W2, b2=nn.b2, W3=nn.W3, b3=nn.b3)
print("Model kaydedildi.")

# Confusion matrix'i görselleştir
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()


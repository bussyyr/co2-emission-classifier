import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from src.preprocess import load_and_preprocess_data
from src.model import NeuralNetwork
import os


# 1. Veriyi al
X, X_test, y, y_test = load_and_preprocess_data("data/CO2 Emissions_Canada.csv")

# Eğitim/Doğrulama ayrımı
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

# 2. Modeli başlat
input_size = X_train.shape[1]
hidden_size1 = 64
hidden_size2 = 32
hidden_size3 = 16
output_size = 3

# Class weights hesapla
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(enumerate(weights))

# Modeli oluştur
nn = NeuralNetwork(input_size, hidden_size1, hidden_size2, hidden_size3, output_size,
                   learning_rate=0.01, class_weights=class_weights)

# Kaydedilmiş model varsa yükle
if os.path.exists("model_weights.npz"):
    weights = np.load("model_weights.npz")
    nn.W1 = weights["W1"]
    nn.b1 = weights["b1"]
    nn.W2 = weights["W2"]
    nn.b2 = weights["b2"]
    nn.W3 = weights["W3"]
    nn.b3 = weights["b3"]
    nn.W4 = weights["W4"]
    nn.b4 = weights["b4"]
    print("Kaydedilmiş model yüklendi.")

# Etiketleri düzleştir
y_train = y_train.astype(int).flatten()
y_val = y_val.astype(int).flatten()
y_test = y_test.astype(int).flatten()

# 3. Eğitim döngüsü (early stopping dahil)
epochs = 3000
best_val_loss = float('inf')
patience = 20
patience_counter = 0
best_weights = {}

val_losses = []  # Validation loss'ları listele

for epoch in range(epochs):
    y_pred = nn.forward(X_train)
    y_pred_copy = y_pred.copy()

    loss = nn.compute_loss(y_pred_copy, y_train)
    nn.backward(X_train, y_train)

    predictions = np.argmax(y_pred_copy, axis=1)
    acc = np.mean(predictions == y_train)

    # Validation loss hesapla
    val_pred = nn.forward(X_val)
    val_loss = nn.compute_loss(val_pred.copy(), y_val)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_weights = {
            'W1': nn.W1.copy(), 'b1': nn.b1.copy(),
            'W2': nn.W2.copy(), 'b2': nn.b2.copy(),
            'W3': nn.W3.copy(), 'b3': nn.b3.copy(),
            'W4': nn.W4.copy(), 'b4': nn.b4.copy()
        }
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping at epoch {epoch+1}")
            break

    if (epoch + 1) % 1000 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss:.4f} - Accuracy: {acc:.4f} - Val Loss: {val_loss:.4f}")

# En iyi ağırlıkları geri yükle
if best_weights:
    nn.W1 = best_weights['W1']
    nn.b1 = best_weights['b1']
    nn.W2 = best_weights['W2']
    nn.b2 = best_weights['b2']
    nn.W3 = best_weights['W3']
    nn.b3 = best_weights['b3']
    nn.W4 = best_weights['W4']
    nn.b4 = best_weights['b4']
    print("⏪ En iyi model yüklendi.")
else:
    print("⚠️ Early stopping tetiklenmedi, model en baştaki ağırlıklarla kaldı.")

# 4. Test doğruluğu
y_test_pred = nn.predict(X_test)
cm = confusion_matrix(y_test, y_test_pred)
test_acc = np.mean(y_test_pred == y_test)

print(f"\nTest Accuracy: {test_acc:.4f}")
print(classification_report(y_test, y_test_pred, target_names=["Low", "Medium", "High"]))

# 5. Modeli kaydet
np.savez("model_weights.npz",
         W1=nn.W1, b1=nn.b1,
         W2=nn.W2, b2=nn.b2,
         W3=nn.W3, b3=nn.b3,
         W4=nn.W4, b4=nn.b4)
print("Model kaydedildi.")

# Confusion matrix'i görselleştir
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# 6. Validation Loss Grafiği
plt.figure(figsize=(8, 5))
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation Loss vs Epochs")
plt.grid(True)
plt.legend()
plt.show()

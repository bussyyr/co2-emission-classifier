import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, learning_rate=0.01, class_weights=None, l2_lambda=0.001):
        self.lr = learning_rate
        self.l2_lambda = l2_lambda
        self.class_weights = class_weights if class_weights is not None else {0: 1.0, 1: 1.0, 2: 1.0}

        # Sabit başlatma (önemli)
        np.random.seed(42)

        # Küçük ağırlıklarla başlat
        self.W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size1))

        self.W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2. / hidden_size1)
        self.b2 = np.zeros((1, hidden_size2))

        self.W3 = np.random.randn(hidden_size2, hidden_size3) * np.sqrt(2. / hidden_size2)
        self.b3 = np.zeros((1, hidden_size3))

        self.W4 = np.random.randn(hidden_size3, output_size) * np.sqrt(2. / hidden_size3)
        self.b4 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stabilite için
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.relu(self.z2)

        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = self.relu(self.z3)

        self.z4 = self.a3 @ self.W4 + self.b4
        self.a4 = self.softmax(self.z4)

        return self.a4

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        eps = 1e-15
        y_pred_safe = np.clip(y_pred, eps, 1 - eps)  # log(0) hatasını önle

        weights = np.array([self.class_weights[label] for label in y_true])
        log_likelihood = -np.log(y_pred_safe[range(m), y_true])
        weighted_loss = log_likelihood * weights
        data_loss = np.sum(weighted_loss) / m

        # L2 regularization
        l2_loss = (
            np.sum(self.W1 ** 2) +
            np.sum(self.W2 ** 2) +
            np.sum(self.W3 ** 2) +
            np.sum(self.W4 ** 2)
        )
        total_loss = data_loss + self.l2_lambda * l2_loss
        return total_loss

    def backward(self, X, y_true):
        m = y_true.shape[0]

        delta4 = self.a4.copy()
        delta4[range(m), y_true] -= 1
        delta4 /= m

        dW4 = self.a3.T @ delta4 + self.l2_lambda * self.W4
        db4 = np.sum(delta4, axis=0, keepdims=True)

        delta3 = (delta4 @ self.W4.T) * self.relu_derivative(self.z3)
        dW3 = self.a2.T @ delta3 + self.l2_lambda * self.W3
        db3 = np.sum(delta3, axis=0, keepdims=True)

        delta2 = (delta3 @ self.W3.T) * self.relu_derivative(self.z2)
        dW2 = self.a1.T @ delta2 + self.l2_lambda * self.W2
        db2 = np.sum(delta2, axis=0, keepdims=True)

        delta1 = (delta2 @ self.W2.T) * self.relu_derivative(self.z1)
        dW1 = X.T @ delta1 + self.l2_lambda * self.W1
        db1 = np.sum(delta1, axis=0, keepdims=True)

        self.W4 -= self.lr * dW4
        self.b4 -= self.lr * db4
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

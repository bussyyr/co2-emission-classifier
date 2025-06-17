import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, learning_rate=0.01, class_weights=None):
        self.lr = learning_rate
        self.class_weights = class_weights if class_weights is not None else {0: 1.0, 1: 1.0, 2: 1.0}

        # Ağırlıklar ve biaslar
        self.W1 = np.random.randn(input_size, hidden_size1)
        self.b1 = np.zeros((1, hidden_size1))

        self.W2 = np.random.randn(hidden_size1, hidden_size2)
        self.b2 = np.zeros((1, hidden_size2))

        self.W3 = np.random.randn(hidden_size2, output_size)
        self.b3 = np.zeros((1, output_size))


    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.relu(self.z2)

        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = self.softmax(self.z3)

        return self.a3

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        weights = np.array([self.class_weights[label] for label in y_true])
        log_likelihood = -np.log(y_pred[range(m), y_true])
        weighted_loss = log_likelihood * weights
        return np.sum(weighted_loss) / m

    def backward(self, X, y_true):
        m = y_true.shape[0]

        delta3 = self.a3
        delta3[range(m), y_true] -= 1
        delta3 /= m

        dW3 = self.a2.T @ delta3
        db3 = np.sum(delta3, axis=0, keepdims=True)

        delta2 = delta3 @ self.W3.T * self.relu_derivative(self.z2)
        dW2 = self.a1.T @ delta2
        db2 = np.sum(delta2, axis=0, keepdims=True)

        delta1 = delta2 @ self.W2.T * self.relu_derivative(self.z1)
        dW1 = X.T @ delta1
        db1 = np.sum(delta1, axis=0, keepdims=True)

        # Ağırlıkları güncelle
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

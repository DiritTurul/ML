import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.n_iter):
            linear = np.dot(X, self.theta) + self.bias
            predictions = self._sigmoid(linear)

            dw = (1/len(y)) * np.dot(X.T, (predictions - y))
            db = (1/len(y)) * np.sum(predictions - y)

            self.theta -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear = np.dot(X, self.theta) + self.bias
        return np.where(self._sigmoid(linear) >= 0.5, 1, 0)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


# Example:
if __name__ == "__main__":
    X = np.array([[2],[4],[6],[8],[10],[12]])
    y = np.array([0,0,0,1,1,1])

    model = LogisticRegression(lr=0.1, n_iter=1000)
    model.fit(X, y)

    X_test = np.array([[3],[5],[9],[11]])
    print("Predictions:", model.predict(X_test))

import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        return Counter(k_nearest_labels).most_common(1)[0][0]


# Example:
if __name__ == "__main__":
    X = np.array([[1,2],[2,3],[3,3],[6,6],[7,7],[8,8]])
    y = np.array([0,0,0,1,1,1])

    model = KNN(k=3)
    model.fit(X, y)

    X_test = np.array([[3,2],[7,7],[5,5]])
    print("Predictions:", model.predict(X_test))

import numpy as np

class MultinomialNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_count = {}
        self.feature_count = {}
        self.total_features = {}

        for c in self.classes:
            X_c = X[y == c]
            self.class_count[c] = X_c.shape[0]
            self.feature_count[c] = np.sum(X_c, axis=0)
            self.total_features[c] = np.sum(self.feature_count[c])

        self.total_samples = len(y)

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.class_count[c] / self.total_samples)
            likelihood = np.sum(np.log((self.feature_count[c] + 1) / 
                                        (self.total_features[c] + len(x))) * x)
            posteriors.append(prior + likelihood)
        return self.classes[np.argmax(posteriors)]


# Example:
if __name__ == "__main__":
    X_train = np.array([[3, 2, 0, 1],
                        [2, 3, 0, 0],
                        [3, 3, 0, 1],
                        [0, 1, 3, 3],
                        [0, 0, 2, 2],
                        [1, 0, 3, 3]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    model = MultinomialNaiveBayes()
    model.fit(X_train, y_train)

    X_test = np.array([[3, 2, 0, 0], [0, 1, 3, 2]])
    print("Predictions:", model.predict(X_test))

import numpy as np

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            self.priors[c] = len(X_c) / len(X)

    def predict(self, X):
        return [self._predict_one(x) for x in X]

    def _predict_one(self, x):
        posteriors = []

        for c in self.classes:
            prior = np.log(self.priors[c])
            likelihood = np.sum(np.log(self._pdf(c, x)))
            posteriors.append(prior + likelihood)

        return self.classes[np.argmax(posteriors)]

    def _pdf(self, c, x):
        mean = self.mean[c]
        var = self.var[c]
        numerator = np.exp(- (x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


# Example:
if __name__ == "__main__":
    X_train = np.array([[1, 20], [2, 21], [1, 22], [2, 20], [3, 22], [5, 5], [6, 6], [7, 7]])
    y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1])

    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)

    X_test = np.array([[1, 21], [6, 5]])
    print("Predictions:", model.predict(X_test))

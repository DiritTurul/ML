import numpy as np
from collections import Counter

# ----- Decision Tree Class -----
class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if len(set(y)) == 1 or depth >= self.max_depth:
            return Counter(y).most_common(1)[0][0]

        best_feat, best_thresh = self._best_split(X, y)
        if best_feat is None:
            return Counter(y).most_common(1)[0][0]

        left_idx = X[:, best_feat] < best_thresh
        right_idx = X[:, best_feat] >= best_thresh

        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return (best_feat, best_thresh, left, right)

    def _best_split(self, X, y):
        best_gini = float('inf')
        best_feat, best_thresh = None, None

        for feat in range(self.n_features):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_y = y[X[:, feat] < thresh]
                right_y = y[X[:, feat] >= thresh]
                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                gini = self._gini_index(left_y, right_y)
                if gini < best_gini:
                    best_gini = gini
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh

    def _gini_index(self, left, right):
        def gini(y):
            count = Counter(y)
            return 1 - sum((n / len(y)) ** 2 for n in count.values())

        total = len(left) + len(right)
        return len(left)/total * gini(left) + len(right)/total * gini(right)

    def predict_one(self, x, node=None):
        if node is None:
            node = self.tree

        if not isinstance(node, tuple):
            return node

        feat, thresh, left, right = node

        if x[feat] < thresh:
            return self.predict_one(x, left)
        else:
            return self.predict_one(x, right)

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])


# ----- Random Forest Class -----
class RandomForest:
    def __init__(self, n_trees=10, max_depth=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            indices = np.random.choice(len(X), len(X), replace=True)
            sample_X, sample_y = X[indices], y[indices]
            tree = DecisionTree(self.max_depth)
            tree.fit(sample_X, sample_y)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        final_preds = []
        for i in range(X.shape[0]):
            votes = Counter(tree_preds[:, i])
            final_preds.append(votes.most_common(1)[0][0])
        return np.array(final_preds)


# ----- Test the Random Forest -----
if __name__ == "__main__":
    # Simple dataset (you can replace this with any dataset)
    X = np.array([
        [2, 3],
        [1, 5],
        [3, 2],
        [6, 7],
        [7, 8],
        [8, 6],
        [5, 5],
        [9, 9]
    ])
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])

    rf = RandomForest(n_trees=5, max_depth=3)
    rf.fit(X, y)

    test_data = np.array([
        [3, 3],
        [7, 7],
        [1, 4],
        [8, 9]
    ])

    predictions = rf.predict(test_data)
    print("Predictions:", predictions)

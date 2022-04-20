import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, split_min=2, maximum_depth=100, num_of_feats=None):
        self.split_min = split_min
        self.maximum_depth = maximum_depth
        self.num_of_feats = num_of_feats
        self.root = None

    def fit(self, X, y):
        self.num_of_feats = X.shape[1] if not self.num_of_feats else min(self.num_of_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        num_of_samples, num_of_features = X.shape
        num_of_labels = len(np.unique(y))

        if (
            depth >= self.maximum_depth
            or num_of_labels == 1
            or num_of_samples < self.split_min
        ):
            leaf_value = self._most_common_label(y)
            return DecisionTreeForRandomForest(value=leaf_value)

        feat_indices = np.random.choice(num_of_features, self.num_of_feats, replace=False)

        best_feat, best_thresh = self._outcome_best(X, y, feat_indices)

        left_indices, right_indices = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_indices, :], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices, :], y[right_indices], depth + 1)
        return DecisionTreeForRandomForest(best_feat, best_thresh, left, right)

    def _outcome_best(self, X, y, feat_indices):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_indices:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._ig(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _ig(self, y, X_column, split_thresh):
        parent_entropy = claculate_entropy(y)

        left_indices, right_indices = self._split(X_column, split_thresh)

        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_indices), len(right_indices)
        e_l, e_r = claculate_entropy(y[left_indices]), claculate_entropy(y[right_indices])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_indices = np.argwhere(X_column <= split_thresh).flatten()
        right_indices = np.argwhere(X_column > split_thresh).flatten()
        return left_indices, right_indices

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
            counter = Counter(y)
            most_common_label = -1
            if len(counter) != 0:
                most_common_label = counter.most_common(1)[0][0]
            return most_common_label

class DecisionTreeForRandomForest:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

def claculate_entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

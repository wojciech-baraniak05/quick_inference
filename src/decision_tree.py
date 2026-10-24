import numpy as np


class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.classes_, y = np.unique(y, return_inverse=True)
        nodes = []
        self._grow(X, np.eye(len(self.classes_))[y], 0, nodes)
        self.children_left, self.children_right, self.feature, self.threshold, self.value = map(np.array, zip(*nodes))
        return self

    def _grow(self, X, Y, depth, nodes):
        node = len(nodes)
        counts = Y.sum(0)
        nodes.append([-1, -1, -2, -2.0, counts / len(Y)])
        if depth == self.max_depth or len(Y) < self.min_samples_split or counts.max() == len(Y):
            return node
        split = self._best_split(X, Y)
        if split is None:
            return node
        f, t = split
        mask = X[:, f] <= t
        left = self._grow(X[mask], Y[mask], depth + 1, nodes)
        right = self._grow(X[~mask], Y[~mask], depth + 1, nodes)
        nodes[node][:4] = left, right, f, t
        return node

    def _best_split(self, X, Y):
        n, m = len(Y), self.min_samples_leaf
        order = np.argsort(X, axis=0)
        Xs = np.take_along_axis(X, order, axis=0)
        L = np.cumsum(Y[order], axis=0)[:-1]
        R = Y.sum(0) - L
        nl = np.arange(1, n)[:, None]
        score = (L**2).sum(2) / nl + (R**2).sum(2) / (n - nl)
        valid = (Xs[1:] > Xs[:-1]) & (nl >= m) & (n - nl >= m)
        score = np.where(valid, score, -np.inf)
        i, f = np.unravel_index(score.argmax(), score.shape)
        if score[i, f] == -np.inf:
            return None
        return f, (Xs[i, f] + Xs[i + 1, f]) / 2

    def apply(self, X):
        X = np.asarray(X, dtype=np.float64)
        node = np.zeros(len(X), dtype=np.intp)
        active = np.arange(len(X))
        while (active := active[self.children_left[node[active]] != -1]).size:
            n = node[active]
            go_left = X[active, self.feature[n]] <= self.threshold[n]
            node[active] = np.where(go_left, self.children_left[n], self.children_right[n])
        return node

    def predict_proba(self, X):
        return self.value[self.apply(X)]

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

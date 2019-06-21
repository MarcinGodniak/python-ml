import numpy as np

class Perceptron:
    def __init__(self, learning_rate=1.0, pocket=False, max_iteration=10000):
        self.leaerning_rate = learning_rate
        self.pocket = pocket
        self.max_iteration = max_iteration
        self.w = None

    def fit(self, X, y):
        current_it = 0
        self.w = np.zeros( X.shape[1] + 1 )
        while current_it <= self.max_iteration:
            if self._step(X, y):
                current_it += 1
            else:
                break

    def _step(self, X, y):
        for i in range(0, X.shape[0]):
            row = X[i, :]
            prediction = self.predict(row)
            if y[i] != prediction:
                self.w[1:] += y[i] * row
                self.w[0] += y[i]
                return True
        else:
            return False

    def predict(self, x):
        if x.dot(self.w[1:]) + self.w[0] >= 0:
            return 1
        else:
            return -1

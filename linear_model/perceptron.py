import numpy as np


class Perceptron(object):
    def __init__(self, learning_rate=1.0, max_iteration=10000):
        self.learning_rate = learning_rate
        self.max_iteration = max_iteration
        self.w = None

    def fit(self, X, y):
        current_it = 0
        self.w = np.zeros( X.shape[1] + 1 )
        while current_it <= self.max_iteration:
            if self._step(X, y):
                current_it += 1
                self._after_step()
            else:
                break

    def _after_step(self):
        pass

    def _step(self, X, y):
        for x, answer in zip(X, y):
            prediction = self.predict(x)
            if answer != prediction:
                self.w[1:] += x * answer * self.learning_rate
                self.w[0] += answer * self.learning_rate
                return True
        else:
            return False


    def predict(self, x):
        if x.dot(self.w[1:]) + self.w[0] >= 0:
            return 1
        else:
            return -1

    def predict_all(self, X):
        y = np.zeros(X.shape[0])
        for i in range(0, X.shape[0]):
            y[i] = self.predict(X[i, :])
        return y

    def get_error(self, X, y):
        error_count = 0
        for x, answer in zip(X, y):
            prediction = self.predict(x)
            if answer != prediction:
                error_count += 1

        return error_count / X.shape[0]


def add_pocket(base):
    class WithPocket(base):
        def __init__(self, *args):
            super().__init__(*args)
            self.best_error = 1.0
            self.best_w = list()
            self.X = None
            self.y = None

        def fit(self, X, y):
            self.X = X
            self.y = y
            super().fit(X, y)
            self.w = self.best_w

        def _after_step(self):
            super()._after_step()
            current_error = super().get_error(self.X, self.y)
            if current_error < self.best_error:
                self.best_w = self.w
                self.best_error = current_error

    return WithPocket


def add_history(base):
    class WithHistory(base):
        def __init__(self, *args):
            super().__init__(*args)
            self.errors = list()
            self.X = None
            self.y = None

        def fit(self, X, y):
            self.X = X
            self.y = y
            super().fit(X, y)
            return self.errors

        def _after_step(self):
            super()._after_step()
            self.errors.append(super().get_error(self.X, self.y))

    return WithHistory


class PerceptronWithHistory(add_history(Perceptron)):
    pass


class PerceptronWithPocket(add_pocket(Perceptron)):
    pass


class PerceptronWithPocketAndHistory(add_history(add_pocket(Perceptron))):
    pass
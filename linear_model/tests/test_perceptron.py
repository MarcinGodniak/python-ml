import pytest

from linear_model.perceptron import Perceptron
from linear_model.perceptron import PerceptronWithHistory
from linear_model.perceptron import PerceptronWithPocket

import numpy as np
import operator


class TestPerceptron(object):
    def test_step_all_fit(self):
        # Given
        p = Perceptron()
        X = np.array([[1, 1],
                      [1, 0],
                      [0, 1],
                      [-1, 0],
                      [0, -1],
                      [-1, -1]])
        y = np.array([1, 1, 1, -1, -1, -1])
        p.w = np.ones(X.shape[1] + 1)
        p.w[0] = 0
        old_w = p.w.copy()

        # When
        assert p._step(X, y) is False

        # Then
        np.testing.assert_array_equal(p.w, old_w)

    def test_step_no_fit(self):
        # Given
        p = Perceptron()
        X = np.array([[1, 1]])
        y = np.array([-1 ])
        p.w = np.ones(X.shape[1] + 1)
        old_w = p.w.copy()

        # When
        assert p._step(X, y)

        # Then
        np.testing.assert_array_compare(operator.__ne__, p.w, old_w)

    def test_fit_1d(self):
        # Given
        p = Perceptron()
        X = np.array([[1],
                      [2]])

        y = np.array([-1, 1])

        # When
        p.fit(X, y)

        # Then
        assert p._step(X, y) is False

    def test_fit_2d(self):
        # Given
        p = Perceptron()
        X = np.array([[1, 1],
                      [2, 2]])

        y = np.array([-1, 1])

        # When
        p.fit(X, y)

        # Then
        assert p._step(X, y) is False

    def test_predict_2d(self):
        # Given
        p = Perceptron()
        X = np.array([[1, 1],
                      [0, 1],
                      [1, 0],
                      [2, 0],
                      [0, 2],
                      [2, 2]])

        y = np.array([-1, -1, -1, 1, 1, 1])

        # When
        p.fit(X, y)

        # Then
        assert p.predict(np.array([3, 3])) == 1


class TestPerceptronWithHistory(object):
    def test_step_all_fit(self):
        # Given
        p = PerceptronWithHistory()
        X = np.array([[1, 1],
                      [1, 0],
                      [0, 1],
                      [-1, 0],
                      [0, -1],
                      [-1, -1]])
        y = np.array([1, 1, 1, -1, -1, -1])

        # When
        errors = p.fit(X, y)

        # Then
        np.testing.assert_array_equal(errors, [1/6, 0])

    def test_fit_2d(self):
        # Given
        p = PerceptronWithHistory()
        X = np.array([[1, 1],
                      [2, 2]])

        y = np.array([-1, 1])

        # When
        errors = p.fit(X, y)

        # Then
        np.testing.assert_array_equal(errors, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0])


class TestPerceptronWithPocket(object):
    def test_step_all_fit(self):
        # Given
        p = PerceptronWithPocket()
        X = np.array([[1, 1],
                      [1, 0],
                      [0, 1],
                      [-1, 0],
                      [0, -1],
                      [-1, -1]])
        y = np.array([1, 1, 1, -1, -1, -1])

        # When
        p.fit(X, y)

        # Then
        assert p._step(X, y) is False
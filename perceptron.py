import numpy as np


class Perceptron:
    def __init__(self, n_iteration, rate):
        self.n_iteration = n_iteration
        self.rate = rate
        self.weights = None
        self.bias = None

    def activate(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, x, y):
        """
        x - np matrix program result
        y - right result (our input)
        """
        samples, features = x.shape  # Get matrix from x
        self.weights = np.zeros(features)  # matrix consider only 0
        self.bias = 0

        for i in range(self.n_iteration):
            output = x @ self.weights + self.bias  # @ - overwrite * for matrix
            predicted = self.activate(output)

            update = self.rate * (y - predicted)
            self.weights += x.T @ update
            self.bias += self.rate * np.sum(update)

    def predicted(self, x):
        output = x @ self.weights + self.bias
        predicted = self.activate(output)
        return predicted


x = np.array([[1, 0], [1, 1], [0, 0]])
y = np.array([0, 1, 0])

per = Perceptron(n_iteration=100, rate=0.1)
per.fit(x, y)

print(per.predicted(np.array([-1, 2])))

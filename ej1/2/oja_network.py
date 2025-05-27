import random
import numpy as np


def constant_learning_rate(t):
    return 1e-3

def decaying_learning_rate(t):
    return 0.5/t

class OjaNetwork:
    def __init__(self, input_size, learning_rate_function, seed = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.weights = np.array([random.uniform(0, 1) for _ in range(input_size)])
        self.learning_rate_function = learning_rate_function

    def weight_update(self, x, output, t):
        #return [w_i + self.learning_rate_function(t) * output * (x_i - w_i * output) for w_i, x_i in zip(self.weights, x)]
        return self.weights + self.learning_rate_function(t) * output * (x - self.weights * output)

    def train(self, training_set, epochs: int):
        for epoch in range(epochs):
            random.shuffle(training_set)
            for x in training_set:
                output = np.dot(self.weights, x)
                #output = sum(w * x_i for w, x_i in zip(self.weights, x))
                self.weights = self.weight_update(x, output, epoch+1)


    def test(self, x):
        return sum(w * x_i for w, x_i in zip(self.weights, x))



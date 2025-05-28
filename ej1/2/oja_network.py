import copy
import random
import numpy as np

def constant_learning_rate(initial_lr, t):
    return initial_lr

def decaying_learning_rate(initial_lr, t):
    return initial_lr/(t+1)

class OjaNetwork:
    def __init__(self, input_size, initial_lr, learning_rate_function, seed = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.weights = np.random.uniform(0, 1, size=input_size)

        self.learning_rate_function = lambda t: learning_rate_function(initial_lr, t)

    def weight_update(self, x, output, t):
        return self.weights + self.learning_rate_function(t) * output * (x - output * self.weights)

    def train(self, training_set, epochs: int):
        training_set = copy.deepcopy(training_set)
        for epoch in range(epochs):
            random.shuffle(training_set)
            for x in training_set:
                output = self.weights @ x
                self.weights = self.weight_update(x, output, epoch)


    def test(self, x):
        return np.dot(self.weights, x)
import random

class OjaNetwork:
        def __init__(self, input_size, learning_rate, seed = None):
            if seed is not None:
                random.seed(seed)

            self.weights = [random.uniform(-1, 1) for _ in range(input_size + 1)]
            self.learning_rate = learning_rate

        def weight_update(self, x, output):
            return [w_i + self.learning_rate*(output * x_i - output**2 * w_i) for w_i, x_i in zip(self.weights, x)]

        def train(self, training_set, epochs: int):

            for epoch in range(epochs):

                random.shuffle(training_set)

                for x in training_set:
                    x_with_bias = x + [1]
                    output = sum(w * x_i for w, x_i in zip(self.weights, x_with_bias))
                    self.weights = self.weight_update(x_with_bias, output)


        def test(self, x):
            x_with_bias = x + [1]
            return sum(w * x_i for w, x_i in zip(self.weights, x_with_bias))

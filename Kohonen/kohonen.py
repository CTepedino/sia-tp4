import numpy as np

class Kohonen:
    def __init__(self, x, y, input_len, learning_rate=0.5, radius=None, decay_function=None):
        self.x = x
        self.y = y  
        self.input_len = input_len  
        self.learning_rate = learning_rate
        self.radius = max(x, y) / 2 if radius is None else radius
        self.weights = np.random.rand(x, y, input_len)
        self.decay_function = decay_function if decay_function else lambda x, t, max_iter: x * np.exp(-t / max_iter)

    def _euclidean_distance(self, a, b):
        return np.linalg.norm(a - b)

    def _get_winner(self, input_vector):
        distances = np.linalg.norm(self.weights - input_vector, axis=2)
        return np.unravel_index(np.argmin(distances), (self.x, self.y))

    def _neighborhood(self, winner, radius):
        d = lambda a, b: np.linalg.norm(np.array(a) - np.array(b))
        return [(i, j) for i in range(self.x) for j in range(self.y) if d((i, j), winner) <= radius]

    def train(self, data, num_iterations=1000):
        for t in range(num_iterations):
            idx = np.random.randint(len(data))
            input_vector = data[idx]
            winner = self._get_winner(input_vector)

            lr = self.decay_function(self.learning_rate, t, num_iterations)
            rad = self.decay_function(self.radius, t, num_iterations)

            for i, j in self._neighborhood(winner, rad):
                delta = lr * (input_vector - self.weights[i, j])
                self.weights[i, j] += delta

    def map_input(self, data):
        return [self._get_winner(x) for x in data]

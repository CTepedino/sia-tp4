import math
import random
from abc import ABC, abstractmethod

import numpy as np

def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def exponential_distance(v1, v2):
    return np.exp(-np.linalg.norm(v1 - v2)**2)


class KohonenNetwork(ABC):
    def __init__(self, k, input_dim, initial_radius, distance_function, initial_learning_rate):
        self.k = k
        self.input_dim = input_dim

        self.initial_radius = initial_radius

        self.distance_function = distance_function
        self.initial_learning_rate = initial_learning_rate

        self.network = np.zeros((k, k, input_dim))

    def init_weights(self, data):
        samples = random.sample(list(data), self.k * self.k)
        samples = np.array(samples)

        self.network = samples.reshape((self.k, self.k, self.input_dim))

    @abstractmethod
    def get_neighbours(self, neuron, radius):
        pass

    def best_neuron(self, input):

        dist = np.empty((self.k, self.k))
        for i in range(self.k):
            for j in range(self.k):
                dist[i, j] = self.distance_function(self.network[i, j], input)

        best_idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
        return best_idx

    def learning_rate_decay(self, t, total_iterations):
        return self.initial_learning_rate * math.exp(-t / total_iterations)

    def radius_decay(self, t, total_iterations):
        if self.initial_radius == 1:
            return 1
        else:
            return self.initial_radius * math.exp(-t / (total_iterations / math.log(self.initial_radius)))

    def train(self, data, iterations, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.init_weights(data)

        for t in range(iterations):
            input_vector = random.choice(data)
            best = self.best_neuron(input_vector)

            lr = self.learning_rate_decay(t, iterations)
            radius = self.radius_decay(t, iterations)

            neighbours = self.get_neighbours(best, radius)

            for n in neighbours:
                self.network[n[0], n[1]] += lr * (input_vector - self.network[n[0], n[1]])

    def map(self, input):
        return self.best_neuron(input)


class QuadraticKohonenNetwork(KohonenNetwork):
    def get_neighbours(self, neuron, radius):
        cx, cy = neuron
        r = int(math.ceil(radius))
        neighbours = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.k and 0 <= ny < self.k:
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist <= radius:
                        neighbours.append((nx, ny))
        return neighbours


class HexagonalKohonenNetwork(KohonenNetwork):
    def get_neighbours(self, neuron, radius):
        cx, cy = neuron
        r = int(math.ceil(radius))
        neighbours = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.k and 0 <= ny < self.k:
                    dz = -dx - dy
                    hex_dist = (abs(dx) + abs(dy) + abs(dz)) / 2
                    if hex_dist <= radius:
                        neighbours.append((nx, ny))
        return neighbours



import math
import random
from copy import deepcopy
from abc import ABC, abstractmethod

def euclidean_distance(v1, v2):
    return math.sqrt(sum((a-b)**2 for a, b in zip(v1, v2)))

def exponential_distance(v1, v2):
    return math.exp(-(euclidean_distance(v1, v2)**2))


class KohonenNetwork(ABC):
    def __init__(self, k, input_dim, initial_radius, distance_function, initial_learning_rate):
        self.k = k
        self.input_dim = input_dim

        if initial_radius < 1:
            raise ValueError("initial_radius must be at least 1")
        self.initial_radius = initial_radius

        self.distance_function = distance_function
        self.initial_learning_rate = initial_learning_rate

        self.network = [[[] for _ in range(k)] for _ in range(k)]

    def init_weights(self, data):
        samples = random.choices(data, k=self.k * self.k)
        idx = 0
        for i in range(self.k):
            for j in range(self.k):
                self.network[i][j] = deepcopy(samples[idx])
                idx += 1

    @abstractmethod
    def get_neighbours(self, neuron, radius):
        pass

    def best_neuron(self, input):
        best_distance = float('inf')
        best = (0, 0)
        for i in range(self.k):
            for j in range(self.k):
                dist = self.distance_function(input, self.network[i][j])
                if dist < best_distance:
                    best_distance = dist
                    best = (i, j)
        return best

    def train(self, data, iterations):
        self.init_weights(data)

        for t in range(iterations):
            input = random.choice(data)
            best_neuron = self.best_neuron(input)

            learning_rate = self.initial_learning_rate * math.exp(-t / iterations)
            radius = self.initial_radius * math.exp(-t / (iterations / math.log(self.initial_radius)))


            neighbours = self.get_neighbours(best_neuron, radius)
            for n in neighbours:
                self.network[n[0]][n[1]] = [w + learning_rate * (x - w) for w, x in zip(self.network[n[0]][n[1]], input)]


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
                    if math.sqrt(dx * dx + dy * dy) <= radius:
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




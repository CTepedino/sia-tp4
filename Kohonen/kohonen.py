import numpy as np

class Kohonen:
    def __init__(self, x, y, input_len, learning_rate=0.5, radius=None, decay_function=None, grid_type="quad"):
        self.x = x
        self.y = y  
        self.input_len = input_len  
        self.learning_rate = learning_rate
        self.radius = max(max(x, y) / 2, 1) if radius is None else max(radius, 1)
        self.weights = np.random.rand(x, y, input_len)
        self.decay_function = decay_function if decay_function else lambda x, t, max_iter: x * np.exp(-t / max_iter)
        self.grid_type = grid_type.lower()

    def _euclidean_distance(self, a, b):
        return np.linalg.norm(a - b)

    def _get_winner(self, input_vector):
        distances = np.linalg.norm(self.weights - input_vector, axis=2)
        return np.unravel_index(np.argmin(distances), (self.x, self.y))

    def _get_hex_neighbors(self, i, j):
        # For even rows
        if i % 2 == 0:
            neighbors = [
                (i-1, j-1), (i-1, j),  # top left, top right
                (i, j-1), (i, j+1),    # left, right
                (i+1, j-1), (i+1, j)   # bottom left, bottom right
            ]
        # For odd rows
        else:
            neighbors = [
                (i-1, j), (i-1, j+1),  # top left, top right
                (i, j-1), (i, j+1),    # left, right
                (i+1, j), (i+1, j+1)   # bottom left, bottom right
            ]
        
        # Filter out invalid positions
        return [(ni, nj) for ni, nj in neighbors 
                if 0 <= ni < self.x and 0 <= nj < self.y]

    def _neighborhood(self, winner, radius):
        if self.grid_type == "hex":
            # For hexagonal grid, we use a different neighborhood calculation
            neighbors = set()
            current = {winner}
            
            # Expand radius times
            for _ in range(int(radius)):
                new_neighbors = set()
                for pos in current:
                    new_neighbors.update(self._get_hex_neighbors(*pos))
                neighbors.update(new_neighbors)
                current = new_neighbors
            
            return list(neighbors)
        else:
            # Original square grid neighborhood
            d = lambda a, b: np.linalg.norm(np.array(a) - np.array(b))
            return [(i, j) for i in range(self.x) for j in range(self.y) 
                   if d((i, j), winner) <= radius]

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

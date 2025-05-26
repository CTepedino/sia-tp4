import numpy as np


def safe_storage(pattern_count, pattern_dimension):
    return pattern_count <= 0.15 * pattern_dimension

def hamming_distance(a, b):
    return np.sum(np.array(a) != np.array(b))

class HopfieldNetwork:
    def __init__(self, stored_patterns):
        self.pattern_count = len(stored_patterns)
        self.pattern_dimension = len(stored_patterns[0])

        if not safe_storage(self.pattern_count, self.pattern_dimension):
            print("Warning: Too many patterns given for the dimension")

        for i in range(self.pattern_count):
            for j in range(i + 1, self.pattern_count):
                d = hamming_distance(stored_patterns[i], stored_patterns[j])
                if d < 0.3 * self.pattern_dimension:
                    print(f"Warning: Patterns {i} and {j} are too similar")

        stored_patterns = np.array(stored_patterns)
        self.weights = (1/self.pattern_dimension) * stored_patterns.transpose() @ stored_patterns
        np.fill_diagonal(self.weights, 0)

    def get_stored(self, pattern, max_epochs = float('inf'), detailed = False):

        state = np.array(pattern)
        epoch = 0
        results = []

        while epoch < max_epochs:

            new_state = np.sign(self.weights @ state)

            #si alguno queda en 0, mantiene el valor del S anterior
            zero_indexes = (new_state == 0)
            new_state[zero_indexes] = state[zero_indexes]

            if detailed:
                results.append({"energy": self.energy(state), "state": state.tolist()})

            if np.array_equal(new_state, state):
                results.append({"energy": self.energy(new_state), "state": new_state.tolist()})
                break

            state = new_state
            epoch += 1

        if detailed:
            return results

        return state

    def energy(self, state):
        return -0.5 * (state @ self.weights @ state)



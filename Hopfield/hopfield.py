import numpy as np


class HopfieldNet:
    def __init__(self, size):
        self.size = size
        self.W = np.zeros((size, size))

    #calcula solo el triangulo superior y luego replica
    def train(self, patterns):
        for p in patterns:
            v = p.flatten()
            upper_tri = np.triu(np.outer(v, v), k=1)
            self.W += upper_tri
            self.W += upper_tri.T
        self.W /= len(patterns)

    def recall(self, pattern, steps=10, verbose=True):
        v = pattern.flatten()
        for step in range(steps):
            v_new = np.sign(self.W @ v)
            v_new[v_new == 0] = 1
            if verbose:
                print(f"Paso {step+1}:")
                print_pattern(v_new.reshape((5,5)))
            if np.array_equal(v, v_new):
                break
            v = v_new
        return v.reshape((5,5))
    
def print_pattern(pattern):
    for row in pattern:
        print(' '.join(['*' if x == 1 else ' ' for x in row]))
print()
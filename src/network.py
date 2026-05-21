import numpy as np

class Network:
    def __init__(self, layers: np.ndarray):
        if type(layers) != np.ndarray: 
            raise TypeError("layers expected <np.ndarray>" 
                        f", got {type(layers)}")
    
        self.layers = layers
        self.size = len(layers)

        self.weights = [np.random.rand(layers[i+1], layers[i]) for i in range(self.size - 1)]
        self.biases = [np.random.rand(layers[i]) for i in range(1, self.size)]

import numpy as np

from .utils import cost, sigmoid

class Network:
    def __init__(self, layers: np.ndarray):
        """Creates a new network, where layer i has layers[i] nodes"""

        if type(layers) != np.ndarray: 
            raise TypeError("layers expected <np.ndarray>" 
                        f", got {type(layers)}")
    
        self.layers = layers
        self.size = len(layers)
        
        self.weights = [np.random.rand(layers[i+1], layers[i]) for i in range(self.size - 1)]
        self.biases = [np.random.rand(layers[i]) for i in range(1, self.size)]

    def feedforward(self, X: np.ndarray):
        """Feeds X into network, returns corresponding Y"""

        if type(X) != np.ndarray: 
            raise TypeError("X expected <np.ndarray>" 
                        f", got {type(X)}")
        
        # check that the shape of X matches
        # the number of nodes in the input layer
        if np.shape(X) != (self.layers[0],):
            raise ValueError("Expected X to have shape"
                             f" ({self.layers[0]}, ), but "
                             f"got shape {np.shape(X)}")
        
        current_layer = X

        for i in range(self.size - 1):
            # multiply input vector by weight matrix
            current_layer = self.weights[i] @ current_layer

            # add bias vector to weighted sum
            current_layer += self.biases[i]

            # call activation function
            current_layer = sigmoid(current_layer)
        
        return current_layer



        

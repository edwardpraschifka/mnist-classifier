import pytest
import numpy as np

from src.network import Network

class TestConstructor:
    def test_network_dims(self):
        """Create a network and check the dimensions of its weight and bias arrays"""

        layers = np.array([3,4,2])
        nw = Network(layers)

        assert nw.size == 3

        assert(np.array_equal(layers, nw.layers))

        assert len(nw.weights) == 2
        assert nw.weights[0].shape == (4,3)
        assert nw.weights[1].shape == (2,4)
        
        assert len(nw.biases) == 2
        assert nw.biases[0].shape == (4,)
        assert nw.biases[1].shape == (2,)

    def test_bad_type(self):
        """Create a network using a layers array of invalid type"""

        with pytest.raises(TypeError):
            layers = list([])
            nw = Network(layers)


class TestFeedForward:

    def test_bad_type(self):
        """Try feedforward using an input array of invalid type"""

        layers = np.array([3,4,2])
        nw = Network(layers)
        X = [1,2,3]

        with pytest.raises(TypeError):
            nw.feedforward(X)

    def test_diff_len(self):
        """Try feedforward using an input array of invalid length"""

        layers = np.array([3,4,2])
        nw = Network(layers)
        X = np.array([1,2,3,4])

        with pytest.raises(ValueError):
            nw.feedforward(X)
    
    def test_output(self):
        """Try feedforward and check accuracy of the result"""

        layers = np.array([3,4,2])
        nw = Network(layers)
        X = np.array([0.1, 0.1, 0.2]).reshape((3,))

        nw.weights[0] = np.array([[0.1, 0.2, 0.3], [0.5, 0.4, 0.2], 
                               [0.1, 0.1, 0.2], [0.3, 0.2 , 0.1]])
        nw.weights[1] = np.array([[0.2, 0.1, 0.2, 0.2], [0.3, 0.1, 0.3, 0.1]])

        
        nw.biases[0] = np.array([0.1, 0.2, -0.1, 0.3])
        nw.biases[1] = np.array([0.1, 0.2])

        out = nw.feedforward(X)
        assert np.array_equal(np.round(out,4), np.array([0.6187, 0.6522]))
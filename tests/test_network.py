import pytest
import numpy as np

from src.network import Network

class TestConstructor:
    def test_network_dims(self):
        """Create a 3-layer network with layers of size 3,4,2"""

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
        with pytest.raises(TypeError):
            layers = list([])
            nw = Network(layers)
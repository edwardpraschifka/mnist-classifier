import pytest
import numpy as np

from src.utils import cost, sigmoid

class TestCostFunction:
    def test_output_pos(self):
        y1 = np.array([1,2,3])
        y2 = np.array([4,5,6])

        out = cost(y1,y2)
        assert out == 27

    def test_output_neg(self):
        y1 = np.array([-1,2,-3])
        y2 = np.array([4,-5,6])

        out = cost(y1,y2)
        assert out == 155

    def test_bad_type(self):
        y1 = list([])
        y2 = np.array([])

        with pytest.raises(TypeError):
            out = cost(y1,y2)
        

    def test_diff_len(self):
        y1 = np.array([1,2,3])
        y2 = np.array([1,2])

        with pytest.raises(ValueError):
            out = cost(y1,y2)


class TestSigmoid:
    def test_output_int(self):
        z = 2
        out = sigmoid(z)

        assert round(out,2) == 0.88
    
    def test_output_vec(self):
        z = np.array([1,2,3])
        out = sigmoid(z)
        rounded_out = np.round(out,2)

        expected_out = np.array([0.73, 0.88, 0.95])

        assert np.array_equal(rounded_out, expected_out)

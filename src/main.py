import numpy as np

from src.network import Network
from src.utils import get_mnist_data, accuracy

def test_mnist():
    """Test model performance on MNIST"""

    nw = Network([])
    nw.load("network.pkl")
    
    (_, _, X_test, Y_test) = get_mnist_data()

    feedforward_results = [nw.feedforward(row.reshape(-1, 1)) for row in X_test]
    predictions = np.array([A[-1].flatten() for (Z, A) in feedforward_results])
    acc = accuracy(predictions.T, Y_test.T)

    print(f"Accuracy = {acc*100}%")
    

if __name__ == "__main__":
    test_mnist()
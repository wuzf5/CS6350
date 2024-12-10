import numpy as np
import pandas as pd
from numpy import ndarray


class NeuralNetwork:
    def __init__(self, input_size: int, hidden_dims: tuple[int], output_size: int):
        self.weights = []
        self.biases = []

        prev_layer_size = input_size
        for hidden_size in hidden_dims:
            # Initializing the weights and biases
            W = np.random.randn(hidden_size, prev_layer_size)
            b = np.zeros((hidden_size, 1))

            self.weights.append(W)
            self.biases.append(b)

            prev_layer_size = hidden_size

        # Output layer
        W_out = np.random.randn(output_size, prev_layer_size)
        b_out = np.zeros((output_size, 1))
        
        self.weights.append(W_out)
        self.biases.append(b_out)
    
    def sigmoid(self, x: ndarray):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x: ndarray):
        return x * (1 - x)
    
    def forward(self, X: ndarray):
        # Neuron values
        Zs = [X]
        
        # Propagate through hidden layers
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            Z = self.sigmoid(np.dot(W, Zs[-1]) + b)
            Zs.append(Z)

        # Output layer
        Z_out = np.dot(self.weights[-1], Zs[-1]) + self.biases[-1]
        Zs.append(Z_out)
        
        return Zs

    def backpropagation(self, x: ndarray, y: ndarray):
        # B = X.shape[0]

        Zs = self.forward(x.T)

        dW_grads = {}
        db_grads = {}

        # Compute output layer gradients
        dZ_current = Zs[-1] - y

        # Propagate backwards through layers
        for layer in range(len(self.biases) - 1, -1, -1):
            # Compute weight and bias gradients
            dW_grads['Layer {}:'.format(layer+1)] = np.dot(dZ_current, Zs[layer].T)
            db_grads['Layer {}:'.format(layer+1)] = np.sum(dZ_current, axis=1, keepdims=True)

            # Stop if we've reached the input layer
            if layer == 0:
                break

            # Compute gradient for neurons in the current layer
            dZ_current = np.dot(self.weights[layer].T, dZ_current) * \
                         self.sigmoid_derivative(Zs[layer])

        return {
            'dW_grads': dW_grads,
            'db_grads': db_grads
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dims", default=(2, 2,))
    args = parser.parse_args()

    Xy = pd.read_csv('../../Datasets/bank-note/train.csv', header=None).to_numpy()
    # Xy = Xy.replace({Xy.columns[-1]: 0}, -1).to_numpy()
    X, Y = Xy[:, :-1], Xy[:, -1]
    model = NeuralNetwork(X.shape[1], args.hidden_dims, 1)
    x, y = X[0], Y[0]
    x = x[None, :]
    grads = model.backpropagation(x, y)
    print(grads)

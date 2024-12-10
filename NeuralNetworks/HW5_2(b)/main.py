import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import ndarray


class NeuralNetwork:
    def __init__(self, input_size: int, hidden_dims: tuple[int], output_size: int,
                        gamma_0: float, d: float, n_epochs: int):
        self.gamma_0 = gamma_0
        self.d = d
        self.update_step = 0
        self.n_epochs = n_epochs
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
        self.weights.append(np.random.randn(output_size, prev_layer_size))
        self.biases.append(np.zeros((output_size, 1)))

    def gamma(self, t):
        return self.gamma_0 / (1 + (self.gamma_0 / self.d) * t)
    
    def sigmoid(self, x: ndarray):
        # a more numerically stable version of sigmoid than np.exp(x) / (1 + np.exp(x))
        return np.exp(np.fmin(x, 0)) / (1 + np.exp(-np.abs(x)))
    
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
        Zs = self.forward(x.T)

        W_grads = {}
        b_grads = {}

        # Compute output layer gradients
        dZ_current = Zs[-1] - y

        # Propagate backwards through layers
        for layer in range(len(self.biases) - 1, -1, -1):
            # Compute weight and bias gradients
            W_grads['Layer {}:'.format(layer+1)] = np.dot(dZ_current, Zs[layer].T)
            b_grads['Layer {}:'.format(layer+1)] = np.sum(dZ_current, axis=1, keepdims=True)

            # Stop if we've reached the input layer
            if layer == 0:
                break

            # Compute gradient for neurons in the current layer
            dZ_current = np.dot(self.weights[layer].T, dZ_current) * \
                         self.sigmoid_derivative(Zs[layer])

        return {
            'W_grads': W_grads,
            'b_grads': b_grads
        }
    
    def _train_one_step(self, x: ndarray, y: ndarray):
        # Compute gradients
        grads = self.backpropagation(x, y)

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.gamma(self.update_step) * grads['W_grads']['Layer {}:'.format(i+1)]
            self.biases[i] -= self.gamma(self.update_step) * grads['b_grads']['Layer {}:'.format(i+1)]
        self.update_step += 1

    def _train_one_epoch(self, X: ndarray, Y: ndarray):
        for x, y in zip(X, Y):
            self._train_one_step(x[None, :], y)

    def train(self, Xy: ndarray):
        B = Xy.shape[0]
        acc = []
        for _ in range(self.n_epochs):
            shuffled_idx = np.random.choice(B, size=B, replace=False)
            D = Xy[shuffled_idx]
            X, Y = D[:, :-1], D[:, -1]
            self._train_one_epoch(X, Y)

            # Get training accuracy after each epoch
            predictions = self.predict(X)
            acc.append((predictions == Y).mean())
        # plt.figure()
        # plt.plot(np.arange(len(acc)) + 1, acc, linewidth=1.5)
        # plt.savefig('g{}_d{}.png'.format(self.gamma_0, self.d))
        # print('gamma_0: {}, d: {}, training error: {}'.format(self.gamma_0, self.d, 1 - acc[-1]))
        return 1 - acc[-1]

    def predict(self, X: ndarray):
        """
        Make predictions
        
        Parameters:
        X: Input features (batch_size x feature_size)
        
        Returns:
        Predicted class labels (batch_size x 1)
        """
        predictions = self.forward(X.T)[-1]
        predictions = (predictions > 0.5).astype(int).T
        return predictions.squeeze(-1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--gamma_0", type=float, default=1.)
    parser.add_argument("--d", type=float, default=1e4)
    parser.add_argument("--n_epochs", type=int, default=100)
    args = parser.parse_args()

    # for g in [0.01, 0.001]:
    #     for d in [10000000, 1000, 100, 10]:
    #         args.gamma_0 = g
    #         args.d = d
    width = args.width
    if width == 5:
        args.gamma_0, args.d = 1, 1e5
    elif width == 10:
        args.gamma_0, args.d = 1, 1e4
    elif width == 25:
        args.gamma_0, args.d = 0.1, 1e3
    elif width == 50:
        args.gamma_0, args.d = 0.01, 1e3
    elif width == 100:
        args.gamma_0, args.d = 0.01, 1e6
    else:
        raise ValueError('need to re-tune gamma_0 and d!')

    print('Computing training error and test error averaged over 10 runs....')
    training_errors, test_errors = [], []
    for _ in range(10):
        Xy = pd.read_csv('../../Datasets/bank-note/train.csv', header=None).to_numpy()
        model = NeuralNetwork(Xy.shape[1] - 1, (args.width,) * 2, 1, args.gamma_0, args.d, args.n_epochs)
        training_error = model.train(Xy)
        training_errors.append(training_error)

        Xy = pd.read_csv('../../Datasets/bank-note/test.csv', header=None).to_numpy()
        X, y = Xy[:, :-1], Xy[:, -1]
        predictions = model.predict(X)
        test_error = 1 - (predictions == y).mean()
        test_errors.append(test_error)
        # print(g, d, 'Average test error: {}'.format(test_error))
    print('width: {}, training error: {}, test error: {}'.format(width, 
                                                                np.array(training_errors).mean(),
                                                                np.array(test_errors).mean()))

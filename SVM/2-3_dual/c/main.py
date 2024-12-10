import numpy as np
import pandas as pd
import scipy.optimize as optimize

class GaussianKernelDualSVM:
    def __init__(self, args):
        self.C = args.C
        self.gamma = args.gamma
        self.alphas = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.b = None
        
    def kernel(self, x1, x2):
        return np.exp(-(np.linalg.norm(x1 - x2, 2))**2 / self.gamma)
    
    def objective(self, alphas, X, y):
        return 0.5 * np.sum(
            (alphas[:, None] * y[:, None]) * (alphas[None, :] * y[None, :]) * 
            np.array([self.kernel(x1, x2) for x1 in X for x2 in X]).reshape(X.shape[0], X.shape[0])
        ) - np.sum(alphas)

    def constraint_eq(self, alphas, y):
        return (alphas * y).sum()
    
    def train(self, X, y):
        B = X.shape[0]
        
        # Initialization
        alphas_0 = np.zeros(B)

        # Constraint
        constraints = [
            {'type': 'eq', 'fun': self.constraint_eq, 'args': (y,)}
        ]
        
        # Optimization
        result = optimize.minimize(
            self.objective, 
            alphas_0, 
            args=(X, y),
            method='SLSQP',
            bounds=[(0, self.C) for _ in range(B)],
            constraints=constraints
        )
        if result.success:
            print('The optimization process has successfully exited')
        else:
            print('The optimization process didn\'t successfully exit due to {}'.format(result.message))
            raise RuntimeError
        
        # Store optimimal alphas
        self.alphas = result.x
        
        # Identify support vectors
        sv_idx = self.alphas > 1e-5
        self.support_vectors = X[sv_idx]
        self.support_vector_labels = y[sv_idx]
        
        # Recover weights and bias
        w = (self.alphas[:, None] * y[:, None] * X).sum(0) # (d,)
        self.b = np.mean([
            y_i - np.sum(self.alphas[sv_idx] * self.support_vector_labels * 
                         np.array([self.kernel(sv, x_i) for sv in self.support_vectors]))
            for x_i, y_i in zip(X[sv_idx], y[sv_idx])
        ])
        return w, self.b, self.support_vectors
    
    def predict(self, X):
        return np.sign([
            np.sum(self.alphas[self.alphas > 1e-5] * 
                   self.support_vector_labels * 
                   np.array([self.kernel(sv, x) for sv in self.support_vectors])) + self.b
            for x in X
        ])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.1, help='[0.1, 0.5, 1, 5, 100]')
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--C", type=float, default=100/873, help='[100/873, 500/873, 700/873], or [0.115, 0.573, 0.802]')
    args = parser.parse_args()

    Xy = pd.read_csv('../../../Datasets/bank-note/train.csv', header=None)
    Xy = Xy.replace({Xy.columns[-1]: 0}, -1).to_numpy()
    model = GaussianKernelDualSVM(args)
    X, y = Xy[:, :-1], Xy[:, -1]
    w, b, svs = model.train(X, y)
    print('The learned w is: {}'.format(w))
    print('The learned b is: {}'.format(b))
    # print('The support vectors are {}'.format(svs))
    np.save('svs_C{}_gamma{}'.format(args.C, args.gamma), svs)
    np.save('w_C{}_gamma{}'.format(args.C, args.gamma), w)
    np.save('b_C{}_gamma{}'.format(args.C, args.gamma), b)
    predictions = model.predict(X)
    avg_err = 1 - (predictions == y).mean()
    print('Training error: {}'.format(avg_err))

    Xy = pd.read_csv('../../../Datasets/bank-note/test.csv', header=None)
    Xy = Xy.replace({Xy.columns[-1]: 0}, -1).to_numpy()
    X, y = Xy[:, :-1], Xy[:, -1]
    predictions = model.predict(X)
    avg_err = 1 - (predictions == y).mean()
    print('Test error: {}'.format(avg_err))

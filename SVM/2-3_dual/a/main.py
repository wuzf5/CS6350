import numpy as np
import pandas as pd
import scipy.optimize as optimize


class DualSVM:
    def __init__(self, args):
        self.C = args.C
        self.alphas = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.b = None
    
    def objective(self, alphas, X, y):
        return (1/2) * np.sum(
            (alphas[:, None] * y[:, None]) * (alphas[None, :] * y[None, :]) * 
            np.array([(x1 * x2).sum() for x1 in X for x2 in X]).reshape(X.shape[0], X.shape[0])
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
                         np.array([(sv * x_i).sum() for sv in self.support_vectors]))
            for x_i, y_i in zip(X[sv_idx], y[sv_idx])
        ])
        return w, self.b
    
    def predict(self, X):
        return np.sign([
            np.sum(self.alphas[self.alphas > 1e-8] * 
                   self.support_vector_labels * 
                   np.array([(sv * x).sum() for sv in self.support_vectors])) + self.b
            for x in X
        ])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--C", type=float, default=100/873, help='[100/873, 500/873, 700/873]')
    args = parser.parse_args()

    Xy = pd.read_csv('../../../Datasets/bank-note/train.csv', header=None)
    Xy = Xy.replace({Xy.columns[-1]: 0}, -1).to_numpy()
    model = DualSVM(args)
    X, y = Xy[:, :-1], Xy[:, -1]
    w, b = model.train(X, y)
    print('The learned w is: {}'.format(w))
    print('The learned b is: {}'.format(b))
    predictions = model.predict(X)
    avg_err = 1 - (predictions == y).mean()
    print('Training error: {}'.format(avg_err))

    Xy = pd.read_csv('../../../Datasets/bank-note/test.csv', header=None)
    Xy = Xy.replace({Xy.columns[-1]: 0}, -1).to_numpy()
    X, y = Xy[:, :-1], Xy[:, -1]
    # prediction = model.predict_one(np.array([4.5459,8.1674,-2.4586,-1.4621]))
    # print(prediction)
    predictions = model.predict(X)
    avg_err = 1 - (predictions == y).mean()
    print('Test error: {}'.format(avg_err))

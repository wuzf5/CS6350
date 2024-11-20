import pandas as pd
import numpy as np


class StandardPerceptron:
    def __init__(self, args, n, B):
        self.n = n
        self.w = np.zeros(n)
        self.T = args.T
        self.B = B
        self.gamma_0 = args.gamma_0
        self.a = args.a
        self.C = args.C

    def gamma_t(self, t):
        return self.gamma_0 / (1 + self.gamma_0 * t / self.a)

    def train(self, Xy):
        assert self.B == Xy.shape[0]
        assert self.n == Xy.shape[1]
        update_step = 0
        for t in range(self.T):
            shuffled_idx = np.random.choice(self.B, size=self.B, replace=False)
            Xy = Xy[shuffled_idx]
            X, y = Xy[:, :-1], Xy[:, -1]
            X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=-1)
            for i, x in enumerate(X):
                if y[i] * (self.w * x).sum() <= 1:
                    sub_gradient = self._get_sub_gradient(x, y[i])
                    self.w -= self.gamma_t(update_step) * y[i] * x
            update_step += 1
        return self.w
    
    def _get_sub_gradient(self, xi, yi):
        if 1 - yi * (self.w * xi).sum() <= 0:
            w = self.w.copy()
            w[-1] = 0
            return w
        else:
            
    
    def predict_one(self, x):
        return np.sign((self.w * x).sum())

    def predict(self, X):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=-1)
        return np.sign((self.w[None, :] * X).sum(-1)) # (B,)
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma_0", type=float, default=0.01)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--a", type=float, default=1)
    parser.add_argument("--C", type=int, default=1, help='[100/873, 500/873, 700/873]')
    args = parser.parse_args()

    Xy = pd.read_csv('../../Datasets/bank-note/train.csv', header=None)
    Xy = Xy.replace({Xy.columns[-1]: 0}, -1).to_numpy()
    B, n = Xy.shape
    model = StandardPerceptron(args, n, B)
    w = model.train(Xy)
    print('The learned w is: {}'.format(w))

    Xy = pd.read_csv('../../Datasets/bank-note/test.csv', header=None)
    Xy = Xy.replace({Xy.columns[-1]: 0}, -1).to_numpy()
    X, y = Xy[:, :-1], Xy[:, -1]
    # prediction = model.predict_one(np.array([4.5459,8.1674,-2.4586,-1.4621]))
    # print(prediction)
    predictions = model.predict(X)
    avg_err = 1 - (predictions == y).mean()
    print('Average test error: {}'.format(avg_err))

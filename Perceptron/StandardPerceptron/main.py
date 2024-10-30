import pandas as pd
import numpy as np


class StandardPerceptron:
    def __init__(self, args, n, B):
        self.n = n
        self.w = np.zeros(n)
        self.T = args.T
        self.B = B
        self.r = args.r

    def train(self, Xy):
        assert self.B == Xy.shape[0]
        assert self.n == Xy.shape[1] - 1
        for t in range(self.T):
            shuffled_idx = np.random.choice(self.B, size=self.B, replace=False)
            Xy = Xy[shuffled_idx]
            X, y = Xy[:, :-1], Xy[:, -1]
            for i, x in enumerate(X):
                if y[i] * (self.w * x).sum() <= 0:
                    self.w += self.r * y[i] * x
        return self.w
    
    def predict_one(self, x):
        return np.sign((self.w * x).sum())

    def predict(self, X):
        return np.sign((self.w[None, :] * X).sum(-1)) # (B,)
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--T", type=int, default=10)
    args = parser.parse_args()

    Xy = pd.read_csv('../../Datasets/bank-note/train.csv', header=None)
    Xy = Xy.replace({Xy.columns[-1]: 0}, -1).to_numpy()
    B, n = Xy.shape
    model = StandardPerceptron(args, n - 1, B)
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

import pandas as pd
import numpy as np


class AveragedPerceptron:
    def __init__(self, args, n, B):
        self.n = n
        self.w = np.zeros(n)
        self.T = args.T
        self.B = B
        self.r = args.r

    def train(self, Xy):
        assert self.B == Xy.shape[0]
        assert self.n == Xy.shape[1] - 1
        X, y = Xy[:, :-1], Xy[:, -1]
        a = np.zeros_like(self.w)
        for t in range(self.T):
            for i, x in enumerate(X):
                if y[i] * (self.w * x).sum() <= 0:
                    self.w += self.r * y[i] * x
                a += self.w
        return a
    
    # def predict_one(self, x):
    #     return np.sign(np.sum([c * np.sign((w*x).sum()) for c, w in zip(self.votes, self.weights)]))

    def predict(self, X):
        return np.sign((a[None, :] * X).sum(-1))
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--T", type=int, default=10)
    args = parser.parse_args()

    Xy = pd.read_csv('../../Datasets/bank-note/train.csv', header=None)
    Xy = Xy.replace({Xy.columns[-1]: 0}, -1).to_numpy()
    B, n = Xy.shape
    model = AveragedPerceptron(args, n - 1, B)
    a = model.train(Xy)
    print('The weight vector a is: {}'.format(a))

    Xy = pd.read_csv('../../Datasets/bank-note/test.csv', header=None)
    Xy = Xy.replace({Xy.columns[-1]: 0}, -1).to_numpy()
    X, y = Xy[:, :-1], Xy[:, -1]
    # prediction = model.predict_one(np.array([-1.8356,-6.7562,5.0585,-0.55044]))
    # print(prediction)
    predictions = model.predict(X)
    avg_err = 1 - (predictions == y).mean()
    print('Average test error: {}'.format(avg_err))

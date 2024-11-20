import pandas as pd
import numpy as np


class VotedPerceptron:
    def __init__(self, args, n, B):
        self.n = n
        self.w = np.zeros(n)
        self.T = args.T
        self.B = B
        self.r = args.r
        self.weights = []
        self.votes = []

    def train(self, Xy):
        assert self.B == Xy.shape[0]
        assert self.n == Xy.shape[1]
        X, y = Xy[:, :-1], Xy[:, -1]
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=-1)
        for t in range(self.T):
            # shuffled_idx = np.random.choice(self.B, size=self.B, replace=False)
            # Xy = Xy[shuffled_idx]
            correct_count = 1
            for i, x in enumerate(X):
                if y[i] * (self.w * x).sum() <= 0:
                    self.votes.append(correct_count)
                    self.weights.append(self.w.copy())
                    self.w += self.r * y[i] * x
                    correct_count = 1
                else:
                    correct_count += 1
        return self.weights, self.votes
    
    # def predict_one(self, x):
    #     return np.sign(np.sum([c * np.sign((w*x).sum()) for c, w in zip(self.votes, self.weights)]))

    def predict(self, X):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=-1)
        return np.sign(np.sum([c * np.sign((w[None, :]*X).sum(-1)) for c, w in zip(self.votes, self.weights)], axis=0))
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--T", type=int, default=10)
    args = parser.parse_args()

    Xy = pd.read_csv('../../Datasets/bank-note/train.csv', header=None)
    Xy = Xy.replace({Xy.columns[-1]: 0}, -1).to_numpy()
    B, n = Xy.shape
    model = VotedPerceptron(args, n, B)
    weights, votes = model.train(Xy)
    # print('The weights are: {}, the votes are: {}'.format(weights, votes))
    # print(np.sum(weights, axis=0))
    np.save('weights', weights)
    np.save('votes', votes)

    Xy = pd.read_csv('../../Datasets/bank-note/test.csv', header=None)
    Xy = Xy.replace({Xy.columns[-1]: 0}, -1).to_numpy()
    X, y = Xy[:, :-1], Xy[:, -1]
    # prediction = model.predict_one(np.array([-1.8356,-6.7562,5.0585,-0.55044]))
    # print(prediction)
    predictions = model.predict(X)
    avg_err = 1 - (predictions == y).mean()
    print('Average test error: {}'.format(avg_err))

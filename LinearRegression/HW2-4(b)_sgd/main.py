import numpy as np
import pandas as pd


class SGD:
    def __init__(self, args):
        self.args = args
        self._epsilon = args.epsilon
        self._max_grad_steps = args.max_grad_steps
        self._lr = args.lr

    def train(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        losses = []
        update_step = 0

        while True:
            update_step += 1

            # random sample
            idx = np.random.choice(np.arange(m))
            xi, yi = X[idx], y[idx]

            # update weights
            y_pred = np.dot(xi, self.w)
            gradient = xi * (y_pred - yi)
            w_new = self.w - self._lr * gradient

            # compute loss
            loss = (1/2) * ((np.dot(X, self.w) - y)**2).sum()
            losses.append(loss)
            print('iteration: {}, loss: {}'.format(update_step, loss))
            
            if (update_step > 1 and np.abs(losses[-1] - losses[-2]) < self._epsilon) or update_step > self._max_grad_steps:
                break
            
            self.w = w_new

        if update_step > self._max_grad_steps:
            print('Didn\'t converge in {} gradient steps.'.format(self._max_grad_steps))
        
        return self.w, losses
    
    def predict(self, X):
        y_pred = np.dot(X, self.w)
        return y_pred


if __name__ == '__main__':
    import argparse
    from matplotlib import pyplot as plt


    parser = argparse.ArgumentParser()
    parser.add_argument("--attribute_list", default=['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                                                    'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'])
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--max_grad_steps", type=int, default=int(200000))
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()


    traindata = pd.read_csv('../../Datasets/concrete/train.csv', header=None)
    train_features = traindata.drop(columns=[traindata.columns[-1]]).to_numpy() # (B, F)
    train_labels = traindata[traindata.columns[-1]].to_numpy() # (B,)

    train_features = np.hstack((train_features, np.ones((train_features.shape[0], 1))))

    model = SGD(args)
    final_w, losses = model.train(train_features, train_labels)


    print('learned weight vector: {}'.format(final_w))
    plt.plot(np.arange(len(losses))+1, losses, linewidth=1.5)
    plt.savefig('training_curve')

    # from numpy.linalg import inv
    # X, y = train_features, train_labels
    # # compute the analytical solution
    # w_opt = inv(X.T @ X) @ X.T @ y
    # model.w = w_opt


    testdata = pd.read_csv('../../Datasets/concrete/test.csv', header=None)
    test_features = testdata.drop(columns=[testdata.columns[-1]]).to_numpy()
    test_labels = testdata[testdata.columns[-1]].to_numpy()
    test_features = np.hstack((test_features, np.ones((test_features.shape[0], 1))))

    predictions_on_testdata = model.predict(test_features)
    test_loss = (1/2) * ((predictions_on_testdata - test_labels)**2).sum()
    print('loss on test dataset: {}'.format(test_loss))

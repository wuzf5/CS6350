import numpy as np
import pandas as pd
from numpy.linalg import inv


if __name__ == '__main__':

    traindata = pd.read_csv('../../Datasets/concrete/train.csv', header=None)
    X = traindata.drop(columns=[traindata.columns[-1]]).to_numpy() # (B, F)
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    y = traindata[traindata.columns[-1]].to_numpy()

    w_opt = inv(X.T @ X) @ X.T @ y
    print('optimal weight vector: {}'.format(w_opt))
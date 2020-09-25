import numpy as np

def relu(X):
    return np.maximum(0, X)

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def cross_entropy(y_, y):
    print('y is {} y'shape is {}'.format(y, y.shape))
    m = y.shape[1]
    cost = 1 / m * np.nansum(np.multiply(-np.log(y_), y) + \
    np.multiply(-np.log(1 - y_), 1 - y))
    return np.squeeze(cost)

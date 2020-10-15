import numpy as np


def get_mse(y, y_):
    sum = 0
    for i in range(len(y)):
        sum += np.square(y[i] - y_[i])
    return np.sqrt(sum//len(y))
import numpy as np
from lab1.part1.util.logger_util import logger

def get_mse(y, y_):
    sum = 0
    for i in range(len(y)):
        sum += np.square(y[i] - y_[i])
    return np.sqrt(sum / len(y))


def get_accuracy(y_predict, y_true):
    accuracy_num = 0
    total_num = len(y_predict)
    for i in range(total_num):
        y = y_true[i][0]
        y_ = y_predict[i]
        logger.debug(y)
        logger.debug(y_)
        if np.where(y_ == np.max(y_)) == np.where(y == np.max(y)):
            accuracy_num += 1
    return accuracy_num / total_num

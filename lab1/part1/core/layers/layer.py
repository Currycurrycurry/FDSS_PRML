import abc
import numpy as np
from lab1.part1.util.logger_util import logger


class BaseLayer(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self,last_layer_size=0, size=0, learning_rate=0.001, weights=None, biases=None):
        self.last_layer_size = last_layer_size
        self.size = size
        self.learning_rate = learning_rate
        self.weights = weights
        self.biases = biases
        self.inputs = np.mat(np.zeros(last_layer_size))
        self.outputs = np.mat(np.zeros(size))

    def set_last_layer_size(self, last_layer_size):
        self.last_layer_size = last_layer_size
        self.inputs = np.mat(np.zeros(last_layer_size))

    def set_size(self, size):
        self.size = size
        self.outputs = np.mat(np.zeros(size))

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_weights(self, weights):
        if weights:
            self.weights = weights
        else:
            self.weights = np.mat(np.random.rand(self.last_layer_size, self.size)) * 0.01

    def set_biases(self, biases):
        if biases:
            self.biases = biases
        else:
            self.biases = np.mat(np.random.rand(self.size)) * 0.01

    @abc.abstractmethod
    def optimized(self):
        return self.outputs

    @abc.abstractmethod
    def get_backward_optimized_delta(self, errors):
        return np.mat(np.zeros(self.size))

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) - self.biases
        self.outputs = self.optimized()
        return self.outputs

    def backward(self, errors):
        delta = self.get_backward_optimized_delta(errors)
        logger.debug(delta)
        result = np.dot(delta, self.weights.T)
        logger.debug('back inputs: {}'.format(self.inputs.T))
        logger.debug('back delta: {}'.format(delta))
        self.weights += self.learning_rate * np.dot(self.inputs.T, delta)
        self.biases -= self.learning_rate * delta
        return result

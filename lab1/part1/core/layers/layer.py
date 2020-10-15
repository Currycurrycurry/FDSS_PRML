import abc
import numpy as np


class BaseLayer(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self,last_layer_size=0, size=0, learning_rate=0.001, weights=None, biases=None):
        self.last_layer_size = last_layer_size
        self.size = size
        self.learning_rate = learning_rate
        self.weights = weights
        self.biases = biases

    # @property
    # @abc.abstractmethod
    # def input_size(self):
    #     pass
    #
    #
    # @value.setter
    # @abc.abstractmethod
    # def input_size(self, input_size):
    #     return
    #
    # @property
    # @abc.abstractmethod
    # def output_size(self):
    #     pass

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
            self.weights = np.mat(np.random.rand(self.last_layer_size, self.size))

    def set_biases(self, biases):
        if biases:
            self.biases = biases
        else:
            self.biases = np.mat(np.random.rand(self.size))

    @abc.abstractmethod
    def forward(self, inputs):
        return
    
    @abc.abstractmethod
    def backward(self, errors):
        return




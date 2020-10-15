from __future__ import absolute_import

import abc
import numpy as np
from lab1.part1.core.layers.layer import BaseLayer


class DenseLayer(BaseLayer):

    # def __init__(self,last_layer_size, size, learning_rate, weights, biases):
        # self.last_layer_size = last_layer_size
        # self.size = size
        # self.learning_rate = learning_rate
        # self.weights = weights
        # self.biases = biases

    def forward(self, inputs):
        print('forward')
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) - self.biases
        return self.outputs


    
    def backward(self, errors):
        print('backward')
        for i in

if __name__ == '__main__':
   print(issubclass(DenseLayer, BaseLayer))
   print(isinstance(DenseLayer(), BaseLayer)) 
   print(dir(BaseLayer))
   print(BaseLayer.__subclasses__())

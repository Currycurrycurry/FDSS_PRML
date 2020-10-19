import numpy as np

from lab1.part1.core.layerFactory import LayerFactory
from lab1.part1.util.logger_util import logger

class FNN:
    def __init__(self, config, learning_rate=0.001):
        self.config = config
        self.network = []
        self.learning_rate = learning_rate
        self.input_size = config['input_size']
        self.layer_num = config['layer_num']
        layers = config['layers']
        last_layer_size = self.input_size
        for i in range(len(layers)):
            layer_config = layers[i]
            size = layer_config['size']
            activation=layer_config['activation']
            weights = layer_config.get('weights', None)
            weights = np.mat(weights) if weights else None
            biases = layer_config.get('biases', None)
            biases = np.mat(biases) if biases else None
            layer = LayerFactory.produce_layer(activation)
            layer.set_last_layer_size(last_layer_size)
            layer.set_learning_rate(learning_rate)
            layer.set_size(size)
            layer.set_weights(weights)
            layer.set_biases(biases)
            self.network.append(layer)
            last_layer_size = size

    def forward(self, x):
        x = np.mat(x)
        logger.debug('x is {}'.format(x))
        for layer in self.network:
            x = layer.forward(x)
        return x

    def backward(self, expected_y, output):
        errors = expected_y - output
        errors = np.mat(errors)
        logger.debug('errors are {}'.format(errors))
        for i in range(self.layer_num - 1, -1, -1):
            layer = self.network[i]
            errors = layer.backward(errors)

    def train(self, train_x, train_y, epoch_num):
        length = len(train_x)
        for _ in range(epoch_num):
            for i in range(length):
                output = self.forward(train_x[i])
                self.backward(train_y[i], output)

    def predict(self, test_x):
        results = []
        for x in test_x:
            result = self.forward(x)
            results.append(result.tolist())
        return results







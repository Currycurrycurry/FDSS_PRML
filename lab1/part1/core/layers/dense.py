from __future__ import absolute_import

import abc
import numpy as np
from lab1.part1.core.layers.layer import BaseLayer
from lab1.part1.util.logger_util import logger


class DenseLayer(BaseLayer):

    def optimized(self):
        return self.outputs

    def get_backward_optimized_delta(self, errors):
        delta = np.mat(np.zeros(self.size))
        ones = np.mat(np.ones((self.outputs.shape[0], self.outputs.shape[1])))
        for i in range(self.size):
            delta[0, i] = errors[0, i] * ones
        return delta


if __name__ == '__main__':
   print(issubclass(DenseLayer, BaseLayer))
   print(isinstance(DenseLayer(), BaseLayer)) 
   print(dir(BaseLayer))
   print(BaseLayer.__subclasses__())

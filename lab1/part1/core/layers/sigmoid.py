import numpy as np
import abc

from lab1.part1.core.layers.layer import BaseLayer
from lab1.part1.util.logger_util import logger


class SigmoidLayer(BaseLayer):

    def optimized(self):
        self.outputs = 1 / (1 + np.exp(-self.outputs))
        return self.outputs

    def get_backward_optimized_delta(self, errors):
        delta1 = np.multiply(self.outputs, 1 - self.outputs)
        delta2 = np.multiply(errors, delta1)
        return delta2


if __name__ == '__main__':
    print(issubclass(SigmoidLayer, BaseLayer))
    print(isinstance(SigmoidLayer(), BaseLayer))
    print(dir(BaseLayer))
    print(BaseLayer.__subclasses__())
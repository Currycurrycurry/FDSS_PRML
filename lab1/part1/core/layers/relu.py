import abc
import numpy as np

from lab1.part1.core.layers.layer import BaseLayer


class ReluLayer(BaseLayer):
    def optimized(self):
        self.outputs = np.maximum(0, self.outputs)
        return self.outputs

    def get_backward_optimized_delta(self, errors):
        delta = self.outputs > 0
        delta2 = np.multiply(errors, delta)
        return delta2


if __name__ == '__main__':
   print(issubclass(ReluLayer, BaseLayer))
   print(isinstance(ReluLayer(), BaseLayer)) 
   print(dir(BaseLayer))
   print(BaseLayer.__subclasses__())
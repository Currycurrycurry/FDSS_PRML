import abc
import numpy as np

from lab1.part1.core.layers.layer import BaseLayer


class SoftmaxLayer(BaseLayer):
    def optimized(self):
        tmp = np.exp(self.outputs)
        return tmp / np.sum(tmp)

    def get_backward_optimized_delta(self, errors):
        return np.diag(self.outputs.tolist()[0]) - np.dot(self.outputs.T, self.outputs)


if __name__ == '__main__':
   print(issubclass(SoftmaxLayer, BaseLayer))
   print(isinstance(SoftmaxLayer(), BaseLayer))
   print(dir(BaseLayer))
   print(BaseLayer.__subclasses__())
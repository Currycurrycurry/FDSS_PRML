import abc
import numpy as np

from lab1.part1.core.layers.layer import BaseLayer


class PReLULayer(BaseLayer):
    def optimized(self):
        output_shape = self.outputs.shape
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                if self.outputs[i, j] < 0:
                    self.outputs[i, j] *= 0.01
        return self.outputs

    def get_backward_optimized_delta(self, errors):
        output_shape = self.outputs.shape
        result = np.mat(np.ones((output_shape[0], output_shape[1])))
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                if self.outputs[i, j] < 0:
                    result[i, j] = 0.01


if __name__ == '__main__':
   print(issubclass(PReLULayer, BaseLayer))
   print(isinstance(PReLULayer(), BaseLayer))
   print(dir(BaseLayer))
   print(BaseLayer.__subclasses__())
import abc

from lab1.part1.core.layers.layer import BaseLayer

class ReluLayer(BaseLayer):

    # def __init__(self,last_layer_size, size, learning_rate, weights, biases):
    #     self.last_layer_size = last_layer_size
    #     self.size = size
    #     self.learning_rate = learning_rate
    #     self.weights = weights
    #     self.biases = biases
        
    def forward(self, inputs):
        print('forward')
    
    def backward(self, errors):
        print('backward')

if __name__ == '__main__':
   print(issubclass(ReluLayer, BaseLayer))
   print(isinstance(ReluLayer(), BaseLayer)) 
   print(dir(BaseLayer))
   print(BaseLayer.__subclasses__())
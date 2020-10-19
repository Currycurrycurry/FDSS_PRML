
from lab1.part1.core.layers.dense import DenseLayer
from lab1.part1.core.layers.relu import ReluLayer
from lab1.part1.core.layers.sigmoid import SigmoidLayer
from lab1.part1.core.layers.softmax import SoftmaxLayer
from lab1.part1.core.layers.preLU import PReLULayer
from lab1.part1.core.layers.leakyRelu import LeakyReLULayer
from lab1.part1.core.layers.elu import ELULayer
from lab1.part1.util.logger_util import logger


class LayerFactory(object):

    @staticmethod
    def produce_layer(type):
        if type == 'dense':
            logger.info(type)
            return DenseLayer()
        elif type == 'relu':
            logger.info(type)
            return ReluLayer()
        elif type == 'sigmoid':
            logger.info(type)
            return SigmoidLayer()
        elif type == 'softmax':
            logger.info(type)
            return SoftmaxLayer()
        elif type == 'prelu':
            logger.info(type)
            return PReLULayer()
        elif type == 'elu':
            logger.info(type)
            return ELULayer()
        elif type == 'leakyrelu':
            logger.info(type)
            return LeakyReLULayer()


if __name__ == '__main__':
    print(LayerFactory.produce_layer('dense'))
            


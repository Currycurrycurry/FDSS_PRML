
from lab1.part1.core.layers.dense import DenseLayer
from lab1.part1.core.layers.relu import ReluLayer
from lab1.part1.core.layers.sigmoid import SigmoidLayer
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


if __name__ == '__main__':
    print(LayerFactory.produce_layer('dense'))
            


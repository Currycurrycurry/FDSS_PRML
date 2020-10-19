import random
import json
import numpy as np
from lab1.part1.util.logger_util import logger
from lab1.part1.util.math_util import get_accuracy
from lab1.part1.util.file_util import load_data
from lab1.part1.core.mlp import FNN

if __name__ == '__main__':
    with open('../config/classification_config.json', 'r', encoding='UTF-8') as f:
        config = json.load(f)
    logger.info(config)
    mlp = FNN(config)
    train_dataset = load_data('../../train/')
    random.shuffle(train_dataset)
    logger.debug(train_dataset)
    train_x = [x[0] for x in train_dataset]
    train_y = [x[1] for x in train_dataset]
    logger.debug(train_x)
    logger.debug(train_y)

    for i in range(5):
        mlp.train(train_x, train_y, 1)
        outputs = mlp.predict(train_x)
        logger.info(outputs)
        logger.info(train_y)
        train_accuracy = get_accuracy(outputs, train_y)
        logger.info('epoch %d, train_accuracy=%.3f' % (i, train_accuracy))
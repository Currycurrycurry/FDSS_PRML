import json
import numpy as np
from lab1.part1.util.logger_util import logger
from lab1.part1.util.math_util import get_mse
from lab1.part1.core.mlp import FNN

if __name__ == '__main__':
    with open('../config/sin_config.json', 'r', encoding='UTF-8') as f:
        config = json.load(f)
    logger.info(config)
    mlp = FNN(config)
    train_x = np.arange(0, 2 * np.pi, 0.001)
    logger.debug(train_x)
    np.random.shuffle(train_x)
    logger.debug(train_x)
    train_y = np.sin(train_x)
    logger.debug(train_y)

    test_x = np.arange(0, 2 * np.pi, 0.01)
    test_y = np.sin(test_x)

    for i in range(200):
        mlp.train(train_x, train_y, 1)
        outputs = mlp.predict(train_x)
        train_errors = get_mse(outputs, train_y)
        outputs = mlp.predict(test_x)
        test_errors = get_mse(outputs, test_y)
        logger.info('epoch {}: train_error = {}, test_error = {}'.format(i, train_errors, test_errors))







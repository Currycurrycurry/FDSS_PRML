import os
import numpy as np
from PIL import Image


def load_data(dir):
    train_x, train_y = [], []
    for i in range(1, 13):
        files = os.listdir(dir+str(i)+'/')
        for file in files:
            if file.endswith('.bmp'):
                file_name = dir + str(i) + '/' + file
                image = Image.open(file_name)
                input = np.array(image).flatten() + 0
                train_x.append(input)
                label_y = np.array(np.zeros(12))
                label_y[i-1] = 1
                train_y.append(label_y)
    return list(zip(train_x, train_y))



import pandas as pd
import numpy as np
from libs import *
import matplotlib as plt

np.random.seed(3)

class FNN:
    def __init__(self, X_train, Y_train, lr=1e-6, layer_num=3, iteration_time=200):
        self.learning_rate = lr
        self.layer_num = layer_num
        self.layer_dims = [0] * layer_num
        self.iteration_time = iteration_time
        self.w = [0] + [np.random.randn(self.layer_dims[i], self.layer_dims[i-1])*0.1 for i in range(1, layer_num)] 
        self.b = [0] + [np.zeros((self.layer_dims[i], 1)) for i in range(1, layer_num)]
        self.dw = [0] * (self.layer_num + 1)
        self.db = [0] * (self.layer_num + 1)
        self.costs = []
        for i in range(iteration_time):
            final_val, caches = self.forward_propagation(X_train)
            cost = cross_entropy(final_val, Y_train)
            if i % 10 == 0:
                print('cost after {} iterations: {}'.format(i, cost))
                costs.append(cost)
            # 反向传播并更新参数
            self.backward_propagation(final_val, Y_train, caches)
        print('length of cost: {}'.format(len(costs)))
        self.draw_loss()

    
    def forward(self, x, w, b):
        return np.dot(w, x) + b
    
    def forward_propagation(self, X):
        caches = []
        a = X
        for i in range(1, self.layer_num-1):
            z = self.forward(a, self.w[i], self.b[i])
            caches.append((a, self.w[i], self.b[i], z))
            a = relu(z)
        z = self.forward(a, self.w[-1], self.b[-1])
        caches.append((a, self.w[-1], self.b[-1], z))
        final_val = sigmoid(z)
        return final_val, caches

    def relu_b(self, dA, Z):
        return np.multiply(dA, np.int64(Z > 0))
    
    def linear_b(self, dZ, cache):
        a, w, b, z = cache
        dw = np.dot(dZ, a.T)
        db = np.sum(dZ, axis=1, keepdims=True)
        da = np.dot(w.T, dZ)
        return da, dw, db
    
    def backward_propagation(self, y_, y, caches):
        m = y.shape[1]
        dz = 1 / m * (y_ - y)
        da, dw, db = self.linear_b(dz, caches[-1])
        self.dw.append(dw)
        self.db.append(db)
        for i in range(self.layer_nums-1, -1, -1):
            a, w, b, z = caches[i]
            dout = self.relu_b(da, z)
            da, dw, db = self.linear_b(dout, caches[i])
            self.dw[i+1] = dw
            self.db[i+1] = db
        # update
        for i in range(self.layer_num):
            self.w[i+1] -= self.learning_rate * self.dw[i+1]
            self.b[i+1] -= self.learning_rate * self.db[i+1]

    def draw_loss(self, costs):
        plt.clf()
        plt.plot(costs)
        plt.xlabel('iterations(10)')
        plt.ylabel('cost')
        plt.show()

    def predict(self, X_test, Y_test):
        m = Y_test.shape[1]
        Y_prediction = np.zeros((1, m))
        prob, caches = self.forward_propagation(X_test)
        pass
    
    def evaluate(self, X_evaluate, Y_evaluate):
        pass


column_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', \
                'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
housing_data = pd.read_csv('../dataset/wine/winequality-red.csv', sep=';', names=column_names)
train_dataset = housing_data.sample(frac=0.9, random_state=0)
test_dataset = housing_data.drop(train_dataset.index)
train_labels = train_dataset.pop('quality')
test_labels = test_dataset.pop('quality')

train_dataset_size = train_dataset.shape[0]

fnn = FNN(train_dataset, train_labels)

    





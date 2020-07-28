import pandas as pd
import numpy as np
import tensorflow as tf

learning_rate = 0.01
training_epochs = 10000

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housing_data = pd.read_csv('./../dataset/house/housing.data', sep='\s+', names=column_names)

train_dataset = housing_data.sample(frac=0.9, random_state=0)
test_dataset = housing_data.drop(train_dataset.index)
train_labels = train_dataset.pop('MEDV')
test_labels = test_dataset.pop('MEDV')

train_dataset_size = train_dataset.shape[0]

def normalize(data):
    return data.apply(lambda column: (column - column.mean()) / column.std())

normed_train_dataset = normalize(train_dataset)
normed_test_dataset = normalize(test_dataset)

X1 = tf.placeholder(tf.float32)
X2 = tf.placeholder(tf.float32)
X3 = tf.placeholder(tf.float32)
X4 = tf.placeholder(tf.float32)
X5 = tf.placeholder(tf.float32)
X6 = tf.placeholder(tf.float32)
X7 = tf.placeholder(tf.float32)
X8 = tf.placeholder(tf.float32)
X9 = tf.placeholder(tf.float32)
X10 = tf.placeholder(tf.float32)
X11 = tf.placeholder(tf.float32)
X12 = tf.placeholder(tf.float32)
X13 = tf.placeholder(tf.float32)

rng = np.random
w1 = tf.Variable(rng.randn(), name="weights", dtype=tf.float32)
w2 = tf.Variable(rng.randn(), name="weights", dtype=tf.float32)
w3 = tf.Variable(rng.randn(), name="weights", dtype=tf.float32)
w4 = tf.Variable(rng.randn(), name="weights", dtype=tf.float32)
w5 = tf.Variable(rng.randn(), name="weights", dtype=tf.float32)
w6 = tf.Variable(rng.randn(), name="weights", dtype=tf.float32)
w7 = tf.Variable(rng.randn(), name="weights", dtype=tf.float32)
w8 = tf.Variable(rng.randn(), name="weights", dtype=tf.float32)
w9 = tf.Variable(rng.randn(), name="weights", dtype=tf.float32)
w10 = tf.Variable(rng.randn(), name="weights", dtype=tf.float32)
w11 = tf.Variable(rng.randn(), name="weights", dtype=tf.float32)
w12 = tf.Variable(rng.randn(), name="weights", dtype=tf.float32)
w13 = tf.Variable(rng.randn(), name="weights", dtype=tf.float32)

b = tf.Variable(rng.randn(), name="biases", dtype=tf.float32)

Y = tf.placeholder(tf.float32)

model = tf.multiply(X1, w1) + tf.multiply(X2, w2) + tf.multiply(X3, w3) + tf.multiply(X4, w4) + tf.multiply(X5, w5) + tf.multiply(X6, w6) +\
    tf.multiply(X7, w7) + tf.multiply(X8, w8) + tf.multiply(X9, w9) + tf.multiply(X10, w10) + tf.multiply(X11,w11) + tf.multiply(X12, w12)+ tf.multiply(X13, w13) + b

loss = tf.reduce_sum(tf.pow(model - Y, 2)) / (7 * train_dataset_size) # 7这个数字是调参数调出来的比较好的乘子

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict={X1: normed_train_dataset['CRIM'], X2: normed_train_dataset['ZN'], \
                                       X3: normed_train_dataset['INDUS'], X4: normed_train_dataset['CHAS'], \
                                       X5: normed_train_dataset['NOX'], X6: normed_train_dataset['RM'], \
                                       X7: normed_train_dataset['AGE'], X8: normed_train_dataset['DIS'], \
                                       X9: normed_train_dataset['RAD'], X10: normed_train_dataset['TAX'], \
                                       X11: normed_train_dataset['PTRATIO'], X12: normed_train_dataset['B'], \
                                       X13: normed_train_dataset['LSTAT'], Y: train_labels})

    predict_y = sess.run(model, feed_dict={X1: normed_test_dataset['CRIM'], X2: normed_test_dataset['ZN'], \
                                          X3: normed_test_dataset['INDUS'], X4: normed_test_dataset['CHAS'], \
                                          X5: normed_test_dataset['NOX'], X6: normed_test_dataset['RM'], \
                                           X7: normed_test_dataset['AGE'], X8: normed_test_dataset['DIS'], \
                                           X9: normed_test_dataset['RAD'], X10: normed_test_dataset['TAX'], \
                                           X11: normed_test_dataset['PTRATIO'], X12: normed_test_dataset['B'], \
                                           X13: normed_test_dataset['LSTAT']})
    print("predictions:")
    print(predict_y)

    rmse = tf.sqrt(tf.reduce_mean(tf.square((predict_y - test_labels))))
    print("\nRMSE(root mean square error): %.4f" % sess.run(rmse))

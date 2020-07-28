import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import metrics

learning_rate = 0.01
training_epochs = 10000

column_names = ['pregnant', 'Plasma glucose concentration', 'Diastolic blood pressure', \
                'Triceps skin fold thickness', 'serum insulin', 'BMI', 'pedigree', 'Age', 'Class']
housing_data = pd.read_csv('./dataset/diabetes/content.txt', sep=',', names=column_names)

train_dataset = housing_data.sample(frac=0.9, random_state=0)
test_dataset = housing_data.drop(train_dataset.index)
train_labels = train_dataset.pop('Class')
test_labels = test_dataset.pop('Class')

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

rng = np.random
w1 = tf.Variable(rng.randn(), name="weights", dtype=tf.float32)
w2 = tf.Variable(rng.randn(), name="weights", dtype=tf.float32)
w3 = tf.Variable(rng.randn(), name="weights", dtype=tf.float32)
w4 = tf.Variable(rng.randn(), name="weights", dtype=tf.float32)
w5 = tf.Variable(rng.randn(), name="weights", dtype=tf.float32)
w6 = tf.Variable(rng.randn(), name="weights", dtype=tf.float32)
w7 = tf.Variable(rng.randn(), name="weights", dtype=tf.float32)
w8 = tf.Variable(rng.randn(), name="weights", dtype=tf.float32)

b = tf.Variable(rng.randn(), name="biases", dtype=tf.float32)

Y = tf.placeholder(tf.float32)

linear = tf.multiply(X1, w1) + tf.multiply(X2, w2) + tf.multiply(X3, w3) + tf.multiply(X4, w4) + tf.multiply(X5, w5) + tf.multiply(X6, w6) +\
                tf.multiply(X7, w7) + tf.multiply(X8, w8) + b
model = tf.pow(1.0 + tf.exp(tf.multiply(linear, -1.0)), -1.0)

loss = -1.0 * tf.reduce_sum(tf.add(tf.multiply(Y, tf.log(model)), tf.multiply(1.0 - Y, tf.log(1.0 - model)))) / tf.cast(train_dataset_size, dtype=tf.float32)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict={X1: normed_train_dataset['pregnant'], X2: normed_train_dataset['Plasma glucose concentration'], \
                                       X3: normed_train_dataset['Diastolic blood pressure'], X4: normed_train_dataset['Triceps skin fold thickness'], \
                                       X5: normed_train_dataset['serum insulin'], X6: normed_train_dataset['BMI'], \
                                       X7: normed_train_dataset['pedigree'], X8: normed_train_dataset['Age'],  Y: train_labels})

    predict_y = sess.run(model, feed_dict={X1: normed_test_dataset['pregnant'], X2: normed_test_dataset['Plasma glucose concentration'], \
                                          X3: normed_test_dataset['Diastolic blood pressure'], X4: normed_test_dataset['Triceps skin fold thickness'], \
                                          X5: normed_test_dataset['serum insulin'], X6: normed_test_dataset['BMI'], \
                                           X7: normed_test_dataset['pedigree'], X8: normed_test_dataset['Age']})

    print("predictions:")
    print(predict_y)

    auc = metrics.roc_auc_score(test_labels, predict_y)  # tf的tf.metrics.auc函数貌似第二个返回值才是auc？
    print("\nAUC(area under ROC curve): " + str(auc))

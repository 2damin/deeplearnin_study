from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import csv
from tensorflow.python.platform import gfile

#tensorflow input data 생성
def get_inputs(x_data, y_data):
    x = tf.constant(x_data)
    y = tf.constant(y_data)
    return x, y

# csv파일에서 넘파이로 변환
def load_csv(file_name):
    matrix = []
    csv_file = csv.reader(gfile.Open(file_name))

    for i in csv_file:
        matrix.append(i)

    npData = np.array(matrix)

    target_value = int(npData[0, 0])
    training_value = int(npData[0, 1])

    target = np.zeros((target_value,), dtype=np.int)
    data = np.zeros((target_value, training_value), dtype=np.float32)

    for j in range(0, target_value):
        for i in range(0, training_value):
            target[j] = np.asarray(npData[j + 1, training_value], dtype=np.int)
            data[j, i] = np.asarray(npData[j + 1, i], dtype=np.float32)

    return target, data , target_value, training_value

# 데이터 정규화
def normalization(npData, target_value, training_value):
    normData = np.zeros((target_value, training_value), dtype=np.float32)
    add = -1
    for j in range(0, target_value):
        mean = np.mean(npData[j, :])
        var = np.var(npData[j, :])
        min = np.min(npData[j, :])
        max = np.max(npData[j, :])

        if(j % (target_value/training_value) == 0):
            add += 1

        for i in range(0, training_value):
            #normData[j, i] = np.asarray(((npData[j, i] - mean) / np.sqrt(var) + add) , dtype=np.float32)
            normData[j, i] = np.asarray(((npData[j, i] - min) / (max - min) + add), dtype=np.float32)

    return normData




# Name of Data sets file
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"
IRIS_EVAL = "iris_eval.csv"



target, data, target_value, training_value = load_csv(IRIS_TRAINING)
target_t, data_t, target_value_t, training_value_t = load_csv(IRIS_TEST)
target_eval, data_eval, target_value_eval, training_value_eval = load_csv(IRIS_EVAL)
"""
target : y_label
data : x_data(feature data)
target_value : data 총 개수
training_value : x_data 1개 당 feature 개수 
"""

#normalized data
normData = normalization(data, target_value, training_value)
normData_t = normalization(data_t, target_value_t, training_value_t)
normData_eval = normalization(data_eval, target_value_eval, training_value_eval)


# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)


# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=5)]


# Build 3 layer DNN.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                           hidden_units=[10, 20, 10],
                                            optimizer = tf.train.AdagradOptimizer(learning_rate=0.1),
                                            n_classes=5,
                                            enable_centered_bias=False,
                                            model_dir="iris_model")

# Fit model.
classifier.fit(x= normData, y= target, steps=1000, batch_size= 8)

# Test_Accuracy.
test_accuracy_score = classifier.evaluate(x= normData_t, y= target_t)["accuracy"]
print('Test_Accuracy: {0:f}'.format(test_accuracy_score))

# Make Evaluation_data.
y =list(classifier.predict(normData_eval, as_iterable=True))
acc = []
for i in range(len(y)):
    acc.append(y[i] == target_eval[i])

# Eval_Accuracy
eval_acc = acc.count(True) / len(acc)
print("Eval accuracy : {}".format(eval_acc))
print("Total number of eval data : {}".format(len(acc)))

from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn


sentence = ("if you want to build a ship, don't drum up people together to"
            " collect wood and don't assign them tasks and work, but rather"
            " teach them to long for the endless immensity of the sea.")

#sentence = ("  형식은, 아뿔싸! 내가 어찌하여 이러한 생각을 하는가, 내 마음이 이렇게 약하던가 하면서 두 주먹을 불끈 쥐고 전신에 힘을 주어 이러한 약한 생각을 떼어 버리려 하나, 가슴속에는 이상하게 불길이 확확 일어난다.")


char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}

seq_length = 10
learning_rate = 0.05
data_dim = len(char_set)
hidden_size = len(char_set)
num_classes = len(char_set)


dataX = []
dataY = []
for i in range(0, len(sentence)- seq_length):
    x_str = sentence[i:i+seq_length]
    y_str = sentence[i+1: i+seq_length+1]
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]

    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX)

X = tf.placeholder(tf.int32, [None,seq_length])
Y = tf.placeholder(tf.int32, [None, seq_length])

#one hot encoding
X_onehot = tf.one_hot(X, num_classes)

def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    return cell

multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(3)], state_is_tuple=True)

outputs, _states =tf.nn.dynamic_rnn(multi_cells, X_onehot, dtype=tf.float32)

X_for_FC = tf.reshape(outputs, [-1,hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_FC, num_classes, activation_fn = None)

#softmax
X_softmax = tf.reshape(outputs, [-1, hidden_size])
softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
softmax_b = tf.get_variable("softmax_b", [num_classes])
outputs = tf.matmul(X_softmax, softmax_w) + softmax_b

#reshape out for seq_loss
outputs = tf.reshape(outputs, [batch_size, seq_length, num_classes])

weights = tf.ones([batch_size, seq_length])

seq_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights = weights)
mean_loss = tf.reduce_mean(seq_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(5000):
    _, l, results =sess.run([train, mean_loss, outputs], feed_dict={X: dataX, Y: dataY})
    for j, result in enumerate(results):
        index = np.argmax(result, axis =1)
        print(i,j,''.join([char_set[t] for t in index]), l)

results = sess.run(outputs, feed_dict={X: dataX})
for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j is 0:
        print(''.join([char_set[t] for t in index]), end='')
    else:
        print(char_set[index[-1]], end='')

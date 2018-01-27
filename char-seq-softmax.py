import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

sample = "if you want you."
idx2char = list(set(sample))
char2idx = {c: i for i, c in enumerate(idx2char)}

# hyper para
dic_size = len(char2idx)
rnn_hidden_size = len(char2idx)
sequence_length = len(sample)-1
num_classes = len(char2idx)
batch_size = 1
learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample]
x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]

X = tf.placeholder(tf.int32, [None, sequence_length]) #x_data
Y = tf.placeholder(tf.int32, [None, sequence_length]) #y_label

#flatten the data
X_onehot = tf.one_hot(X, num_classes)
X_for_softmax = tf.reshape(X_onehot, [-1, rnn_hidden_size])

#softmax_w = tf.get_variable("softmax_w", [rnn_hidden_size, num_classes])
#softmax_b = tf.get_variable("softmax_b", [num_classes])
#outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b

cell = tf.contrib.rnn.BasicLSTMCell(num_units = rnn_hidden_size, state_is_tuple = True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X_onehot, initial_state= initial_state, dtype= tf.float32)


#expend the data (revive the batches
outputs = tf.reshape(outputs, [batch_size,sequence_length,num_classes])
weights = tf.ones([batch_size, sequence_length])

#compute sequence loss
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits = outputs, targets = Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print(i, "loss:", l, "prediction:", result, "".join(result_str))
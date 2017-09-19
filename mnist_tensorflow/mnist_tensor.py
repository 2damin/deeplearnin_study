import sys, os
sys.path.append(os.pardir)
import tensorflow as tf
import mnist_tensorflow.mnist.input_data as data
import matplotlib.pyplot as plt

def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'), b))


def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


#read data
mnist = data.read_data_sets("/tmp/data/", one_hot=True)


#define all CNN parameters
learning_rate = 0.01
training_iters = 100000
batch_size = 128
display_step = 10
n_input = 784 #data size(28*28)
n_classes = 10
dropout = 0.75

x = tf.placeholder(tf.float32, [None, n_input])
_X = tf.reshape(x, shape =[-1, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

loss_plot = []
step_plot = []


#First
wc1 = tf.Variable(tf.random_normal([5, 5, 1, 32], name='w'))
bc1 = tf.Variable(tf.random_normal([32], name ='bias'))

conv1 = conv2d(_X, wc1, bc1)
conv1 = max_pool(conv1, k=2)
conv1 = tf.nn.dropout(conv1, keep_prob)

#Second
wc2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
bc2 = tf.Variable(tf.random_normal([64]))

conv2 = conv2d(conv1, wc2, bc2)
conv2 = max_pool(conv2, k=2)
conv2 = tf.nn.dropout(conv2, keep_prob)

#densely connected layer
wd1 = tf.Variable(tf.random_normal([7*7*64, 1024]))
bd1 = tf.Variable(tf.random_normal([1024]))

dense1 = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]])
dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, wd1), bd1))
dense1 = tf.nn.dropout(dense1, keep_prob)

#Readout layer
wout = tf.Variable(tf.random_normal([1024, n_classes]))
bout = tf.Variable(tf.random_normal([n_classes]))

pred = tf.add(tf.matmul(dense1, wout), bout)

#testing and traing the model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#launch
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 1

    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})

        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})

            print("Iter" + str(step*batch_size) + ", Minibatch loss=" + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

            #make matrix for plot
            loss_plot.append(loss)
            step_plot.append(step)

        step += 1

    print("Optimization finished")

    print("Testing accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:10000], y: mnist.test.labels[:10000], keep_prob: 1.0}))

    plt.plot(step_plot, loss_plot)
    plt.show()








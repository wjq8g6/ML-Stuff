import pandas as pd
import tensorflow as tf
import csv
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

training_iters = 500
batch_size = 50
display_step = 10
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

x = tf.placeholder(tf.float32,shape=[None, 784])
y_place = tf.placeholder(tf.float32, shape=[None,10])
X_test = pd.read_csv('test.csv').values


weights = tf.Variable(tf.zeros([784,10]))

bias = tf.Variable(tf.zeros([10]))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, weights):
    return tf.nn.conv2d(x, weights, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

def cnn_model(x):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    keep_prob = 0.75
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    output = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return output



def train(x):
    prediction = cnn_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_place,logits=prediction))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    n_epochs = 20

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            epoch_loss = 0
            for k in range(training_iters):
                curr_x, curr_y = mnist.train.next_batch(batch_size)
                k, c = sess.run([optimizer,cost], feed_dict = {x: curr_x, y_place: curr_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of ', n_epochs, ' loss: ', epoch_loss)
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y_place,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float32'))
        print('Accuracy: ', accuracy.eval({x: mnist.test.images, y_place:mnist.test.labels}))
        predict = sess.run(prediction, feed_dict = {x:X_test})
        return predict


ret = train(x)
newArr = []
newArr.append(['ImageId','Label'])
with open('pred.csv','w',newline='') as file:
    cwr = csv.writer(file, delimiter=',')
    count = 1
    for k in ret:
        blank = []
        blank.append(count)
        blank.append(np.argmax(k))
        newArr.append(blank)
        count+=1
    #np.savetxt(file, newArr, delimiter=',')
    cwr.writerows(newArr)

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
training_iters = 20000
batch_size = 100
display_step = 10
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

n_nodes_hl1 = 350
n_nodes_hl2 = 350
n_nodes_hl3 = 350

x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float')

def nn_model(data):
    hidden1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
               'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
               'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
               'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
               'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden1_layer['weights']),hidden1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden2_layer['weights']), hidden2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden3_layer['weights']), hidden3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
    return output

def train(x):
    prediction = nn_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    n_epochs = 50

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            epoch_loss = 0
            for k in range(int(mnist.train.num_examples/batch_size)):
                curr_x, curr_y = mnist.train.next_batch(batch_size)
                k, c = sess.run([optimizer,cost], feed_dict = {x: curr_x, y: curr_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of ', n_epochs, ' loss: ', epoch_loss)
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy: ', accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))

train(x)
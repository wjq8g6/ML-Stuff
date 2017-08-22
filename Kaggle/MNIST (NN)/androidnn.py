
from __future__ import print_function
import pandas as pd
import tensorflow as tf
import csv
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

n_nodes_hl1 = 200
n_nodes_hl2 = 200
n_nodes_hl3 = 200

x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float')

all_data = pd.read_csv('android.csv')
Y_train = all_data['label']
X_train = all_data.drop(['label'],axis=1)

hidden1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
hidden2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                 'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
hidden3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                 'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'biases': tf.Variable(tf.random_normal([n_classes]))}

def nn_model(data):

    l1 = tf.add(tf.matmul(data, hidden1_layer['weights']),hidden1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden2_layer['weights']), hidden2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden3_layer['weights']), hidden3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
    return output

def one_hot(curr_y):
    lst = [0,0,0,0,0,0,0,0,0,0]
    lst[curr_y[0]] = 1
    return lst

def train(x):
    prediction = nn_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    n_epochs = 3

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            epoch_loss = 0
            for k in range(len(Y_train)):
                curr_x = X_train.iloc[[k]].values
                curr_y = one_hot(Y_train.iloc[[k]].values)
                i, c = sess.run([optimizer,cost], feed_dict = {x: curr_x, y: curr_y})
                epoch_loss += c
                if((k % 1000) == 0):
                    print('Epoch', k, 'completed out of ', len(Y_train), ' loss: ', epoch_loss)
                    epoch_loss = 0
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy: ', accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))
        variables_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)
        with open('weights.txt', 'w',newline='\n') as file:
            w_count = 1
            b_count = 1
            for i in range(len(values)):
                v = values[i]
                if (i % 2) == 0:
                    file.write("Weights " + str(w_count)+ "\n")
                    for j in v:
                        for k in j:
                            file.write(str(k)+ "\n")
                    w_count+=1
                else:
                    file.write("Biases " + str(b_count) + "\n")
                    for j in v:
                        file.write(str(j)+ "\n")
                    b_count += 1

train(x)


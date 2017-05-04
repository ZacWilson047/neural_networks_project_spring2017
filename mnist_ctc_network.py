""""
A Bidirectional RNN with a CTC output layer to perform classification in the MNIST dataset.

Note: this program is a modification of a static bidirectional RNN by Aymeric Damien.
The original program can be found at the link below:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

"""
To classify images using a bidirectional recurrent neural network, we consider
every image row as a sequence of pixels. Because MNIST image shape is 28*28px,
we will then handle 28 sequences of 28 steps for every sample.
"""

# Parameters
learning_rate = 0.01
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_labels = 10 # MNIST total classes (0-9 digits)
n_classes = 11 # num labels + 1 for the blank label

# tf Graph input -- has shape [batch_size, n_steps, n_input]
x = tf.placeholder("float", [None, n_steps, n_input], name="x_placeholder")
targets = tf.sparse_placeholder(tf.int32, name="targets_placeholder")


def bidirectional_rnn(x):
    # data input shape: (batch_size, n_steps, n_input)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=lstm_fw_cell,
        cell_bw=lstm_bw_cell,
        inputs=x,
        sequence_length=[n_steps]*batch_size,
        dtype=tf.float32)

    # concatenate the forward and backward layer outputs
    recurrent_layer_output = tf.concat([output_fw, output_bw], 2)

    output = tf.layers.dense(
        inputs=recurrent_layer_output,
        units=n_classes)
    return output

# prediction (i.e. logits)
pred = bidirectional_rnn(x)

# Define loss and optimizer
# Put pred tensor in time-major order (as opposed to batch-major order)
pred = tf.transpose(pred, (1, 0, 2))


# seq_len placeholder has shape [batch_size]
seq_len = tf.placeholder(tf.int32, [None])
loss = tf.nn.ctc_loss(targets, pred, seq_len, time_major=True)
cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
decoded, log_prob = tf.nn.ctc_greedy_decoder(pred, seq_len)
label_error_rate = tf.reduce_mean(
    tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

def process_y_batch(batch_y):
    """Returns (indices, values)."""
    batch_y = batch_y.astype(np.int32).tolist()
    indices = []
    values = []
    batch_size = len(batch_y)
    for batch_no in xrange(batch_size):
        val = batch_y[batch_no].index(1)
        indices.append([batch_no, 0])
        values.append(val)
    return indices, values

# Launch the graphn
with tf.Session() as sess:
    seq_len_val = [n_steps] * batch_size
    # Initializing the variables
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Transform Y into an int32 sparse tensor 
        indices, values = process_y_batch(batch_y)
        indices = tf.cast(indices, tf.int32)
        values = tf.cast(values, tf.int32)
        shape = tf.cast(tf.shape(batch_y), tf.int64)
        sparse_batch_y = (indices, values, shape)
        sparse_batch_y = sess.run(sparse_batch_y)

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, targets: sparse_batch_y, seq_len: seq_len_val})
        if step % display_step == 0:
            # Calculate batch label error rate
            ler = sess.run(label_error_rate, feed_dict={x: batch_x, targets: sparse_batch_y, seq_len: seq_len_val})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, targets: sparse_batch_y, seq_len: seq_len_val})
            print("Iter " + str(step*batch_size) + ", Minibatch LER = " + \
                  "{:.6f}, Minibatch loss = {:.6f}".format(ler, loss))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    test_y = tf.cast(test_label, tf.int32)
    indices, values = process_y_batch(test_label)
    indices = tf.cast(indices, tf.int32)
    values = tf.cast(values, tf.int32)
    
    shape = tf.cast(tf.shape(test_y), tf.int64)
    sparse_test_y = (indices, values, shape)
    sparse_test_y = sess.run(sparse_test_y)
    print("Testing Label Error Rate:", \
        sess.run(label_error_rate, feed_dict={x: test_data, targets: sparse_test_y, seq_len: [n_steps]*128}))

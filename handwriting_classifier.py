"""Trains a Connectionist Temporal Classification network to perform online
handwriting recognition on the IAM Online Database."""

from __future__ import print_function

import glob
import json
import random
import sys
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

FLAGS = None
TARGET_CHARS_FILE = "/Users/zacwilson/data/iam_handwriting/characters.json"
with open(TARGET_CHARS_FILE, "r") as target_chars_fp:
    TARGET_CHARS = json.load(target_chars_fp)

# Parameters
learning_rate = 0.01
training_iters = 100000
batch_size = 128
display_step = 10


# Get list of full filepaths for all serialized training examples
# This variable gets shuffled at the start of each new training epoch
training_filenames = glob.glob(
    "/Users/zacwilson/data/iam_handwriting/training/*")

# Network Parameters
n_input = 5 # input feautures: x_coordinates, y_coordinates, timestamps,
            # is_stroke_start, and is_stroke_end, in that order
n_hidden = 128 # hidden layer num of features
n_labels = len(TARGET_CHARS) # num different characters in all target labels
n_classes = n_labels + 1 # add 1 to n_labels to account for blank label


def main():
    """Trains a CTC network to perform online handwriting recognition."""
    # tf Graph input -- has shape [batch_size, seq_len, n_input]
    x = tf.placeholder("float", [None, None, n_input], name="x_placeholder")
    targets = tf.sparse_placeholder(tf.int32, name="targets_placeholder")

    # seq_len placeholder has shape [batch_size]
    seq_len = tf.placeholder(tf.int32, [None])

    # prediction (i.e. logits)
    pred = BiRNN(x, seq_len)

    # Put pred tensor in time-major order (as opposed to batch-major order)
    pred = tf.transpose(pred, (1, 0, 2))
    #print("shape of pred (now in time-major): " + str(pred.shape)) # DEBUG LINE

    # Define loss and optimizer
    loss = tf.nn.ctc_loss(targets, pred, seq_len, time_major=True)
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    #decoded, log_prob = tf.nn.ctc_greedy_decoder(pred, seq_len)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(pred, seq_len)
    label_error_rate = tf.reduce_mean(
        tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    # Launch the graph
    with tf.Session() as sess:
        # Initializing the variables
        init = tf.global_variables_initializer()
        sess.run(init)

        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x, batch_y, seq_len_vals = next_train_batch(batch_size)
            sparse_batch_y = sess.run(batch_y)

            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, targets: sparse_batch_y, seq_len: seq_len_vals})
            if step % display_step == 0:
                # Calculate batch label error rate
                ler = sess.run(label_error_rate, feed_dict={x: batch_x, targets: sparse_batch_y, seq_len: seq_len_vals})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, targets: sparse_batch_y, seq_len: seq_len_vals})
                print("Iter " + str(step*batch_size) + ", Minibatch LER = " + \
                      "{:.6f}, Minibatch loss = {:.6f}".format(ler, loss))
            step += 1
        print("Optimization Finished!")


def BiRNN(x, seq_lens):
    """TODO: full docstring; seq_lens is np_array of actual input seq lens.
    Actually seq_lens is a tf.placeholder"""
    # data input shape: (batch_size, seq_lens, n_input)

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
        sequence_length=seq_lens,
        dtype=tf.float32)

    # concatenate the forward and backward layer outputs
    recurrent_layer_output = tf.concat([output_fw, output_bw], 2)

    output = tf.layers.dense(
        inputs=recurrent_layer_output,
        units=n_classes)
    return output


def next_train_batch(batch_size):
    """Returns the next training batch.

    If next batch corresponds to the start of a new epoch, then
    the global variable training_filenames will be shuffled in place.

    Args:
        batch_size: The size of the batch that will be returned.

    Returns:
        A tuple of (batch_x, batch_y, seq_len_vals). batch_x is
        a six dimensional numpy array of with dtype=numpy.float32.
        The first dimension of batch_x corresponds to the batch size.
        The five remaining dimensions correspond to the following
        five features: x_coordinates, y_coordinates, timestamps,
        is_stroke_start, and is_stroke_start, in that order. batch_y contains
        data for the label sequence represented as the tuple
        (indices, values, shape) used to feed the sparse_batch_y tensor.
        The i'th integer in the values array corresponds to the index of the
        i'th character of the label sequence in the TARGET_CHARS array.
        seq_len_vals is a one dimensional vector (num array with
        dtype=numpy.int32) of length batch_size containing the actual
        lengths of input data vectors for each batch.

    Raises:
        IOError: If an IOError occurs when trying to read one of the
            serialized training files.

    """
    # get value of "static" next_train_batch variable
    if not hasattr(next_train_batch, "cur_index"):
        next_train_batch.cur_index = 0 # doesn't exist yet, so initialize it
    next_index = next_train_batch.cur_index + batch_size

    batch_filenames = [] # training files to use for this batch
    if next_index >= len(training_filenames): # new epoch detected
        batch_filenames.extend(training_filenames[next_train_batch.cur_index:])
        random.shuffle(training_filenames)
        next_train_batch.cur_index = next_index % (len(training_filenames) - 1)
        next_train_batch.cur_index -= 1
        batch_filenames.extend(training_filenames[:next_train_batch.cur_index])
    else:
        cur_index = next_train_batch.cur_index
        batch_filenames = training_filenames[cur_index:next_index]
        next_train_batch.cur_index = next_index

    deserialized_examples = []
    longest_input_len = 0
    longest_label_len = 0
    for training_filename in batch_filenames:
        with open(training_filename, "r") as filein:
            example = json.load(filein)
            label_len = len(example["labels"])
            longest_input_len = max(longest_input_len, example["seq_length"])
            longest_label_len = max(longest_label_len, label_len)
            deserialized_examples.append(example)

    batch_x = np.empty([batch_size, longest_input_len, n_input], dtype=np.float32)
    batch_y_indices = []
    batch_y_values = []
    seq_lens = np.empty([batch_size], dtype=np.int32)

    for i in xrange(len(deserialized_examples)):
        example = deserialized_examples[i]
        seq_lens[i] = example["seq_length"]
        for j in xrange(example["seq_length"]):
            batch_x[i][j][0] = example["x_coordinates"][j]
            batch_x[i][j][1] = example["y_coordinates"][j]
            batch_x[i][j][2] = example["timestamps"][j]
            batch_x[i][j][3] = example["is_stroke_start"][j]
            batch_x[i][j][4] = example["is_stroke_end"][j]
        labels = example["labels"]
        for j in xrange(len(labels)):
            batch_y_values.append(labels[j])
            batch_y_indices.append([i, j])
    batch_y_indices = tf.cast(batch_y_indices, tf.int32)
    batch_y_values = tf.cast(batch_y_values, tf.int32)
    batch_y_shape = tf.cast([batch_size, longest_label_len], tf.int64)
    sparse_batch_y = (batch_y_indices, batch_y_values, batch_y_shape)
    return batch_x, sparse_batch_y, seq_lens


if __name__ == "__main__":
    main()

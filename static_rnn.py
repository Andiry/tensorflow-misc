import tensorflow as tf
import numpy as np

ni = 3  # n_inputs
nn = 5  # n_neurons

X0 = tf.placeholder(tf.float32, [None, ni])
X1 = tf.placeholder(tf.float32, [None, ni])

Wx = tf.Variable(tf.random_normal(shape=[ni, nn], dtype=tf.float32))
Wy = tf.Variable(tf.random_normal(shape=[nn, nn], dtype=tf.float32))
b = tf.Variable(tf.zeros([1, nn], dtype=tf.float32))

Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)

init = tf.global_variables_initializer()

X0_batch = np.array([[0,1,2],[3,4,5],[6,7,8],[9,0,1]])
X1_batch = np.array([[9,8,7],[0,0,0],[6,5,4],[3,2,1]])

print("Test 1")
with tf.Session() as sess:
  init.run()
  Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict = {X0: X0_batch, X1: X1_batch})
  print(Y0_val)
  print(Y1_val)

# static_rnn

print("Test 2")
tf.reset_default_graph()

X0 = tf.placeholder(tf.float32, [None, ni])
X1 = tf.placeholder(tf.float32, [None, ni])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = nn)
output_seqs, states = tf.contrib.rnn.static_rnn(
    basic_cell, [X0, X1], dtype = tf.float32)
Y0, Y1 = output_seqs

init = tf.global_variables_initializer()
with tf.Session() as sess:
  init.run()
  Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict = {X0: X0_batch, X1: X1_batch})
  print(Y0_val)
  print(Y1_val)


print("Test 3")
tf.reset_default_graph()
n_steps = 2
X = tf.placeholder(tf.float32, [None, n_steps, ni])
X_seqs = tf.unstack(tf.transpose(X, perm=[1,0,2]))
basic_cell1 = tf.contrib.rnn.BasicRNNCell(num_units = nn)
output_seqs, states = tf.contrib.rnn.static_rnn(
    basic_cell1, X_seqs, dtype = tf.float32)
outputs = tf.transpose(tf.stack(output_seqs), perm = [1,0,2])

X_batch = np.array([
    [[0,1,2],[3,4,5]],[[6,7,8],[9,0,1]],
    [[9,8,7],[0,0,0]],[[6,5,4],[3,2,1]]])

init = tf.global_variables_initializer()
with tf.Session() as sess:
  init.run()
  output_val = sess.run(outputs, feed_dict = {X: X_batch})
  print(output_val)

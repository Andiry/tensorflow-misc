import tensorflow as tf
from tensorflow.contrib.layers import fully_connected as fc

ns = 28 # n_steps
ni = 28 # n_inputs
no = 10 # n_outputs
nn = 150 # n_neurons

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, ns, ni])
y = tf.placeholder(tf.int32, [None])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = nn)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype = tf.float32)
print(outputs.shape, states.shape)

logits = fc(states, no, activation_fn=None)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=y, logits=logits)

loss = tf.reduce_mean(xentropy)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

correct = tf.nn.in_top_k(logits, y,1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/")
X_test = mnist.test.images.reshape((-1, ns, ni))
y_test = mnist.test.labels

ne = 20
bs = 150

options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

with tf.Session() as sess:
  init.run()
  for epoch in range(ne):
    for iteration in range(mnist.train.num_examples // bs):
      X_batch, y_batch = mnist.train.next_batch(bs)
      X_batch = X_batch.reshape((-1, ns, ni))
      sess.run(train_op, feed_dict = {X:X_batch, y:y_batch},
               options=options, run_metadata=run_metadata)
      pass

    acc_train = accuracy.eval(feed_dict = {X:X_batch, y:y_batch})
    acc_test = accuracy.eval(feed_dict = {X:X_test, y:y_test})
    print(epoch, "Train accuracy: ", acc_train, "Test accuracy: ", acc_test)

  writer = tf.summary.FileWriter('/tmp/mnist_rnn', sess.graph)
  writer.close()
  tf.train.write_graph(sess.graph, '/tmp/', 'mnist_rnn.pbtxt')
  open('/tmp/mnist_rnn-stepstats.pbtxt', 'w').write(str(run_metadata.step_stats))

meta_graph_def = tf.train.export_meta_graph(filename='/tmp/mnist_rnn.meta')

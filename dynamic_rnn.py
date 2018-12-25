import tensorflow as tf
import numpy as np

ni = 3  # n_inputs
nn = 5  # n_neurons
n_steps = 2

X = tf.placeholder(tf.float32, [None, n_steps, ni])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = nn)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

X_batch = np.array([
    [[0,1,2],[3,4,5]],[[6,7,8],[9,0,1]],
    [[9,8,7],[0,0,0]],[[6,5,4],[3,2,1]]])

init = tf.global_variables_initializer()
options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

with tf.Session() as sess:
  init.run()
  output_val, states_val = sess.run([outputs, states], feed_dict = {X: X_batch},
                                    options=options, run_metadata=run_metadata)
  print(output_val)
  print(states_val)
  tf.train.write_graph(sess.graph, '/tmp/', 'dynamic_rnn.pbtxt')
  open('/tmp/dynamic_rnn-stepstats.pbtxt', 'w').write(str(run_metadata.step_stats))

meta_graph_def = tf.train.export_meta_graph(filename='/tmp/dynamic_rnn.meta')

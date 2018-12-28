import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("/tmp/data/")

n_inputs = 28 * 28
n_outputs = 10

with tf.name_scope('input'):
  x = tf.placeholder(tf.float32, shape=(None, n_inputs), name="x-input")
  y_ = tf.placeholder(tf.int64, shape=(None), name="y-input")

with tf.name_scope('input_reshape'):
  image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
  tf.summary.image('input', image_shaped_input, 10)
  pass

with tf.name_scope('softmax_layer'):
  with tf.name_scope('weights'):
    weights = tf.Variable(tf.zeros([n_inputs, n_outputs]))
    tf.summary.histogram('weights', weights)
    pass
  with tf.name_scope('biases'):
    biases = tf.Variable(tf.zeros([n_outputs]))
    tf.summary.histogram('biases', biases)
    pass
  with tf.name_scope('Wx_plus_b'):
    y = tf.matmul(x, weights) + biases

with tf.name_scope("cross_entropy"):
  diff = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y)
  with tf.name_scope('total'):
    cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope("train"):
  train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

with tf.name_scope("accuracy"):
  with tf.name_scope('correct_prediction'):
    correct = tf.nn.in_top_k(y, y_, 1)
    pass
  accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

def feed_dict(train):
  if train:
    xs, ys = mnist.train.next_batch(100, fake_data=False)
  else:
    xs, ys = mnist.test.images, mnist.test.labels
  return {x: xs, y_: ys}

merged = tf.summary.merge_all()
init = tf.global_variables_initializer()

n_epochs = 2000

with tf.Session() as sess:
  train_writer = tf.summary.FileWriter('/tmp/summary/mnist/train', sess.graph)
  test_writer = tf.summary.FileWriter('/tmp/summary/mnist/test')
  init.run()

  for i in range(n_epochs):
    if i % 10 == 0:
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:
      if i % 100 == 99:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True),
                              options=run_options, run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  train_writer.close()
  test_writer.close()


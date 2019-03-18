"""TODO(andiryxu): DO NOT SUBMIT without one-line documentation for while.

TODO(andiryxu): DO NOT SUBMIT without a detailed description of while.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import tensorflow as tf

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  i = tf.constant(0)
  c = lambda i: tf.less(i, 10)
  b = lambda i: tf.add(i, 1)
  r = tf.while_loop(c, b, [i])

  init = tf.global_variables_initializer()

  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()

  print('Start...')
  with tf.Session(target="local") as sess:
    sess.run(init)
    print(sess.run(r, options=options, run_metadata=run_metadata))
    tf.train.write_graph(sess.graph, '/tmp/', 'while.pbtxt')
    open('/tmp/while-stepstats.pbtxt', 'w').write(str(run_metadata.step_stats))
    writer = tf.summary.FileWriter('/tmp/while', sess.graph)
    writer.close()

  meta_graph_def = tf.train.export_meta_graph(filename='/tmp/while.meta')


if __name__ == '__main__':
  app.run(main)

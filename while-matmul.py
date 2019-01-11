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
  m0 = tf.ones([2, 2])
  m = tf.ones([2, 2])
  c = lambda i, m: tf.less(i, 10)
  b = lambda i, m: [tf.add(i, 1), tf.matmul(m, m0)]
  r = tf.while_loop(c, b, [i, m])

  init = tf.global_variables_initializer()

  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()

  name = 'while-matmul'
  dump_dir = '/tmp/'
  graph_path = ''.join([name, '.pbtxt'])
  step_stats_path = ''.join([dump_dir, name, '-stepstats.pbtxt'])
  tensorboard_path = ''.join([dump_dir, name])
  meta_graph_path =  ''.join([dump_dir, name, '.meta'])

  print('Start...')
  with tf.Session() as sess:
    sess.run(init)
    print(sess.run(r, options=options, run_metadata=run_metadata))

    tf.train.write_graph(sess.graph, dump_dir, graph_path)
    open(step_stats_path, 'w').write(str(run_metadata.step_stats))
    writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
    writer.add_run_metadata(run_metadata, 'step001')
    writer.close()

  meta_graph_def = tf.train.export_meta_graph(filename=meta_graph_path)

if __name__ == '__main__':
  app.run(main)

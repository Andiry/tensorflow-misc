"""TODO(andiryxu): DO NOT SUBMIT without one-line documentation for test.

TODO(andiryxu): DO NOT SUBMIT without a detailed description of test.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  with tf.device('/device:GPU:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    pass

  c = tf.matmul(a, b)
  # Creates a session with log_device_placement set to True.
  sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
  # Runs the op.
  print(sess.run(c))

if __name__ == '__main__':
  tf.app.run(main)

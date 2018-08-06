import tensorflow as tf
import os
import sys
import reader

DATA_PATH = "../simple-examples/data/"
train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)
print(len(train_data))
print(train_data[:100])

result = reader.ptb_producer(train_data, 4, 5)

with tf.Session() as sess:
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess = sess, coord = coord)
  for i in range(3):
    x, y = sess.run(result)
    print("X%d: " %(i), x)
    print("Y%d: " %(i), y)
  coord.request_stop()
  coord.join(threads)


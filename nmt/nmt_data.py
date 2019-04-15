"""TODO(andiryxu): DO NOT SUBMIT without one-line documentation for nmt_data.

TODO(andiryxu): DO NOT SUBMIT without a detailed description of nmt_data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

MAX_LEN = 50
SOS_ID = 1

def MakeDataset(file_path):
  dataset = tf.data.TextLineDataset(file_path)
  dataset = dataset.map(lambda string : tf.string_split([string]).values)
  dataset = dataset.map(lambda string : tf.string_to_number(string, tf.int32))

  dataset = dataset.map(lambda x : (x, tf.size(x)))
  return dataset


def MakeSrcTrgDataset(src_path, trg_path, batch_size):
  src_data = MakeDataset(src_path)
  trg_data = MakeDataset(trg_path)

  # [0][0] : src sentence
  # [0][1] : src sentence length
  # [1][0] : trg sentence
  # [1][1] : trg sentence length
  dataset = tf.data.Dataset.zip((src_data, trg_data))

  def FilterLength(src_tuple, trg_tuple):
    ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
    src_len_ok = tf.logical_and(tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN))
    trg_len_ok = tf.logical_and(tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN))
    return tf.logical_and(src_len_ok, trg_len_ok)

  dataset = dataset.filter(FilterLength)

  def MakeTrgInput(src_tuple, trg_tuple):
    ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
    trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis = 0)
    return ((src_input, src_len), (trg_input, trg_label, trg_len))

  dataset = dataset.map(MakeTrgInput)

  dataset = dataset.shuffle(10000)

  padded_shapes = (
      (tf.TensorShape([None]), tf.TensorShape([])),
      (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([])))

  batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
  return batched_dataset


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

if __name__ == '__main__':
  tf.app.run(main)

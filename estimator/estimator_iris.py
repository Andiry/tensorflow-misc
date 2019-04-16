"""TODO(andiryxu): DO NOT SUBMIT without one-line documentation for estimator_iris.

TODO(andiryxu): DO NOT SUBMIT without a detailed description of estimator_iris.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def my_input_fn(file_path, perform_shuffle=False, repeat_count=1):
  def decode_csv(line):
    parsed_line = tf.decode_csv(line, [[0.], [0.], [0.], [0.], [0]])
    return {"x": parsed_line[:-1],}, parsed_line[-1:]

  dataset = tf.data.TextLineDataset(file_path).skip(1).map(decode_csv)

  if perform_shuffle:
    dataset = dataset.shuffle(buffer_size=256)

  dataset = dataset.repeat(repeat_count)
  dataset = dataset.batch(32)
  iterator = dataset.make_one_shot_iterator()

  batch_features, batch_labels = iterator.get_next()

  return batch_features, batch_labels

feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 10],
    n_classes=3)

classifier.train(
    input_fn=lambda: my_input_fn("../iris_data/iris_training.csv", True, 10))

test_results = classifier.evaluate(
    input_fn=lambda: my_input_fn("../iris_data/iris_test.csv"))

print("\nTest accuracy: %g %%" % (test_results["accuracy"] * 100))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

if __name__ == '__main__':
  tf.app.run(main)

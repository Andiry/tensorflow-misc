import time
import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import mnist_lenet_inference
import mnist_lenet_train

EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [mnist.test.num_examples,
                                        mnist_lenet_inference.IMAGE_SIZE,
                                        mnist_lenet_inference.IMAGE_SIZE,
                                        mnist_lenet_inference.NUM_CHANNELS],
                           name = 'x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_lenet_inference.OUTPUT_NODE], name = 'y-input')

        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        global_step = tf.Variable(0, trainable = False)

        regularizer = tf.contrib.layers.l2_regularizer(mnist_lenet_train.REGULARAZTION_RATE)
        y = mnist_lenet_inference.inference(x, False, regularizer)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variables_averages = tf.train.ExponentialMovingAverage(
            mnist_lenet_train.MOVING_AVERAGE_DECAY)

        variables_to_restore = variables_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        n = math.ceil(mnist.test.num_examples / mnist.test.num_examples)

        for i in range(n):
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_lenet_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    xs, ys = mnist.test.next_batch(mnist.test.num_examples)
                    reshaped_xs = np.reshape(xs, (
                        mnist.test.num_examples,
                        mnist_lenet_inference.IMAGE_SIZE,
                        mnist_lenet_inference.IMAGE_SIZE,
                        mnist_lenet_inference.NUM_CHANNELS))

                    accuracy_score = sess.run(accuracy,
                                              feed_dict = {x: reshaped_xs, y_: ys})

                    print("After %s steps, validation accuracy = %g" %
                          (global_step, accuracy_score))

                else:
                    print("No checkpoint file found")
                    return

                time.sleep(EVAL_INTERVAL_SECS)

def main(argv = None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot = True)

    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()

import tensorflow as tf

def fibonacci():
  x = tf.get_variable("x", initializer=0)
  y = tf.get_variable("y", initializer=1)

  f_n = x + y
  update_x = tf.assign(x, y)
  update_y = tf.assign(y, f_n)

  with tf.control_dependencies([update_x, update_y]):
    return tf.identity(f_n)


def run():
  tf.reset_default_graph()
  result = fibonacci()

  init = tf.global_variables_initializer()

  for _ in range(5):
    with tf.Session() as sess:
      sess.run(init)
      sequence = [sess.run(result) for _ in range(10)]

    print(sequence)


run()


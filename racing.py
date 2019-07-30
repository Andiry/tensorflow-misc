import tensorflow as tf

x = tf.get_variable("x", shape=(), initializer=tf.zeros_initializer())
assign_x = tf.assign(x, 10.0)

z = x + 1.0

init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  for _ in range(0, 1000):
    print(sess.run([assign_x, z]))

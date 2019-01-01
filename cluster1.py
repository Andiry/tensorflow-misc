import tensorflow as tf

c = tf.constant("Hello from server 2!")

cluster = tf.train.ClusterSpec(
    {"local": ["localhost:2222", "localhost:2223"]})

server = tf.train.Server(cluster, job_name='local', task_index=1)

sess = tf.Session(server.target, config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))

server.join()




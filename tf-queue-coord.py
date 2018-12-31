import tensorflow as tf

queue = tf.FIFOQueue(100, "float")
enqueue_op = queue.enqueue([tf.random_normal([1])])

qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)
tf.train.add_queue_runner(qr)
out_tensor = queue.dequeue()

options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

name = 'tf-queue-coord'
dump_dir = '/tmp/'
graph_path = ''.join([name, '.pbtxt'])
step_stats_path = ''.join([dump_dir, name, '-stepstats.pbtxt'])
tensorboard_path = ''.join([dump_dir, name])
meta_graph_path =  ''.join([dump_dir, name, '.meta'])


with tf.Session() as sess:
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  tf.train.write_graph(sess.graph, dump_dir, graph_path)
  writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
  writer.close()

  meta_graph_def = tf.train.export_meta_graph(filename=meta_graph_path)

  for _ in range(3):
    o = sess.run(out_tensor, options=options, run_metadata=run_metadata)[0]
    print(o)

  coord.request_stop()
  coord.join(threads)

  open(step_stats_path, 'w').write(str(run_metadata.step_stats))


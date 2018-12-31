import tensorflow as tf

q = tf.FIFOQueue(2, "int32")

init = q.enqueue_many(([0, 10],))

x = q.dequeue()

y = x + 1

q_inc = q.enqueue([y])

options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

name = 'tf-queue'
dump_dir = '/tmp/'
graph_path = ''.join([name, '.pbtxt'])
step_stats_path = ''.join([dump_dir, name, '-stepstats.pbtxt'])
tensorboard_path = ''.join([dump_dir, name])
meta_graph_path =  ''.join([dump_dir, name, '.meta'])

with tf.Session() as sess:
  init.run()
  for i in range(10):
    v, _ = sess.run([x, q_inc], options=options, run_metadata=run_metadata)
    print(v)

  tf.train.write_graph(sess.graph, dump_dir, graph_path)
  open(step_stats_path, 'w').write(str(run_metadata.step_stats))
  writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
  writer.close()

meta_graph_def = tf.train.export_meta_graph(filename=meta_graph_path)


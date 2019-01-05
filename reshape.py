import tensorflow as tf

x = tf.ones([5, 4])
y = tf.reshape(x, [-1, 10])

init = tf.global_variables_initializer()

options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

name = 'reshape'
dump_dir = '/tmp/'
graph_path = ''.join([name, '.pbtxt'])
step_stats_path = ''.join([dump_dir, name, '-stepstats.pbtxt'])
tensorboard_path = ''.join([dump_dir, name])
meta_graph_path =  ''.join([dump_dir, name, '.meta'])

with tf.Session() as sess:
    init.run()
    print(sess.run(y, options=options, run_metadata=run_metadata))
    tf.train.write_graph(sess.graph, dump_dir, graph_path)
    open(step_stats_path, 'w').write(str(run_metadata.step_stats))
    writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
    writer.add_run_metadata(run_metadata, 'step001')
    writer.close()

meta_graph_def = tf.train.export_meta_graph(filename=meta_graph_path)


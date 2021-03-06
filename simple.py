import tensorflow as tf

x = tf.Variable(3, name = 'x')
y = tf.Variable(4, name = 'y')

f = x * x * y + y + 2

init = tf.global_variables_initializer()

options = tf.RunOptions()
options.output_partition_graphs = True
options.trace_level = tf.RunOptions.FULL_TRACE
run_metadata = tf.RunMetadata()

name = 'simple'
dump_dir = '/tmp/'
graph_path = ''.join([name, '.pbtxt'])
step_stats_path = ''.join([dump_dir, name, '-stepstats.pbtxt'])
tensorboard_path = ''.join([dump_dir, name])
meta_graph_path =  ''.join([dump_dir, name, '.meta'])

with tf.Session() as sess:
    init.run()
    print(sess.run(f, options=options, run_metadata=run_metadata))
    tf.train.write_graph(sess.graph, dump_dir, graph_path)
    open(step_stats_path, 'w').write(str(run_metadata.step_stats))
    writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
    writer.add_run_metadata(run_metadata, 'step001')
    writer.close()

meta_graph_def = tf.train.export_meta_graph(filename=meta_graph_path)


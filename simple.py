import tensorflow as tf

x = tf.Variable(3, name = 'x')
y = tf.Variable(4, name = 'y')

f = x * x * y + y + 2

init = tf.global_variables_initializer()

options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

with tf.Session() as sess:
    init.run()
    print(sess.run(f, options=options, run_metadata=run_metadata))
    tf.train.write_graph(sess.graph, '/tmp/', 'simple.pbtxt')
    open('/tmp/simple-stepstats.pbtxt', 'w').write(str(run_metadata.step_stats))

meta_graph_def = tf.train.export_meta_graph(filename='/tmp/simple.meta')


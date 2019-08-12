import tensorflow as tf

with tf.device('/gpu:0'):
  a = tf.constant([1.5, 1.2], shape = [1,1,1,2], name = 'a')
  a1 = tf.constant([[2.0], [1.0]], shape = [1,1,2,1], name = 'b')

  min_a = tf.reduce_min(a)
  max_a = tf.reduce_max(a)
  min_a1 = tf.reduce_min(a1)
  max_a1 = tf.reduce_max(a1)

  x = tf.quantization.quantize(a, min_a, max_a, tf.quint8)
  x1 = tf.quantization.quantize(a1, min_a1, max_a1, tf.quint8)
  y1 = tf.nn.quantized_conv2d(x.output,x1.output,x.output_min,x.output_max,x1.output_min,x1.output_max,[1,1,1,1],"SAME")

options = tf.RunOptions()
options.output_partition_graphs = True
options.trace_level = tf.RunOptions.FULL_TRACE
run_metadata = tf.RunMetadata()

name = 'quantized_conv2d'
dump_dir = '/tmp/'
graph_path = ''.join([name, '.pbtxt'])
step_stats_path = ''.join([dump_dir, name, '-stepstats.pbtxt'])
tensorboard_path = ''.join([dump_dir, name])
meta_graph_path =  ''.join([dump_dir, name, '.meta'])

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=True)) as sess:
    print(sess.run(y1, options=options, run_metadata=run_metadata))
    tf.train.write_graph(sess.graph, dump_dir, graph_path)
    open(step_stats_path, 'w').write(str(run_metadata.step_stats))
    writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
    writer.add_run_metadata(run_metadata, 'step001')
    writer.close()

meta_graph_def = tf.train.export_meta_graph(filename=meta_graph_path)


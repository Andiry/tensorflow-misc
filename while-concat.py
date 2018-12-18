import tensorflow as tf

# From virtual_scheduler_test::CreateGrapplerItemWithLoop()

with tf.Graph().as_default():
  i0 = tf.constant(0)
  m0 = tf.ones([2, 2])
  c = lambda i, m: i < 10
  b = lambda i, m: [i + 1, tf.concat([m, m], axis = 0)]
  r = tf.while_loop(
      c, b, loop_vars=[i0, m0],
      shape_invariants=[i0.get_shape(), tf.TensorShape([None, 2])])

  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()

  with tf.Session() as sess:
    print(sess.run(r, options=options, run_metadata=run_metadata))
    tf.train.write_graph(sess.graph, '/tmp/', 'while-concat.pbtxt')
    open('/tmp/while-concat-stepstats.pbtxt', 'w').write(str(run_metadata.step_stats))

    meta_graph_def = tf.train.export_meta_graph(filename='/tmp/while-concat.meta')



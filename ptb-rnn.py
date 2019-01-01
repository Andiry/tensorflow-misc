import tensorflow as tf
import numpy as np
import reader

DATA_PATH = '/data/ptb/data'
HIDDEN_SIZE = 200
NUM_LAYERS = 2
VOCAB_SIZE = 10000

LEARNING_RATE = 1.0
TRAIN_BATCH_SIZE = 20
TRAIN_NUM_STEP = 35

EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1
NUM_EPOCH = 2
KEEP_PROB = 0.5

MAX_GRAD_NORM = 5

class PTBModel(object):
  def __init__(self, is_training, batch_size, num_steps):
    self.batch_size = batch_size
    self.num_steps = num_steps
    self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
    if is_training:
      lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,
                                                output_keep_prob = KEEP_PROB)
      pass
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * NUM_LAYERS)

    self.initial_state = cell.zero_state(batch_size, tf.float32)
    embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])

    inputs = tf.nn.embedding_lookup(embedding, self.input_data)

    if is_training:
      inputs = tf.nn.dropout(inputs, KEEP_PROB)
      pass

    outputs = []
    state = self.initial_state
    with tf.variable_scope("RNN"):
      for step in range(num_steps):
        if step > 0:
          tf.get_variable_scope().reuse_variables()
        cell_output, state = cell(inputs[:, step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])

    weight = tf.get_variable('weight', [HIDDEN_SIZE, VOCAB_SIZE])
    bias = tf.get_variable('bias', [VOCAB_SIZE])
    logits = tf.matmul(output, weight) + bias

    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self.targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype = tf.float32)])

    self.cost = tf.reduce_sum(loss) / batch_size
    self.final_state = state

    if not is_training:
      return

    trainable_variables = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)

    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    self.train_op = optimizer.apply_gradients(
        zip(grads, trainable_variables))


def run_epoch(session, model, data_queue, train_op, output_log, epoch_size,
              run_metadata):
  total_costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

  for step in range(epoch_size):
    feed_dict = {}
    x, y = session.run(data_queue)
    feed_dict[model.input_data] = x
    feed_dict[model.targets] = y

    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    cost, state, _ = session.run(
        [model.cost, model.final_state, train_op],
        feed_dict=feed_dict,
        options=options,
        run_metadata=run_metadata)

    total_costs += cost
    iters += model.num_steps

    if output_log and step % 100 == 0:
      print("After %d steps, perplexity %.3f" % (
          step, np.exp(total_costs / iters)))

  return np.exp(total_costs / iters)


def main(_):
  train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)

  train_data_len = len(train_data)
  train_batch_len = train_data_len // TRAIN_BATCH_SIZE  # batch的个数
  train_epoch_size = (train_batch_len - 1) // TRAIN_NUM_STEP  # 该epoch的训练次数

  valid_data_len = len(valid_data)
  valid_batch_len = valid_data_len // EVAL_BATCH_SIZE
  valid_epoch_size = (valid_batch_len - 1) // EVAL_NUM_STEP

  test_data_len = len(test_data)
  test_batch_len = test_data_len // EVAL_BATCH_SIZE
  test_epoch_size = (test_batch_len - 1) // EVAL_NUM_STEP

  initializer = tf.random_uniform_initializer(-0.05, 0.05)

  with tf.variable_scope("language_model",
                         reuse=None, initializer=initializer):
    train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

  with tf.variable_scope("language_model",
                         reuse=True, initializer=initializer):
    eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

  train_queue = reader.ptb_producer(train_data, train_model.batch_size,
                                    train_model.num_steps)
  valid_queue = reader.ptb_producer(valid_data, eval_model.batch_size,
                                    eval_model.num_steps)
  test_queue = reader.ptb_producer(test_data, eval_model.batch_size,
                                   eval_model.num_steps)

  name = 'ptb-rnn'
  dump_dir = '/tmp/'
  graph_path = ''.join([name, '.pbtxt'])
  step_stats_path = ''.join([dump_dir, name, '-stepstats.pbtxt'])
  tensorboard_path = ''.join([dump_dir, name])
  meta_graph_path =  ''.join([dump_dir, name, '.meta'])
  run_metadata = tf.RunMetadata()

  with tf.Session() as sess:
    tf.global_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(NUM_EPOCH):
      print("Interation %d" % (i + 1))

      print("Training")
      run_epoch(sess, train_model, train_queue, train_model.train_op,
                True, train_epoch_size, run_metadata)

      tf.train.write_graph(sess.graph, dump_dir, graph_path)
      open(step_stats_path, 'w').write(str(run_metadata.step_stats))
      writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
      writer.close()
      meta_graph_def = tf.train.export_meta_graph(filename=meta_graph_path)

      print("Evaluating")
      valid_perplexity = run_epoch(
          sess, eval_model, valid_queue, tf.no_op(), True, valid_epoch_size,
          run_metadata)
      print("Epoch: %d, Validation perplexity %.3f" %(
          i + 1, valid_perplexity))

      print("Testing")
      test_perplexity = run_epoch(
          sess, eval_model, test_queue, tf.no_op(), True, test_epoch_size,
          run_metadata)
      print("Test perplexity %.3f" % test_perplexity)

      coord.request_stop()
      coord.join(threads)


if __name__ == "__main__":
  tf.app.run()

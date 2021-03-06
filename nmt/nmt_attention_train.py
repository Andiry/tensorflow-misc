"""TODO(andiryxu): DO NOT SUBMIT without one-line documentation for nmt_train.

TODO(andiryxu): DO NOT SUBMIT without a detailed description of nmt_train.
"""
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nmt_data import *

SRC_TRAIN_DATA = "/data/en-zh/train.tags.en-zh.en.deletehtml.segment.id"
TRG_TRAIN_DATA = "/data/en-zh/train.tags.en-zh.zh.deletehtml.segment.id"

CHECKPOINT_PATH = "./checkpoint/seq2seq_attention_ckpt"
HIDDEN_SIZE = 1024
NUM_LAYERS = 2
SRC_VOCAB_SIZE = 10000
TRG_VOCAB_SIZE = 4000
BATCH_SIZE = 100
NUM_EPOCH = 30
KEEP_PROB = 0.8
MAX_GRAD_NORM = 5
SHARE_EMB_AND_SOFTMAX = True

class NMTModel(object):
  def __init__(self):
    self.enc_cell_fw = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)
    self.enc_cell_bw = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)
    self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)
         for _ in range(NUM_LAYERS)])

    self.src_embedding = tf.get_variable("src_emb",
                                         [SRC_VOCAB_SIZE, HIDDEN_SIZE])
    self.trg_embedding = tf.get_variable("trg_emb",
                                         [TRG_VOCAB_SIZE, HIDDEN_SIZE])

    if SHARE_EMB_AND_SOFTMAX:
      self.softmax_weight = tf.transpose(self.trg_embedding)
    else:
      self.softmax_weight = tf.get_variable("weight",
                                            [HIDDEN_SIZE, TRG_VOCAB_SIZE])
    self.softmax_bias = tf.get_variable("softmax_bias", [TRG_VOCAB_SIZE])

  def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
    batch_size = tf.shape(src_input)[0]

    src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)
    trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)

    src_emb = tf.nn.dropout(src_emb, KEEP_PROB)
    trg_emb = tf.nn.dropout(trg_emb, KEEP_PROB)

    with tf.variable_scope("encoder"):
      enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(
          self.enc_cell_fw, self.enc_cell_bw, src_emb, src_size, dtype=tf.float32)
      enc_outputs = tf.concat([enc_outputs[0], enc_outputs[1]], -1)

    with tf.variable_scope("decoder"):
      attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
          HIDDEN_SIZE, enc_outputs, memory_sequence_length=src_size)

      attention_cell = tf.contrib.seq2seq.AttentionWrapper(
          self.dec_cell, attention_mechanism, attention_layer_size=HIDDEN_SIZE)

      dec_outputs, _ = tf.nn.dynamic_rnn(attention_cell,
                                         trg_emb, trg_size, dtype=tf.float32)

    output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
    logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels = tf.reshape(trg_label, [-1]), logits = logits)

    label_weights = tf.sequence_mask(
        trg_size, maxlen=tf.shape(trg_label)[1], dtype=tf.float32)
    label_weights = tf.reshape(label_weights, [-1])
    cost = tf.reduce_sum(loss * label_weights)
    cost_per_token = cost / tf.reduce_sum(label_weights)

    trainable_variables = tf.trainable_variables()

    grads = tf.gradients(cost / tf.to_float(batch_size),
                         trainable_variables)

    grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
    train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
    return cost_per_token, train_op

def run_epoch(session, cost_op, train_op, saver, step):
  while True:
    try:
      cost, _ = session.run([cost_op, train_op])
      if step % 10 == 0:
        print("After %d steps, per token cost %.3f" % (step, cost))
      if step % 200 == 0:
        saver.save(session, CHECKPOINT_PATH, global_step=step)
      step += 1
    except tf.errors.OutOfRangeError:
      break
  return step


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  initializer = tf.random_uniform_initializer(-0.05, 0.05)

  with tf.variable_scope("nmt_model", reuse=None,
                         initializer=initializer):
    train_model = NMTModel()

  data = MakeSrcTrgDataset(SRC_TRAIN_DATA, TRG_TRAIN_DATA, BATCH_SIZE)
  iterator = data.make_initializable_iterator()
  (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()

  cost_op, train_op = train_model.forward(src, src_size, trg_input,
                                          trg_label, trg_size)

  saver = tf.train.Saver()
  step = 0

  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(NUM_EPOCH):
      print("In iteration %d:" % (i + 1))
      sess.run(iterator.initializer)
      step = run_epoch(sess, cost_op, train_op, saver, step)

if __name__ == '__main__':
  tf.app.run(main)

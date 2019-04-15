"""TODO(andiryxu): DO NOT SUBMIT without one-line documentation for prb-vocab.

TODO(andiryxu): DO NOT SUBMIT without a detailed description of prb-vocab.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import sys
from operator import itemgetter

RAW_DIR = "/data/ptb/data/"
VOCAB_OUTPUT = "ptb.vocab"

def get_id(word):
  return word_to_id[word] if word in word_to_id else word_to_id['<unk>']

for OUTPUT_DATA in ['ptb.train', 'ptb.valid', 'ptb.test']:
  with codecs.open(VOCAB_OUTPUT, 'r', 'utf-8') as f_vocab:
    vocab = [w.strip() for w in f_vocab.readlines()]
  word_to_id = {k:v for (k,v) in zip(vocab, range(len(vocab)))}

  RAW_DATA = RAW_DIR + OUTPUT_DATA + '.txt'
  fin = codecs.open(RAW_DATA, 'r', 'utf-8')
  fout = codecs.open(OUTPUT_DATA, 'w', 'utf-8')

  for line in fin:
    words = line.strip().split() + ['<eos>']
    out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
    fout.write(out_line)

  fin.close()
  fout.close()


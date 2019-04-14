"""TODO(andiryxu): DO NOT SUBMIT without one-line documentation for prb-vocab.

TODO(andiryxu): DO NOT SUBMIT without a detailed description of prb-vocab.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
from operator import itemgetter

RAW_DATA = "/data/ptb/data/ptb.train.txt"
VOCAB_OUTPUT = "ptb.vocab"

counter = collections.Counter()
with codecs.open(RAW_DATA, "r", "utf-8") as f:
  for line in f:
    for word in line.strip().split():
      counter[word] += 1

sorted_word_to_cnt = sorted(counter.items(),
                            key = itemgetter(1),
                            reverse = True)
sorted_words = [x[0] for x in sorted_word_to_cnt]

sorted_words = ['<eos>'] + sorted_words

with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as file_output:
  for word in sorted_words:
    file_output.write(word + '\n')

import sys
OUTPUT_DATA = 'ptb.train'

with codecs.open(VOCAB_OUTPUT, 'r', 'utf-8') as f_vocab:
  vocab = [w.strip() for w in f_vocab.readlines()]
word_to_id = {k:v for (k,v) in zip(vocab, range(len(vocab)))}

def get_id(word):
  return word_to_id[word] if word in word_to_id else word_to_id['<unk>']

fin = codecs.open(RAW_DATA, 'r', 'utf-8')
fout = codecs.open(OUTPUT_DATA, 'w', 'utf-8')

for line in fin:
  words = line.strip().split() + ['<eos>']
  out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
  fout.write(out_line)

fin.close()
fout.close()


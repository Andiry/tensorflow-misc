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

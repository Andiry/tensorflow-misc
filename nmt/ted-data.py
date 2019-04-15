"""TODO(andiryxu): DO NOT SUBMIT without one-line documentation for ted-data.

TODO(andiryxu): DO NOT SUBMIT without a detailed description of ted-data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections
from operator import itemgetter
from stanfordcorenlp import StanfordCoreNLP
import tqdm


def deletehtml(filename1, filename2):
  f1 = open(filename1, 'r')
  f2 = open(filename2, 'r')

  data1 = f1.readlines()
  data2 = f2.readlines()

  assert len(data1) == len(data2)
  fw1 = open(filename1 + '.deletehtml', 'w')
  fw2 = open(filename2 + '.deletehtml', 'w')

  print('deletehtml...')

  for line1, line2 in tqdm.tqdm(zip(data1, data2)):
    line1 = line1.strip()
    line2 = line2.strip()
    if line1 and line2:
      if '<' not in line1 and '>' not in line1 and '<' not in line2 and '>' not in line2:
        fw1.write(line1 + '\n')
        fw2.write(line2 + '\n')

  fw1.close()
  f1.close()
  fw2.close()
  f2.close()

  return filename1 + '.deletehtml', filename2 + '.deletehtml'

def segment_sentence(filename, vocab_size, lang):
  nlp = StanfordCoreNLP("/data/stanford-corenlp-full-2018-10-05", lang=lang)
  with open(filename, 'r') as f:
    data = f.readlines()
    counter = collections.Counter()
    f1 = open(filename + '.segment', 'w')
    print("Segmenting...")
    for line in tqdm.tqdm(data):
      line = line.strip()
      word_list = nlp.word_tokenize(line.strip())
      sentence = ' '.join(word_list)
      f1.write(sentence.encode('utf-8') + '\n')
      for word in word_list:
        counter[word] += 1
    f1.close()
  nlp.close()

  sorted_word_to_cnt = sorted(counter.items(), key = itemgetter(1), reverse = True)
  sorted_words =  ["<unk>","<sos>","<eos>"] + [x[0] for x in sorted_word_to_cnt]

  if len(sorted_words) > vocab_size:
    sorted_words = sorted_words[:vocab_size]

  with open(filename + '.vocab', 'w') as fw:
    for word in sorted_words:
      fw.write(word.encode('utf-8') + '\n')

  return filename + '.segment'

def convert_to_id(filename, vocab_file):
  with open(vocab_file, 'r') as f:
    data = f.readlines()
    vocab = [w.strip() for w in data]
  word_to_id = {k:v for (k, v) in zip(vocab, range(len(vocab)))}

  with open(filename, 'r') as f:
    data = f.readlines()
    f1 = open(filename + '.id', 'w')
    print("Converting " + filename + ' to id...')
    for line in tqdm.tqdm(data):
      words = line.strip().split() + ['<eos>']
      ids = ' '.join([str(word_to_id[word])
                      if word in word_to_id else str(word_to_id['<unk>'])
                      for word in words])
      f1.write(ids + '\n')
    f1.close()

  return filename + '.id'

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  src = '/data/en-zh/train.tags.en-zh.en'
  trg = '/data/en-zh/train.tags.en-zh.zh'
  src_vocab_size = 10000
  trg_vocab_size = 4000

  src1, trg1 = deletehtml(src, trg)

  src2 = segment_sentence(src1, src_vocab_size, lang='en')
  trg2 = segment_sentence(trg1, trg_vocab_size, lang='zh')

  src3 = convert_to_id(src + '.deletehtml.segment', src + '.deletehtml.vocab')
  trg3 = convert_to_id(trg + '.deletehtml.segment', trg + '.deletehtml.vocab')

if __name__ == '__main__':
  tf.app.run(main)

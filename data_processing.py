#!/usr/bin/python
# -*- coding: utf-8 -*-


import os
import sys
import numpy as np
from collections import Counter
import tensorflow.contrib.keras as kr
import pandas as pd

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False

def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content

def open_file(filename, mode='r'):
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)

def read_vocab(vocab_dir):
    with open_file(vocab_dir) as fp:
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id

def read_file(filename):
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                temp = line.strip().split('\t')
                label = temp[0]
                content = temp[1]
                if content:
                    contents.append(content.split(' '))
                    labels.append(native_content(label))
                else:
                    print("failed")
            except:
                print("failed")
                pass
    f.close()
    return contents, labels

def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    words = ['<PAD>'] + list(words)
    f = open_file(vocab_dir, mode='w')
    f.write('\n'.join(words) + '\n')
    f.close

def process_file_words(filename, word_to_id, max_length=10000):
    contents, labels = read_file(filename)
    data_id = []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)

    return pd.DataFrame(x_pad), labels

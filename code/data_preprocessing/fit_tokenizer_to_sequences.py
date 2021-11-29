'''
Author: renjunxiang
Date: 2021-11-28 15:54:30
LastEditTime: 2021-11-29 15:58:19
LastEditors: czqmike
Description: 
FilePath: /DLPredictor/code/data_preprocessing/fit_tokenizer_to_sequences.py
'''

import pickle
import jieba
import json, os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

jieba.setLogLevel('WARN')

num_fact = 150000
num_words = 40000
maxlen = 400
cutlen = 10000
dataset_name = 'train'

tokenizer_fact = Tokenizer(num_words=num_words)

path_to_tokenizer = '../../weight/tokenizer/'
if not os.path.exists(path_to_tokenizer):
    os.makedirs(path_to_tokenizer)
path_to_fact_seq = '../../data/CAIL2018-Small/fact_seq/'
if not os.path.exists(path_to_fact_seq):
    os.makedirs(path_to_fact_seq)
path_to_pad_seq = '../../data/CAIL2018-Small/pad_seq/'
if not os.path.exists(path_to_pad_seq):
    os.makedirs(path_to_pad_seq)

## Train tokenizer
# for i in range(num_fact // cutlen):
#     print('start fact_cut_%d_%d' % (i * cutlen, i * cutlen + cutlen))
#     with open('../../data/CAIL2018-Small/data_cut/fact_cut_' + dataset_name + '_%d_%d_new.pkl' % (i * cutlen, i * cutlen + cutlen), mode='rb') as f:
#         fact_cut = pickle.load(f)
#     texts_cut_len = len(fact_cut)
#     n = 0
#     ## 分批训练
#     while n < texts_cut_len:
#         tokenizer_fact.fit_on_texts(texts=fact_cut[n:n + 10000])
#         n += 10000
#         if n < texts_cut_len:
#             print('tokenizer finish fit %d samples' % n)
#         else:
#             print('tokenizer finish fit %d samples' % texts_cut_len)
#     print('finish fact_cut_%d_%d' % (i * cutlen, i * cutlen + cutlen))

# with open(path_to_tokenizer + 'tokenizer_fact_%d.pkl' % (num_words), mode='wb') as f:
#     pickle.dump(tokenizer_fact, f)

## Use tokenizer
with open(path_to_tokenizer + 'tokenizer_fact_%d.pkl' % (num_words), mode='rb') as f:
    tokenizer_fact=pickle.load(f)
## texts_to_sequences
for i in range(num_fact // cutlen):
    print('start fact_cut_%d_%d' % (i * cutlen, i * cutlen + cutlen))
    with open('../../data/CAIL2018-Small/data_cut/fact_cut_' + dataset_name + '_%d_%d_new.pkl' % (i * cutlen, i * cutlen + cutlen), mode='rb') as f:
        fact_cut = pickle.load(f)
    # 分批执行 texts_to_sequences
    fact_seq = tokenizer_fact.texts_to_sequences(texts=fact_cut)
    with open(path_to_fact_seq + 'fact_seq_%d_%d.pkl' % (i * cutlen, i * cutlen + cutlen), mode='wb') as f:
        pickle.dump(fact_seq, f)
    print('finish fact_cut_%d_%d' % (i * cutlen, i * cutlen + cutlen))

## pad_sequences
for i in range(num_fact // cutlen):
    print('start fact_cut_%d_%d' % (i * cutlen, i * cutlen + cutlen))
    with open(path_to_fact_seq + 'fact_seq_%d_%d.pkl' % (i * cutlen, i * cutlen + cutlen), mode='rb') as f:
        fact_seq = pickle.load(f)
    texts_cut_len = len(fact_seq)
    n = 0
    fact_pad_seq = []
    # 分批执行pad_sequences
    while n < texts_cut_len:
        fact_pad_seq += list(pad_sequences(fact_seq[n:n + 20000], maxlen=maxlen,
                                           padding='post', value=0, dtype='int'))
        n += 20000
        if n < texts_cut_len:
            print('finish pad_sequences %d samples' % n)
        else:
            print('finish pad_sequences %d samples' % texts_cut_len)
    with open(path_to_pad_seq + 'fact_pad_seq_%d_%d_%d.pkl' % (maxlen, i * cutlen, i * cutlen + cutlen),
              mode='wb') as f:
        pickle.dump(fact_pad_seq, f)

# 汇总pad_sequences,5G,16G内存够用
fact_pad_seq = []
for i in range(num_fact // cutlen):
    print('start fact_cut_%d_%d' % (i * cutlen, i * cutlen + cutlen))
    with open(path_to_pad_seq + 'fact_pad_seq_%d_%d_%d.pkl' % (maxlen, i * cutlen, i * cutlen + cutlen),
              mode='rb') as f:
        fact_pad_seq += pickle.load(f)
fact_pad_seq = np.array(fact_pad_seq)
np.save(path_to_pad_seq + 'fact_pad_seq_%d_%d.npy' % (num_words, maxlen), fact_pad_seq)
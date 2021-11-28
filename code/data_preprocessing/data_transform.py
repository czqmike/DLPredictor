'''
Author: renjunxiang
Date: 2021-11-28 15:58:45
LastEditTime: 2021-11-28 20:29:10
LastEditors: czqmike
Description: 
FilePath: /DLPredictor/code/data_preprocessing/data_transform.py
'''
import json
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import jieba

jieba.setLogLevel('WARN')


class data_transform():
    def __init__(self):
        self.data_path = None
        self.data = None
        self.texts_cut = None
        self.tokenizer = None
        self.label_set = {}
        self.extraction = {}
        self.tokenizer_fact = None

    def read_data(self, path=None):
        '''
        读取json文件,必须readlines，否则中间有格式会报错
        :param path: 文件路径
        :return:json数据
        eg. data_valid = data_transform.read_data(path='../../data/data_valid.json')
        '''
        self.data_path = path
        f = open(path, 'r', encoding='utf8')
        data_raw = f.readlines()
        data = []
        for num, data_one in enumerate(data_raw):
            try:
                data.append(json.loads(data_one))
            except Exception as e:
                print('num: %d', '\n',
                      'error: %s', '\n',
                      'data: %s' % (num, e, data_one))
        self.data = data

    def extract_data(self, name='accusation'):
        '''
        提取需要的信息，以字典形式存储
        :param name: 提取内容
        :return: 事实描述,罪名,相关法条
        eg. data_valid_accusations = data_transform.extract_data(name='accusation')
        '''
        data = self.data
        if name == 'fact':
            extraction = list(map(lambda x: x['fact'], data))
        elif name in ['accusation', 'relevant_articles']:
            extraction = list(map(lambda x: x['meta'][name], data))
        elif name == 'imprisonment':
            extraction = []
            for i in data:
                if i['meta']['term_of_imprisonment']['death_penalty']:
                    extraction.append([500])
                elif i['meta']['term_of_imprisonment']['life_imprisonment']:
                    extraction.append([400])
                else:
                    extraction.append([i['meta']['term_of_imprisonment']['imprisonment']])
        self.extraction.update({name: extraction})

    def cut_texts(self, texts=None, need_cut=True, word_len=1, texts_cut_savepath=None):
        '''
        文本分词剔除停用词
        :param texts:文本列表
        :param need_cut:是否需要分词
        :param word_len:保留词语长度
        :param texts_cut_savepath:保存路径
        :return:
        '''
        if need_cut:
            if word_len > 1:
                texts_cut = [[word for word in jieba.lcut(one_text) if len(word) >= word_len] for one_text in texts]
            else:
                texts_cut = [jieba.lcut(one_text) for one_text in texts]
        else:
            if word_len > 1:
                texts_cut = [[word for word in one_text if len(word) >= word_len] for one_text in texts]
            else:
                texts_cut = texts

        if texts_cut_savepath is not None:
            with open(texts_cut_savepath, 'w') as f:
                json.dump(texts_cut, f)
        return texts_cut

    def text2seq(self, texts_cut=None, tokenizer_fact=None, num_words=2000, maxlen=30):
        '''
        文本转序列，训练集过大全部转换会内存溢出，每次放5000个样本
        :param texts_cut: 分词后的文本列表
        :param tokenizer:转换字典
        :param num_words:字典词数量
        :param maxlen:保留长度
        :return:向量列表
        eg. ata_transform.text2seq(texts_cut=train_fact_cut,num_words=2000, maxlen=500)
        '''
        texts_cut_len = len(texts_cut)

        if tokenizer_fact is None:
            tokenizer_fact = Tokenizer(num_words=num_words)
            if texts_cut_len > 10000:
                print('文本过多，分批转换')
            n = 0
            # 分批训练
            while n < texts_cut_len:
                tokenizer_fact.fit_on_texts(texts=texts_cut[n:n + 10000])
                n += 10000
                if n < texts_cut_len:
                    print('tokenizer finish fit %d samples' % n)
                else:
                    print('tokenizer finish fit %d samples' % texts_cut_len)
            self.tokenizer_fact = tokenizer_fact

        # 全部转为数字序列
        fact_seq = tokenizer_fact.texts_to_sequences(texts=texts_cut)
        print('finish texts to sequences')

        # 内存不够，删除
        del texts_cut

        n = 0
        fact_pad_seq = []
        # 分批执行pad_sequences
        while n < texts_cut_len:
            fact_pad_seq += list(pad_sequences(fact_seq[n:n + 10000], maxlen=maxlen,
                                               padding='post', value=0, dtype='int'))
            n += 10000
            if n < texts_cut_len:
                print('finish pad_sequences %d samples' % n)
            else:
                print('finish pad_sequences %d samples' % texts_cut_len)
        self.fact_pad_seq = fact_pad_seq

    def creat_label_set(self, name):
        '''
        获取标签集合，用于one-hot
        :param name: 待创建集合的标签名称
        :return:
        '''
        if name == 'accusation':
            name_f = 'accu'
        elif name == 'relevant_articles':
            name_f = 'law'
        with open('../../data/%s.txt' % name_f, encoding='utf-8') as f:
            label_set = f.readlines()
        label_set = [i[:-1] for i in label_set]
        self.label_set.update({name: np.array(label_set)})

    def creat_label(self, label, label_set):
        '''
        构建标签one-hot
        :param label: 原始标签
        :param label_set: 标签集合
        :return: 标签one-hot
        eg. creat_label(label=data_valid_accusations[12], label_set=accusations_set)
        '''
        label_str = [str(i) for i in label]
        label_zero = np.zeros(len(label_set))
        label_zero[np.in1d(label_set, label_str)] = 1
        return label_zero

    def creat_labels(self, label_set=None, labels=None, name='accusation'):
        '''
        调用creat_label遍历标签列表生成one-hot二维数组
        :param label_set: 标签集合,数组
        :param labels: 标签数据，二维列表，没有则调用extract_data函数提取
        :param name:
        :return:
        '''
        if label_set is None:
            label_set = self.label_set[name]
        if labels is None:
            labels = self.extraction[name]
        labels_one_hot = list(map(lambda x: self.creat_label(label=x, label_set=label_set), labels))
        return labels_one_hot

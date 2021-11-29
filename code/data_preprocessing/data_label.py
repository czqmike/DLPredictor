'''
Author: czqmike
Date: 2021-11-28 15:54:30
LastEditTime: 2021-11-28 20:33:03
LastEditors: czqmike
Description: Generate label files (.npy) from json files. 
FilePath: /DLPredictor/code/data_preprocessing/data_label.py
'''
from data_transform import data_transform
import jieba
import numpy as np
import os

jieba.setLogLevel('WARN')

num_words = 40000
maxlen = 400
data_path = '../../data/CAIL2018-Small/'
file_names = ['data_test.json', 'data_train.json', 'data_valid.json']
label_types = ['accusation']
########################################################################################
for file_name in file_names:
	dataTransformer = None
	dataTransformer = data_transform()
	dataTransformer.read_data(data_path + file_name)
	suffix = file_name.split('.')[0].split('_')[1] # test, train or valid
	for label_type in label_types:
		dataTransformer.extract_data(name=label_type)
		dataTransformer.creat_label_set(name=label_type)
		labels = dataTransformer.creat_labels(name=label_type)
		if not os.path.exists(data_path + 'labels'):
			os.makedirs(data_path + 'labels')
		np.save(data_path + f'labels/label_{suffix}_{label_type}.npy', labels)
'''
# big数据集处理
data_transform_big = data_transform()

# 读取json文件,1710857行
data_transform_big.read_data(path='./data/cail2018_big.json')

# 创建数据one-hot标签
data_transform_big.extract_data(name='accusation')
# big_accusations = data_transform_big.extraction['accusation']
data_transform_big.creat_label_set(name='accusation')
big_labels = data_transform_big.creat_labels(name='accusation')
np.save(data_path + 'labels/labels_accusation.npy', big_labels)

# big数据集处理
data_transform_big = data_transform()

# 读取json文件,1710857行
data_transform_big.read_data(path='./data/cail2018_big.json')
data_transform_big.extract_data(name='relevant_articles')
big_relevant_articless = data_transform_big.extraction['relevant_articles']
data_transform_big.creat_label_set(name='relevant_articles')
big_labels = data_transform_big.creat_labels(name='relevant_articles')
np.save('./data/labels/big_labels_relevant_articles.npy', big_labels)

# big数据集处理
data_transform_big = data_transform()

# 读取json文件,1710857行
data_transform_big.read_data(path='./data/cail2018_big.json')

# 创建刑期连续变量
data_transform_big.extract_data(name='imprisonment')
big_imprisonments = data_transform_big.extraction['imprisonment']
np.save('./data/labels/big_labels_imprisonments.npy', big_imprisonments)

# big数据集处理
data_transform_big = data_transform()

# 读取json文件,1710857行
data_transform_big.read_data(path='./data/cail2018_big.json')

# 创建刑期离散变量
data_transform_big.extract_data(name='imprisonment')
big_imprisonments = data_transform_big.extraction['imprisonment']
data_transform_big.creat_label_set(name='imprisonment')
big_labels = data_transform_big.creat_labels(name='imprisonment')
np.save('./data/labels/big_labels_imprisonments_discrete.npy', big_labels)
'''
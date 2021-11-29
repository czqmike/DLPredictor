'''
Author: renjunxiang
Date: 2021-11-28 15:54:30
LastEditTime: 2021-11-29 15:27:37
LastEditors: czqmike
Description: 
FilePath: /DLPredictor/code/data_preprocessing/data_cut.py
'''
from data_transform import data_transform
import json, os
import pickle
import jieba
import numpy as np

jieba.setLogLevel('WARN')

cutlen = 10000
########################################################################################
# big数据集处理
data_transform_big = data_transform()

path_to_folder = '../../data/CAIL2018-Small/'
if not os.path.exists(path_to_folder):
    os.makedirs(path_to_folder)
# 读取json文件
data_transform_big.read_data(path=path_to_folder + 'data_train.json')

# 提取需要信息
data_transform_big.extract_data(name='fact') # list[str], shape=(num_fact, len_fact)
data_size = len(data_transform_big.extraction['fact'])

# 分词并保存原始分词结果，词语长度后期可以再改
for i in range(data_size // cutlen):
    texts = data_transform_big.extraction['fact'][i*cutlen:(i*cutlen + cutlen)]
    ## fact_cut: [[cut_str1, cut_str2, cut_str3...]], shape=(fact_num, fact_len)
    fact_cut = data_transform_big.cut_texts(texts=texts, word_len=1,
                                                need_cut=True)
    with open(path_to_folder + 'data_cut/fact_cut_train_%d_%d.pkl' % (i*cutlen, i*cutlen + cutlen), mode='wb') as f:
        pickle.dump(fact_cut, f)
    '''['昌宁县', '人民检察院', '指控', '，', '2014', '年', '4', '月', '19', '日', '下午', '16', '时许', 
    '，', '被告人', '段', '某驾', '拖车', '经过', '鸡飞乡', '澡塘', '街子', '，', '时逢', '堵车', '，', '段', 
    '某', '将', '车', '停', '在', '“', '冰凉', '一夏', '”', '冷饮店', '门口', '，', '被害人', '王某', '的', '侄子', 
    '王', '2', '某', '示意', '段', '某', '靠边', '未果', '，', '后', '上前', '敲打', '车门', '让', '段', '某', 
    '离开', '，', '段', '某', '遂', '驾车', '离开', '，', '但', '对此', '心', '生', '怨愤', '。', '同年', '4', '月',
     '21', '日', '22', '时许', '，', '被告人', '段', '某', '酒后', '与其', '妻子', '王', '1', '某', '一起', '准备', 
     '回家', '，', '走到', '鸡', '飞乡', '澡塘', '街', '富达', '通讯', '手机', '店门口', '时', '停下', '，', '段', 
     '某', '进入', '手机店', '内', '对', '被害人', '王某', '进行', '吼', '骂', '，', '紧接着', '从', '手机店', 
     '出来', '拿', '得', '一个', '石头', '又', '冲进', '手机店', '内朝王', '某', '头部', '打', '去', '，', '致王', 
     '某右', '额部', '粉碎性', '骨折', '、', '右', '眼眶', '骨', '骨折', '。', '经', '鉴定', '，', '被害人', '王某', 
     '此次', '损伤', '程度', '为', '轻伤', '一级', '。']'''
    print('finish fact_cut_%d_%d' % (i*cutlen, i*cutlen + cutlen))

for i in range(data_size // cutlen):
    print('start fact_cut_%d_%d' % (i*cutlen, i*cutlen + cutlen))
    with open(path_to_folder + 'data_cut/fact_cut_train_%d_%d.pkl' % (i*cutlen, i*cutlen + cutlen), mode='rb') as f:
        fact_cut = pickle.load(f)
    data_transform_big = data_transform()
    fact_cut_new = data_transform_big.cut_texts(texts=fact_cut,
                                                    word_len=2,
                                                    need_cut=False)
    '''['昌宁县', '人民检察院', '指控', '2014', '19', '下午', '16', '时许', '被告人', '某驾', '拖车', '经过', 
    '鸡飞乡', '澡塘', '街子', '时逢', '堵车', '冰凉', '一夏', '冷饮店', '门口', '被害人', '王某', '侄子', '示意', 
    '靠边', '未果', '上前', '敲打', '车门', '离开', '驾车', '离开', '对此', '怨愤', '同年', '21', '22', '时许', 
    '被告人', '酒后', '与其', '妻子', '一起', '准备', '回家', '走到', '飞乡', '澡塘', '富达', '通讯', '手机', 
    '店门口', '停下', '进入', '手机店', '被害人', '王某', '进行', '紧接着', '手机店', '出来', '一个', '石头', 
    '冲进', '手机店', '内朝王', '头部', '致王', '某右', '额部', '粉碎性', '骨折', '眼眶', '骨折', '鉴定', '被害人', 
    '王某', '此次', '损伤', '程度', '轻伤', '一级']
    '''
    with open(path_to_folder + 'data_cut/fact_cut_%d_%d_new.pkl' % (i*cutlen, i*cutlen + cutlen), mode='wb') as f:
        pickle.dump(fact_cut_new, f)
    print('finish fact_cut_%d_%d' % (i*cutlen, i*cutlen + cutlen))

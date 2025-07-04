'''
Author: czqmike
Date: 2021-11-28 16:43:54
LastEditTime: 2021-11-29 17:01:23
LastEditors: czqmike
Description: 
FilePath: /DLPredictor/code/main.py
'''
import numpy as np
np.random.seed(1006) # for reproducibility
from model.model_CNN import model_CNN_accusation
from sklearn.model_selection import train_test_split
import os, argparse, time
from evaluate import predict2both, predict2half, predict2top, f1_avg
from keras.models import Model
import pandas as pd


parser = argparse.ArgumentParser(description='Model Controller')
parser.add_argument('--train', default=True, type=bool, help='Whether to train the model')
parser.add_argument('--model_name', default='CNN', type=str, help='model name')
parser.add_argument('--infer_type', default='accusation', type=str, 
                    help='infer results type: accusation | imprisonment')
args = parser.parse_args()

## rel
#rel = "comparison"
#rel = "temporal"
#rel = "expansion"
rel = "contingency"

MAX_LEN = 400
NUM_WORDS = 40000
CUTLEN = 10000
OUTPUT_DIM = 202
path_to_dataset = '../data/CAIL2018-Small/'

if __name__ == '__main__':
    # fact数据集
    fact = np.load(path_to_dataset + f'pad_seq/fact_pad_seq_{MAX_LEN}_{0}_{CUTLEN}.pkl', allow_pickle=True)
    fact_train, fact_test = train_test_split(fact, test_size=0.05, random_state=1)
    print("type fact_train: ", type(fact_train))
    print("len fact_train: ", len(fact_train))
    fact_train = np.array(fact_train)
    fact_test = np.array(fact_test)
    del fact

    # 标签数据集
    labels = np.load(path_to_dataset + 'labels/label_train_accusation.npy')
    labels = labels[:CUTLEN]
    labels_train, labels_test = train_test_split(labels, test_size=0.05, random_state=1)
    print("type labels_train: ", type(labels_train))
    print("len labels_train: ", len(labels_train))
    labels_train = np.array(labels_train)
    labels_test = np.array(labels_test)
    del labels

    model = model_CNN_accusation(num_words=NUM_WORDS, maxlen=MAX_LEN, output_dim=OUTPUT_DIM,
                                       kernel_size=4)

    if args.train:
        BATCH_SIZE = 8
        EPOCH = 20
        n_start = 1
        n_end = 21
        score_list1 = []
        score_list2 = []

        for i in range(n_start, n_end):
            model.fit(x=fact_train, y=labels_train, batch_size=BATCH_SIZE, epochs=1, verbose=1)

            path_to_weight = f'../weight/accusation/{NUM_WORDS}_{MAX_LEN}/'
            if not os.path.exists(path_to_weight):
                os.makedirs(path_to_weight)
            model.save_weights(path_to_weight + 'CNN_epochs_%d.weight' % i)

            y = model.predict(fact_test[:])
            y1 = predict2top(y)
            y2 = predict2half(y)
            y3 = predict2both(y)

            print('%s accu:' % i)
            # 只取最高置信度的准确率
            s1 = [(labels_test[i] == y1[i]).min() for i in range(len(y1))]
            print(sum(s1) / len(s1))
            # 只取置信度大于0.5的准确率
            s2 = [(labels_test[i] == y2[i]).min() for i in range(len(y1))]
            print(sum(s2) / len(s2))
            # 结合前两个
            s3 = [(labels_test[i] == y3[i]).min() for i in range(len(y1))]
            print(sum(s3) / len(s3))

            print('%s f1:' % i)
            # 只取最高置信度的准确率
            s4 = f1_avg(y_pred=y1, y_true=labels_test)
            print(s4)
            # 只取置信度大于0.5的准确率
            s5 = f1_avg(y_pred=y2, y_true=labels_test)
            print(s5)
            # 结合前两个
            s6 = f1_avg(y_pred=y3, y_true=labels_test)
            print(s6)

            score_list1.append([i,
                                sum(s1) / len(s1),
                                sum(s2) / len(s2),
                                sum(s3) / len(s3)])
            score_list2.append([i, s4, s5, s6])
            print(pd.DataFrame(score_list1))
            print(pd.DataFrame(score_list2))

        print('end', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        print('#####################\n')
                
    # else:
    #     model.load_weights(save_weights_path)
    #     test_result_path = 'results/' + dataset + '/test_result.json'
    #     precision, recall, f1_score = metric(
    #         subject_model, object_model, test_data, id2rel, tokenizer, isExactMatch, test_result_path, get_weights)
    #     print(f'p: {precision:.8f}\t r: {recall:.8f}\t f1: {f1_score:.8f}')

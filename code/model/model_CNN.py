'''
Author: czqmike
Date: 2021-11-28 17:26:10
LastEditTime: 2021-11-28 18:18:55
LastEditors: czqmike
Description: 
FilePath: /DLPredictor/code/model/model_CNN.py
'''
import keras
from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPool1D
from keras.layers import Dense, Dropout, BatchNormalization
from numpy import ndarray


def model_CNN_accusation(num_words: int, maxlen: int, output_dim: int, kernel_size: int, 
						 fact_train: ndarray, label_train: ndarray)->keras.models.Model:
	data_input = Input(shape=[fact_train.shape[1]])
	word_vec = Embedding(input_dim=num_words + 1,
						input_length=maxlen,
						output_dim=output_dim,
						mask_zero=0,
						name='Embedding')(data_input)
	x = word_vec
	x = Conv1D(filters=512, kernel_size=[kernel_size], strides=1, padding='same', activation='relu')(x)
	x = GlobalMaxPool1D()(x)
	x = BatchNormalization()(x)
	x = Dense(1000, activation="relu")(x)
	x = Dropout(0.2)(x)
	x = Dense(label_train.shape[1], activation="sigmoid")(x)
	model = Model(inputs=data_input, outputs=x)
	model.compile(loss='binary_crossentropy',
				optimizer='adam',
				metrics=['accuracy'])
	model.summary()

	return model
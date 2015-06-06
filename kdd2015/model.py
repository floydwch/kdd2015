# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU


def build_model():
    max_features = 8

    model = Sequential()

    model.add(Embedding(8, 8))
    # model.add(LSTM(
    #     8, 720,
    #     # activation='sigmoid',
    #     # inner_activation='hard_sigmoid'
    #     return_sequences=True
    # ))
    # model.add(TimeDistributedDense(720, 240))
    model.add(LSTM(
        8, 1,
        activation='sigmoid',
        inner_activation='hard_sigmoid'
        # return_sequences=True
    ))
    # model.add(Dropout(0.2))
    # model.add(TimeDistributedDense(8, 1))
    model.add(Activation('sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        class_mode="binary"
    )

    return model

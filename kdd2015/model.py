# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU


def build_model():
    # max_features = 8

    model = Sequential()

    # model.add(Embedding(8, 64))
    # model.add(LSTM(
    #     8, 720,
    #     # activation='sigmoid',
    #     # inner_activation='hard_sigmoid'
    #     return_sequences=True
    # ))
    # model.add(TimeDistributedDense(720, 240))
    # lstm1 = LSTM(
    #     512, 256,
    #     # activation='sigmoid',
    #     # inner_activation='hard_sigmoid',
    #     return_sequences=True
    # )
    # # lstm1.connect(lstm1)
    # model.add(lstm1)
    model.add(Dropout(0.5))
    model.add(LSTM(
        34, 64,
        # activation='sigmoid',
        # inner_activation='hard_sigmoid',
        # return_sequences=True
    ))
    model.add(Activation('sigmoid'))
    # model.add(TimeDistributedDense(64, 32))
    model.add(Dropout(0.5))
    # model.add(GRU(
    #     32, 16,
    #     # activation='sigmoid',
    #     # inner_activation='hard_sigmoid',
    #     # return_sequences=True
    # ))
    # model.add(Flatten())
    # model.add(Dropout(0.5))
    model.add(Dense(64, 1))
    model.add(Activation('softmax'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        class_mode="binary"
    )

    return model

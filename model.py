# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM


def build_model():
    max_features = 20000

    model = Sequential()

    model.add(Embedding(max_features, 256))
    model.add(LSTM(256, 128))
    model.add(Dropout(0.5))
    model.add(Dense(128, 1))
    model.add(Activation('sigmoid'))

    model.compile(
        loss='binary_crossentropy', optimizer='adam', class_mode="binary")

    return model

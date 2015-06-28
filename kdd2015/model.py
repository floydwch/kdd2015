# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers.core import Dense, MaxoutDense, Dropout, Activation, TimeDistributedDense, Flatten, Merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN, SimpleDeepRNN, JZS1, JZS2, JZS3
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.regularizers import l2


def build_model():
    # max_features = 8

    # model_left = Sequential()
    # model_left.add(GRU(
    #     33, 512,
    #     # activation='sigmoid',
    #     # inner_activation='hard_sigmoid',
    #     # return_sequences=True
    # ))
    # model_left.add(Activation('sigmoid'))
    # # model_left.add(Dense(512, 256, activation='sigmoid'))
    # model_right = Sequential()
    # model_right.add(GRU(
    #     33, 512,
    #     # activation='sigmoid',
    #     # inner_activation='hard_sigmoid',
    #     # return_sequences=True
    # ))
    # model_right.add(Activation('sigmoid'))
    # model_right.add(Dense(512, 256, activation='sigmoid'))

    rnn = Sequential()
    rnn.add(Dropout(0.5))
    rnn.add(JZS3(63, 256))
    rnn.add(MaxoutDense(256, 256, nb_feature=64))
    rnn.add(Dropout(0.5))
    rnn.add(Dense(256, 128, activation='sigmoid'))
    rnn.add(MaxoutDense(128, 128, nb_feature=64))
    rnn.add(Dropout(0.5))
    # rnn.add(Dense(128, 64))
    # rnn.add(MaxoutDense(64, 64, nb_feature=32))
    # rnn.add(Dropout(0.5))

    # rnn2 = Sequential()
    # rnn2.add(Dropout(0.5))
    # rnn2.add(JZS3(63, 256))
    # rnn2.add(MaxoutDense(256, 256, nb_feature=64))
    # rnn2.add(Dropout(0.5))
    # rnn.add(Dense(256, 128, activation='sigmoid'))
    # rnn.add(Dropout(0.5))

    mlp = Sequential()
    mlp.add(Dropout(0.5))
    mlp.add(Dense(15, 64, activation='sigmoid'))
    mlp.add(MaxoutDense(64, 64, nb_feature=32))
    mlp.add(Dropout(0.5))

    # mlp.add(BatchNormalization((32,)))

    mlp.add(Dense(64, 64, activation='sigmoid'))
    mlp.add(MaxoutDense(64, 64, nb_feature=32))
    mlp.add(Dropout(0.5))
    # mlp.add(BatchNormalization((32,)))

    # mlp.add(Dense(128, 256, activation='softmax'))
    # mlp.add(Dropout(0.5))

    model = Sequential()
    model.add(Merge([rnn, mlp], mode='concat'))

    model.add(Dense(192, 1, activation='sigmoid'))
    # model.add(BatchNormalization((1,)))

    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        class_mode="binary"
    )

    return model

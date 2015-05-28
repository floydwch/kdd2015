# -*- coding: utf-8 -*-
import os.path

import pandas as pd
from numpy import array, vstack
from keras.preprocessing import sequence
import h5py

from .feature import indexing


def load_csv():
    log_train = pd.read_csv('log_train.csv')
    log_test = pd.read_csv('log_test.csv')
    df_ans = pd.read_csv('truth_train.csv', header=None)

    log_train['train'] = True
    log_test['train'] = False

    df_ans.columns = ['enrollment_id', 'dropout']
    df_ans = df_ans.set_index('enrollment_id')

    df = pd.concat([log_train, log_test], ignore_index=True)
    # df = df.merge(df_ans, 'left')

    return df, df_ans


def load_data():
    df, df_ans = load_csv()

    if not os.path.isfile('data.h5'):
        x_train, y_train, x_test = df2array(df, df_ans)

        with h5py.File('data.h5', 'w') as h5f:
            h5f.create_dataset('x_train', data=x_train)
            h5f.create_dataset('y_train', data=y_train)
            h5f.create_dataset('x_test', data=x_test)
    else:
        with h5py.File('data.h5', 'r') as h5f:
            x_train = h5f['x_train'][:]
            y_train = h5f['y_train'][:]
            x_test = h5f['x_test'][:]

    return df, df_ans, x_train, y_train, x_test


def df2array(df, df_ans):
    maxlen = 100
    df = indexing(df)
    enrollments = df.groupby('enrollment_id').groups
    x_train = []
    y_train = []
    x_test = []

    # event_ids are the indices of events in the dataframe
    for enrollment_id, event_ids in enrollments.items():
        if enrollment_id in df_ans.index:
            x_train.append(
                array([df.iloc[event_id]['event'] for event_id in event_ids]))
            y_train.append(df_ans.ix[enrollment_id]['dropout'])
        else:
            x_test.append(
                array([df.iloc[event_id]['event'] for event_id in event_ids]))

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    return vstack(x_train), array(y_train), vstack(x_test)

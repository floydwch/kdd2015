# -*- coding: utf-8 -*-
import os.path
import pandas as pd
from numpy import array, vstack
import h5py
from keras.preprocessing import sequence


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


def indexing(df):
    index_items = df['event'].value_counts().index.tolist()
    index_map = dict(map(reversed, enumerate(index_items)))

    df['event'] = df['event'].map(index_map)

    return df


def df2array(df, df_ans):
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

    x_train = sequence.pad_sequences(x_train)
    x_test = sequence.pad_sequences(x_test)

    return vstack(x_train), array(y_train), vstack(x_test)


def load_data():
    df, df_ans = load_csv()

    if not os.path.isfile('data.h5'):
        x_train, y_train, x_test = df2array(df, df_ans)

        h5f = h5py.File('data.h5', 'w')
        h5f.create_dataset('x_train', data=x_train)
        h5f.create_dataset('y_train', data=y_train)
        h5f.create_dataset('x_test', data=x_test)
        h5f.close()
    else:
        h5f = h5py.File('data.h5', 'r')
        x_train = h5f['x_train'][:]
        y_train = h5f['y_train'][:]
        x_test = h5f['x_test'][:]
        h5f.close()

    return df, df_ans, x_train, y_train, x_test


if __name__ == '__main__':
    df, df_ans, x_train, y_train, x_test = load_data()













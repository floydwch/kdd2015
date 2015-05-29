# -*- coding: utf-8 -*-
from numpy import array, vstack
from keras.preprocessing import sequence


def indexing(df):
    index_items = df['event'].value_counts().index.tolist()
    index_map = dict(map(reversed, enumerate(index_items, 1)))

    df['event'] = df['event'].map(index_map)

    return df


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

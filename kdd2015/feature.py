# -*- coding: utf-8 -*-
from __future__ import print_function

from itertools import groupby
from multiprocessing import Pool

# from dateutil import parser
from pandas import date_range
from numpy import array, vstack, random
from keras.preprocessing import sequence
from more_itertools import chunked, flatten, first

from .analyze import time_bound

from ptpdb import set_trace


def indexing(df):
    index_items = df['event'].value_counts().index.tolist()
    index_map = dict(map(reversed, enumerate(index_items, 1)))

    df['event'] = df['event'].map(index_map)

    return df


def _n_run_len_less_coding(seq):
    n = 1
    skip = 0
    while True:
        groups = list(chunked(seq[skip:], n))
        if len(groups) == 1:
            break
        seq = seq[:skip] + list(flatten(map(first, groupby(groups))))
        last_len = len(groups[-1])
        if last_len < n:
            skip += 1
            continue
        n += 1
    return seq


def _event_collapse(series):
    # set_trace()
    if len(series) == 0:
        result = '-1'
    else:
        result = ','.join(
            map(str, _n_run_len_less_coding(series.values.tolist())))
    # set_trace()
    return result


def _choice(series):
    if len(series):
        return random.choice(series)
    else:
        return -1


def _build_array(args):
    logs_df, enrollments_df = args[0], args[1]
    enrollment_id, event_ids = args[2], args[3]

    if not hasattr(_build_array, '_time_bound_cache'):
        _build_array._time_bound_cache = {}

    course_id = enrollments_df.loc[enrollment_id]['course_id']
    if course_id not in _build_array._time_bound_cache:
        _build_array._time_bound_cache[course_id] = time_bound(
            logs_df, enrollments_df, course_id)

    # set_trace()
    logs_df = logs_df.loc[event_ids]
    logs_df = logs_df[['time', 'event']]
    logs_df = logs_df.set_index('time')
    logs_df = logs_df.resample('H', how=_choice)
    # logs_df.reset_index(drop=True)

    start_time, end_time = _build_array._time_bound_cache[course_id]
    hr_indices = date_range(start_time, periods=720, freq='H').to_native_types()

    result = []
    for hr_index in hr_indices:
        set_trace()
        assert 0 <= len(logs_df[hr_index:hr_index]['event']) <= 1
        if len(logs_df[hr_index:hr_index]['event']) == 1:
            event = logs_df[hr_index:hr_index]['event'][0]
        else:
            event = -1
        result.append(event)

    # set_trace()
    # previous_time = logs_df.index[0]
    # result = []

    # day_diff = (previous_time - start_time).days

    # # try:
    # assert day_diff >= 0
    # # except:
    #     # set_trace()

    # for i in range(day_diff):
    #     result.append(-1)  # no-op

    # for event_id in logs_df.index:
    #     # set_trace()
    #     # time = event_id
    #     events = map(int, logs_df.loc[event_id]['event'].split(','))

    #     for event in events:
    #         result.append(event)
    #     # day_diff = (time - previous_time).days

    #     # assert day_diff >= 0

    #     # for i in range(day_diff):
    #     #     result.append(-2)  # no-op

    #     # result.append(logs_df.iloc[event_id]['event'])
    #     # previous_time = time
    # else:
    #     day_diff = (end_time - event_id).days

    #     assert day_diff >= 0

    #     for i in range(day_diff):
    #         result.append(-1)  # no-op
    # print(enrollment_id, len(result))
    # set_trace()
    assert len(result) == 720
    print(enrollment_id)
    return (enrollment_id, array(result))


def exclude(logs_df):
    logs_df = logs_df[logs_df['event'] != 'page_close']
    logs_df = logs_df[logs_df['event'] != 'nagivate']
    logs_df.reset_index(drop=True)
    return logs_df


def df2array(df, df_ans, enrollments_df):
    maxlen = 720
    df = exclude(df)
    df = indexing(df)
    enrollments = df.groupby('enrollment_id').groups
    x_train = []
    y_train = []
    x_test = []

    enrollments_df = enrollments_df.set_index('enrollment_id', False)

    # pool = Pool()
    built_arrays = dict(map(_build_array, map(
        lambda x: (df, enrollments_df, x[0], x[1]), enrollments.items())))
    # pool.close()
    # pool.join()

    for enrollment_id, built_array in built_arrays.items():
        if enrollment_id in df_ans.index:
            x_train.append(built_array)
            y_train.append(df_ans.ix[enrollment_id]['dropout'])
        else:
            x_test.append(built_array)

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    return vstack(x_train), array(y_train), vstack(x_test)
    # return x_train, array(y_train), x_test

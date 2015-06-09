# -*- coding: utf-8 -*-
from __future__ import print_function

from itertools import groupby
from collections import Counter
from multiprocessing import Pool

# from dateutil import parser
# from pandas import date_range
from numpy import array, vstack, random
from keras.preprocessing import sequence
from more_itertools import chunked, flatten, first

from .analyze import time_bound

# from ptpdb import set_trace


TIMESTEP_WINDOW_SIZE = 8


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


def _time_events(series):
    events = list(
        map(first, Counter(series).most_common(TIMESTEP_WINDOW_SIZE)))

    for i in range(TIMESTEP_WINDOW_SIZE - len(events)):
        events.append(0)

    return events


def _build_array(args):
    logs_df, enrollments_df = args[0], args[1]
    enrollment_id, event_ids = args[2], args[3]

    if not hasattr(_build_array, '_time_bound_cache'):
        _build_array._time_bound_cache = {}

    course_id = enrollments_df.loc[enrollment_id]['course_id']
    # if course_id == '81UZtt1JJwBFYMj5u38WNKCSVA4IJSDv':
        # set_trace()
    if course_id not in _build_array._time_bound_cache:
        _build_array._time_bound_cache[course_id] = time_bound(
            logs_df, enrollments_df, course_id)

    logs_df = logs_df.loc[event_ids]
    logs_df = logs_df[['time', 'event']]
    logs_df = logs_df.set_index('time')
    logs_df = logs_df.resample('D', how=_time_events)
    # logs_df.reset_index(drop=True)
    # set_trace()

    course_start_time, course_end_time = \
        _build_array._time_bound_cache[course_id]

    enrollment_start_time = logs_df.index[0]
    enrollment_end_time = logs_df.index[-1]
    # hr_indices = date_range(
    #     course_start_time, periods=720, freq='H').to_native_types()

    result = []

    late_days = (enrollment_start_time - course_start_time).days
    for diff in range(late_days):
        result.append([0] * TIMESTEP_WINDOW_SIZE)

    # set_trace()

    for events in logs_df['event'].values:
        result.append(events)

    leave_early_days = (course_end_time - enrollment_end_time).days
    for diff in range(leave_early_days):
        result.append([0] * TIMESTEP_WINDOW_SIZE)

    # for hr_index in hr_indices:
    #     set_trace()
    #     assert 0 <= len(logs_df[hr_index:hr_index]['event']) <= 1
    #     if len(logs_df[hr_index:hr_index]['event']) == 1:
    #         event = logs_df[hr_index:hr_index]['event'][0]
    #     else:
    #         event = -1
    #     result.append(event)

    # set_trace()
    # previous_time = logs_df.index[0]
    # result = []

    # day_diff = (previous_time - course_start_time).days

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
    #     day_diff = (course_end_time - event_id).days

    #     assert day_diff >= 0

    #     for i in range(day_diff):
    #         result.append(-1)  # no-op
    # print(enrollment_id, len(result))
    # set_trace()
    # set_trace()
    # result = list(result)
    # try:
    # set_trace()
    assert len(result) == 30
    # except:
    # set_trace()
    # print(enrollment_id)
    # print((course_end_time - course_start_time).days)
    return (enrollment_id, array(result))


def exclude(logs_df):
    logs_df = logs_df[logs_df['event'] != 'page_close']
    logs_df = logs_df[logs_df['event'] != 'nagivate']
    logs_df.reset_index(drop=True)
    return logs_df


def df2array(df, df_ans, enrollments_df):
    # maxlen = TIMESTEP_WINDOW_SIZE * 30
    # df = exclude(df)
    df = indexing(df)
    enrollments = df.groupby('enrollment_id').groups
    x_train = []
    y_train = []
    x_test = []

    enrollments_df = enrollments_df.set_index('enrollment_id', False)

    # pool = Pool(2)
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

    # x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    # x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    return array(x_train), array(y_train), array(x_test)
    # return x_train, array(y_train), x_test

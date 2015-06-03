# -*- coding: utf-8 -*-
from __future__ import print_function

from multiprocessing import Pool

from dateutil import parser
from numpy import array, vstack
from keras.preprocessing import sequence

from .analyze import time_bound


def indexing(df):
    index_items = df['event'].value_counts().index.tolist()
    index_map = dict(map(reversed, enumerate(index_items, 1)))

    df['event'] = df['event'].map(index_map)

    return df


# def _day_diff(day1, day2):
#     # month_days = {
#     #     1: 31,
#     #     2: 28,
#     #     3: 31,
#     #     4: 30,
#     #     5: 31,
#     #     6: 30,
#     #     7: 31,
#     #     8: 31,
#     #     9: 30,
#     #     10: 31,
#     #     11: 30,
#     #     12: 31
#     # }

#     # try:
#     diff = (day2 - day1).days
#     # assert day1 <= day2
#     # assert 0 <= (day2[0] - day1[0]) <= 1
#     # except:
#         # print(day1, day2)

#     # if (day2[0] - day1[0]) == 0:
#     #     diff = (day2[1] - day1[1]) * month_days[day1[1]] + \
#     #         (day2[2] - day1[2])
#     # else:
#     #     diff = (day2[1] + 12 - day1[1]) * month_days[day1[1]] + \
#     #         (day2[2] - day1[2])

#     return diff


def _build_array(args):
    logs_df, enrollments_df = args[0], args[1]
    enrollment_id, event_ids = args[2], args[3]

    if not hasattr(_build_array, '_time_bound_cache'):
        _build_array._time_bound_cache = {}

    course_id = enrollments_df.loc[enrollment_id]['course_id']
    if course_id not in _build_array._time_bound_cache:
        _build_array._time_bound_cache[course_id] = time_bound(
            logs_df, enrollments_df, course_id)

    start_time, end_time = _build_array._time_bound_cache[course_id]

    previous_time = parser.parse(logs_df.iloc[event_ids[0]]['time'][:10])
    result = []

    day_diff = (previous_time - start_time).days

    assert day_diff >= 0

    for i in range(day_diff):
        result.append(-1)  # no-op

    for event_id in event_ids:
        time = parser.parse(logs_df.iloc[event_id]['time'][:10])
        day_diff = (time - previous_time).days

        assert day_diff >= 0

        for i in range(day_diff):
            result.append(-2)  # no-op

        result.append(logs_df.iloc[event_id]['event'])
        previous_time = time
    else:
        day_diff = (end_time - time).days

        assert day_diff >= 0

        for i in range(day_diff):
            result.append(-3)  # no-op
    # print(enrollment_id, len(result))
    return (enrollment_id, array(result))


def exclude(logs_df):
    logs_df = logs_df[logs_df['event'] != 'page_close']
    logs_df = logs_df[logs_df['event'] != 'nagivate']
    logs_df.index = range(len(logs_df))
    return logs_df


def df2array(df, df_ans, enrollments_df):
    maxlen = None
    df = exclude(df)
    df = indexing(df)
    enrollments = df.groupby('enrollment_id').groups
    x_train = []
    y_train = []
    x_test = []

    enrollments_df = enrollments_df.set_index('enrollment_id', False)

    pool = Pool()
    built_arrays = dict(pool.map(_build_array, map(
        lambda x: (df, enrollments_df, x[0], x[1]), enrollments.items())))
    pool.close()
    pool.join()

    for enrollment_id, built_array in built_arrays.items():
        if enrollment_id in df_ans.index:
            x_train.append(built_array)
            y_train.append(df_ans.ix[enrollment_id]['dropout'])
        else:
            x_test.append(built_array)

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    return vstack(x_train), array(y_train), vstack(x_test)

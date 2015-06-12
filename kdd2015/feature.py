# -*- coding: utf-8 -*-
from __future__ import print_function

from itertools import groupby, repeat
from collections import Counter
from multiprocessing import Pool

# from dateutil import parser
import pandas as pd
from pandas import date_range, DataFrame, Series, IndexSlice, MultiIndex
from numpy import array, vstack, random, zeros, array_split
from keras.preprocessing import sequence
from more_itertools import chunked, flatten, first

from .analyze import time_bound, fetch_user

from ipdb import set_trace


TIMESTEP_WINDOW_SIZE = 34


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
    # set_trace()
    events = [0] * TIMESTEP_WINDOW_SIZE

    for feature_index, count in series.value_counts().iteritems():
        events[feature_index + 24] = count

    for hour in series.index.hour:
        events[hour] += 1

    login_hours = len(set(series.index.hour))
    events[24] = login_hours

    # events = list(
    #     map(first, Counter(series).most_common(TIMESTEP_WINDOW_SIZE)))

    # for i in range(TIMESTEP_WINDOW_SIZE - len(events)):
    #     events.append(0)
    # from ptpdb import set_trace
    # set_trace()

    return events


def _build_array(args):
    logs_df, enrollments_df = args[0], args[1],
    enrollment_id, event_ids = args[2], args[3]

    print(enrollment_id)

    if not hasattr(_build_array, '_time_bound_cache'):
        _build_array._time_bound_cache = {}

    course_id = enrollments_df.loc[enrollment_id]['course_id']
    username = enrollments_df.loc[enrollment_id]['username']

    if course_id not in _build_array._time_bound_cache:
        _build_array._time_bound_cache[course_id] = time_bound(
            logs_df, enrollments_df, course_id)

    enrollment_logs_df = logs_df.loc[event_ids]
    enrollment_logs_df = enrollment_logs_df[['time', 'event']]
    enrollment_logs_df = enrollment_logs_df.set_index('time')
    enrollment_logs_df = enrollment_logs_df.resample('D', how=_time_events)

    course_start_time, course_end_time = \
        _build_array._time_bound_cache[course_id]

    enrollment_start_time = enrollment_logs_df.index[0]
    enrollment_end_time = enrollment_logs_df.index[-1]

    result = []

    late_days = (enrollment_start_time - course_start_time).days
    for diff in range(late_days):
        result.append([0] * TIMESTEP_WINDOW_SIZE)

    for events in enrollment_logs_df['event'].values:
        result.append(events)

    leave_early_days = (course_end_time - enrollment_end_time).days
    for diff in range(leave_early_days):
        result.append([0] * TIMESTEP_WINDOW_SIZE)

    # if not hasattr(_build_array, '_fetch_user_cache'):
    #     _build_array._fetch_user_cache = {}

    # if username not in _build_array._fetch_user_cache:
    #     _build_array._fetch_user_cache[username] = fetch_user(
    #         logs_df, enrollments_df, username)

        user_logs_df = logs_df[logs_df['username'] == username]

        if not isinstance(user_logs_df, Series):
            user_logs_df = user_logs_df[user_logs_df.index != enrollment_id]
            user_logs_df['enrollment_id'] = user_logs_df.index
            # TODO user_logs_df.index.unique()
            user_logs_df = user_logs_df.set_index('time')
            # user_logs_df.index = user_logs_df.index.normalize()

            # events_count = len(user_logs_df)

            # set_trace()
            enum_dates = enumerate(
                date_range(course_start_time, course_end_time))
            enum_dates = filter(
                lambda i_date: (i_date[0] >= late_days) and
                (i_date[0] < (30 - leave_early_days)),
                enum_dates
            )

            dates_counter = Counter(
                user_logs_df.index.map(lambda x: str(x.date())))

            normalized_indices = user_logs_df.index.normalize()

            # set_trace()

            for i, date in enum_dates:
                date = str(date.date())

                # print(i)

                present_in_others = 0

                # try:
                if date in normalized_indices:
                    present_in_others = len(
                        user_logs_df[date]['enrollment_id'].unique())
                # except:
                    # set_trace()

                # if events_count == 0:
                    # break

                # assert events_count > 0

                # logs present in other enrollments same date.
                # set_trace()
                # print(date)
                # try:
                result[i][32] = dates_counter[date]
                    # events_count -= result[i][32]
                # except:
                    # set_trace()
                # print(result[i][32])

                result[i][33] = present_in_others

    assert len(result) == 30
    return (enrollment_id, array(result))


def exclude(logs_df):
    logs_df = logs_df[logs_df['event'] != 'page_close']
    logs_df = logs_df[logs_df['event'] != 'nagivate']
    logs_df.reset_index(drop=True)
    return logs_df


def df2array(logs_df, truth_df, enrollments_df):
    logs_df = indexing(logs_df)
    truth_df = truth_df.set_index('enrollment_id')
    enrollments = logs_df.groupby('enrollment_id').groups
    x_train = []
    y_train = []
    x_test = []

    enrollments_df = enrollments_df.set_index('enrollment_id', False)

    pool = Pool()
    built_arrays = dict(pool.map(_build_array, map(
        lambda x: (logs_df, enrollments_df, x[0], x[1]), enrollments.items())))
    pool.close()
    pool.join()

    for enrollment_id, built_array in built_arrays.items():
        if enrollment_id in truth_df.index:
            x_train.append(built_array)
            y_train.append(truth_df.ix[enrollment_id]['dropout'])
        else:
            x_test.append(built_array)

    return array(x_train), array(y_train), array(x_test)


def _extract_date_event_freq(index, log_df):
        print(index[0])
        # set_trace()
        freqs = log_df.xs(index, level=['enrollment_id', 'date'])['event'].value_counts(sort=False).values
        categories = log_df['event'].cat.categories.values
        return Series(freqs, categories)


def _extract_enrollment_event_freq(args):
    index, log_df = args[0], args[1]
    # print(index)
    # set_trace()
    # freqs = log_df.loc(index)['event'].value_counts(sort=False).values
    categories = log_df['event'].cat.categories.values
    multi_index = MultiIndex.from_arrays([[index[0]] * 30, [i for i in range(30)]], names=['enrollment_id', 'day'])
    event_freq_df = DataFrame(index=multi_index, columns=categories)
    event_freq_df['date'] = index[1]
    event_freq_df.reset_index(inplace=True)
    event_freq_df = event_freq_df[['enrollment_id', 'date']].apply(_extract_date_event_freq, axis=1, args=(log_df,))
    # set_trace()
    # freqs = log_df.xs(
    #     [index[0], dates],
    #     level=['enrollment_id', 'date']).value_counts(sort=False).values
    # freqs = [freqs_dict[freq_key] for freq_key in categories]

    # event_freq_df.loc[0] = freqs
    # set_trace()
    return event_freq_df


def append_features(enrollment_df, log_df):
    # def extract_event_freq(index):
    #     print(index)
    #     freqs = log_df.xs(
    #         index,
    #         level=['enrollment_id', 'date'])['event'].value_counts(
    #             sort=False).values
    #     categories = log_df['event'].cat.categories.values
    #     return Series(freqs, categories)
    njob = 4

    log_df = log_df.reset_index()
    log_df.set_index(['enrollment_id', 'date'], inplace=True)
    # log_df = log_df['event']

    # enrollment_df.reset_index(inplace=True)
    enrollment_df = enrollment_df[['date']]
    enrollment_df.index = enrollment_df.index.droplevel(1)
    # set_trace()
    # indices = [(i, enrollment_df.xs(i, level='enrollment_id')['date']
    #             ) for i in enrollment_df.index.get_level_values(
    #                 'enrollment_id').unique()]
    records = [(i, enrollment_df.loc[i]['date'].values) for i in enrollment_df.index.unique()]
    # set_trace()
    pool = Pool(njob)
    partial_event_freq_df = pool.map(
        _extract_enrollment_event_freq,
        zip(records, repeat(log_df)))
    event_freq_df = pd.concat(partial_event_freq_df)
    # set_trace()
    event_freq_df.reset_index(drop=True, inplace=True)
    enrollment_df.reset_index(inplace=True)
    enrollment_df = pd.concat([enrollment_df, event_freq_df], axis=1)
    set_trace()

    # enrollment_df = enrollment_df.join(
    #     enrollment_df[['enrollment_id', 'date']].apply(
    #         extract_event_freq, axis=1))

    pool.close()
    pool.join()

    # set_trace()

    return enrollment_df

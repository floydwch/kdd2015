# -*- coding: utf-8 -*-
from __future__ import print_function, division

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

# from ipdb import set_trace


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


def _hour_zone(time):
    hour = time.hour

    if hour in [22, 23, 0, 1, 2, 3]:
        zone = 'z0'
    elif hour in range(4, 9):
        zone = 'z1'
    elif hour in range(9, 12):
        zone = 'z2'
    elif hour in range(12, 15):
        zone = 'z3'
    elif hour in range(15, 18):
        zone = 'z4'
    elif hour in range(18, 22):
        zone = 'z5'

    return zone


def _get_hour(time):
    return time.hour


def _extract_date_event_freq(index, log_df):
        print(index[0])
        enrollment_id, date, username = index[0], index[1], index[2]
        # set_trace()

        enrollment_logs = log_df.xs((enrollment_id, date), level=['enrollment_id', 'date'])
        user_logs = log_df.xs((username, date), level=['username', 'date'])

        freqs = enrollment_logs['event'].value_counts(sort=False).values
        event_cat_features = log_df['event'].cat.categories.values
        features = Series(freqs, event_cat_features)

        freqs_sum = freqs.sum()
        assert freqs_sum >= 0

        if freqs_sum > 0:
            normal_freqs = freqs / freqs_sum
        else:
            normal_freqs = freqs

        normal_event_cat_features = 'normal_' + event_cat_features

        for i, category in enumerate(normal_event_cat_features):
            features[category] = normal_freqs[i]

        hour_zones = enrollment_logs['time'].map(_hour_zone).value_counts(sort=False, normalize=True).to_dict()
        # set_trace()
        for zone in ['z0', 'z1', 'z2', 'z3', 'z4', 'z5']:
            features[zone] = hour_zones.get(zone, 0)

        hour_distri = enrollment_logs['time'].map(_get_hour).value_counts(sort=False).to_dict().keys()

        features['hour_count'] = len(hour_distri)

        hour_distri = sorted(hour_distri)
        longest_cont_hours = 1
        tmp_longest_cont_hours = 1

        for i in range(1, len(hour_distri)):
            if hour_distri[i] - hour_distri[i - 1] == 1:
                tmp_longest_cont_hours += 1
            else:
                tmp_longest_cont_hours = 1

            if tmp_longest_cont_hours > longest_cont_hours:
                longest_cont_hours = tmp_longest_cont_hours

        if len(hour_distri) == 0:
            longest_cont_hours = 0

        features['longest_cont_hours'] = longest_cont_hours

        # set_trace()

        user_logs.reset_index(inplace=True)
        # set_trace()
        date_log_count = user_logs['enrollment_id'].count()
        enrollment_log_count = user_logs[user_logs['enrollment_id'] == enrollment_id]['enrollment_id'].count()

        if date_log_count == 0:
            act_ratio = 0
        else:
            act_ratio = enrollment_log_count / date_log_count

        features['act_ratio'] = act_ratio

        login_hours = len(user_logs['time'].map(_get_hour).unique())
        features['login_hours'] = login_hours

        total_hour_zones = user_logs['time'].map(_hour_zone).value_counts(sort=False, normalize=True).to_dict()
        for zone in ['z0', 'z1', 'z2', 'z3', 'z4', 'z5']:
            features[zone + '_total'] = total_hour_zones.get(zone, 0)

        total_course = len(user_logs['course_id'].unique())
        features['total_course'] = total_course

        # user_logs.xs(username, level='username')[]

        # if total_course > 0:
        #     dropout_stat = user_logs[['enrollment_id', 'dropout']].drop_duplicates()['dropout'].value_counts().to_dict()
        #     dropout_count = dropout_stat.get(1, 0)
        #     no_dropout_count = dropout_stat.get(0, 0)
        #     unknow_dropout_count = total_course - (dropout_count + no_dropout_count)
        #     dropout_prob = (dropout_count + 0.5 * unknow_dropout_count) / total_course
        # else:
        #     dropout_prob = 0.5

        # features['dropout_prob'] = dropout_prob

        # set_trace()
        return features


def _extract_enrollment_event_freq(args):
    index, log_df = args[0], args[1]
    enrollment_id, dates, username = index[0], index[1], index[2]
    # print(index)
    # set_trace()
    # freqs = log_df.loc(index)['event'].value_counts(sort=False).values
    categories = log_df['event'].cat.categories.values
    multi_index = MultiIndex.from_arrays([[enrollment_id] * 30, [i for i in range(30)]], names=['enrollment_id', 'day'])
    event_freq_df = DataFrame(index=multi_index, columns=categories)
    event_freq_df['date'] = dates
    event_freq_df['username'] = [username] * 30
    # set_trace()
    event_freq_df.reset_index(inplace=True)
    event_freq_df = event_freq_df[['enrollment_id', 'date', 'username']].apply(_extract_date_event_freq, axis=1, args=(log_df,))
    # set_trace()
    # freqs = log_df.xs(
    #     [index[0], dates],
    #     level=['enrollment_id', 'date']).value_counts(sort=False).values
    # freqs = [freqs_dict[freq_key] for freq_key in categories]
    first_day = event_freq_df[event_freq_df['hour_count'] != 0].index[0]
    last_day = event_freq_df[event_freq_df['hour_count'] != 0].index[-1]
    # set_trace()
    event_freq_df['first_day'] = first_day
    event_freq_df['last_day'] = last_day
    # set_trace()
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
    # njob = 4

    log_df = log_df.reset_index()
    log_df.set_index(['enrollment_id', 'date', 'username'], inplace=True)
    # log_df = log_df['event']

    # enrollment_df.reset_index(inplace=True)
    enrollment_df = enrollment_df[['date', 'username']]
    enrollment_df.index = enrollment_df.index.droplevel(1)
    # set_trace()
    # indices = [(i, enrollment_df.xs(i, level='enrollment_id')['date']
    #             ) for i in enrollment_df.index.get_level_values(
    #                 'enrollment_id').unique()]
    # set_trace()
    records = [(i, enrollment_df.loc[i]['date'].values, enrollment_df.loc[i]['username'].iloc[0]) for i in enrollment_df.index.unique()]
    # set_trace()
    pool = Pool()
    partial_event_freq_df = pool.map(
        _extract_enrollment_event_freq,
        zip(records, repeat(log_df)))
    # set_trace()
    event_freq_df = pd.concat(partial_event_freq_df)
    # set_trace()
    event_freq_df.reset_index(drop=True, inplace=True)
    enrollment_df.reset_index(inplace=True)
    # set_trace()
    enrollment_df = enrollment_df.join(event_freq_df)
    # enrollment_df.set_index('enrollment_id', inplace=True)
    # set_trace()

    # enrollment_df = enrollment_df.join(
    #     enrollment_df[['enrollment_id', 'date']].apply(
    #         extract_event_freq, axis=1))

    pool.close()
    pool.join()

    # set_trace()

    return enrollment_df

# -*- coding: utf-8 -*-
from __future__ import print_function, division

from datetime import datetime
from itertools import groupby, repeat, combinations
from collections import defaultdict
from multiprocessing import Pool
from subprocess import call
import os.path

# from dateutil import parser
import pandas as pd
import numpy as np
from pandas import date_range, DataFrame, Series, IndexSlice, MultiIndex
from pandas.tslib import Timestamp
from numpy import array, vstack, random, zeros, array_split, mean
from keras.preprocessing import sequence
from more_itertools import chunked, flatten, first
from sklearn.preprocessing import OneHotEncoder
# from numba import jit

from .data import load_df
from .analyze import time_bound, fetch_user
from .__init__ import setting


FEATURE_NAMES = [
    'access_server',
    'access_browser',
    'discussion',
    'nagivate',
    'page_close',
    'problem_server',
    'problem_browser',
    'video',
    'wiki',
    'normal_access_server',
    'normal_access_browser',
    'normal_discussion',
    'normal_nagivate',
    'normal_page_close',
    'normal_problem_server',
    'normal_problem_browser',
    'normal_video',
    'normal_wiki',
    'z0',
    'z1',
    'z2',
    'z3',
    'z4',
    'z5',
    'hour_count',
    'longest_cont_hours',
    'act_ratio',
    'login_hours',
    'z0_total',
    'z1_total',
    'z2_total',
    'z3_total',
    'z4_total',
    'z5_total',
    'total_course',
    'first_day',
    'last_day',
    'chapter_0',
    'chapter_1',
    'chapter_2',
    'chapter_3',
    'chapter_4',
    'chapter_5',
    'chapter_6',
    'sequential_0',
    'sequential_1',
    'sequential_2',
    'sequential_3',
    'sequential_4',
    'sequential_5',
    'sequential_6',
    'video_0',
    'video_1',
    'video_2',
    'video_3',
    'video_4',
    'video_5',
    'video_6',
    'problem_0',
    'problem_1',
    'problem_2',
    'problem_3',
    'problem_4',
    'problem_5',
    'problem_6',
]

ZONE_FEATURE_NAMES = ['z0', 'z1', 'z2', 'z3', 'z4', 'z5']

ZONE_TOTAL_FEATURE_NAMES = [
    'z0_total', 'z1_total', 'z2_total',
    'z3_total', 'z4_total', 'z5_total'
]


def _hour_zone(time):
    hour = time.hour
    z0 = 'z0'
    z1 = 'z1'
    z2 = 'z2'
    z3 = 'z3'
    z4 = 'z4'
    z5 = 'z5'

    zone = {
        22: z0,
        23: z0,
        0: z0,
        1: z0,
        2: z0,
        3: z0,
        4: z1,
        5: z1,
        6: z1,
        7: z1,
        8: z1,
        9: z2,
        10: z2,
        11: z2,
        12: z3,
        13: z3,
        14: z3,
        15: z4,
        16: z4,
        17: z4,
        18: z5,
        19: z5,
        20: z5,
        21: z5
    }

    # if hour in [22, 23, 0, 1, 2, 3]:
    #     zone = 'z0'
    # elif hour in range(4, 9):
    #     zone = 'z1'
    # elif hour in range(9, 12):
    #     zone = 'z2'
    # elif hour in range(12, 15):
    #     zone = 'z3'
    # elif hour in range(15, 18):
    #     zone = 'z4'
    # elif hour in range(18, 22):
    #     zone = 'z5'

    return zone[hour]


def _get_hour(time):
    return time.hour


def _separate_source(enrollment_log):
    if enrollment_log['event'] in ['access', 'problem']:
        enrollment_log['event'] = \
            enrollment_log['event'] + '_' + enrollment_log['source']
        assert enrollment_log['event'] in [
            'access_server',
            'access_browser',
            'problem_server',
            'problem_browser'
        ]

    return enrollment_log


def _extract_weeks(enrollment_log):
    video_weeks = [0] * 7
    problem_weeks = [0] * 7
    chapter_weeks = [0] * 7
    sequential_weeks = [0] * 7

    if enrollment_log['category'] == 'video':
        assert enrollment_log['week'] >= 0
        if enrollment_log['week'] < 6:
            video_weeks[int(enrollment_log['week'])] += 1
        else:
            video_weeks[6] += 1
    elif enrollment_log['category'] == 'problem':
        assert enrollment_log['week'] >= 0
        if enrollment_log['week'] < 6:
            problem_weeks[int(enrollment_log['week'])] += 1
        else:
            problem_weeks[6] += 1
    elif enrollment_log['category'] == 'chapter':
        assert enrollment_log['week'] >= 0
        if enrollment_log['week'] < 6:
            chapter_weeks[int(enrollment_log['week'])] += 1
        else:
            chapter_weeks[6] += 1
    elif enrollment_log['category'] == 'sequential':
        assert enrollment_log['week'] >= 0
        if enrollment_log['week'] < 6:
            sequential_weeks[int(enrollment_log['week'])] += 1
        else:
            sequential_weeks[6] += 1

    return array(video_weeks), array(problem_weeks), \
        array(chapter_weeks), array(sequential_weeks)


def _extract_date_event_freq(index, log_df):
        print(index[0])
        enrollment_id, date, username = index[0], index[1], index[2]

        features = Series(zeros(len(FEATURE_NAMES)), FEATURE_NAMES)

        user_logs = log_df.xs(date, level='date')
        if user_logs.shape[0] > 0:
            if enrollment_id in user_logs.index:
                enrollment_logs = user_logs.loc[enrollment_id]
                if isinstance(enrollment_logs, DataFrame):
                    enrollment_logs = enrollment_logs.apply(
                        _separate_source, axis=1)
                    freqs = enrollment_logs['event'].value_counts(sort=False).to_dict()
                    hour_zones = enrollment_logs['time'].map(_hour_zone).value_counts(sort=False, normalize=True).to_dict()
                    hour_distri = enrollment_logs['time'].map(_get_hour).value_counts(sort=False).to_dict().keys()

                    video_weeks = zeros(7)
                    problem_weeks = zeros(7)
                    chapter_weeks = zeros(7)
                    sequential_weeks = zeros(7)

                    for _, enrollment_log in enrollment_logs[['category', 'week']].iterrows():

                        if enrollment_log['category'] == 'video':
                            assert enrollment_log['week'] >= 0
                            if enrollment_log['week'] < 6:
                                video_weeks[int(enrollment_log['week'])] += 1
                            else:
                                video_weeks[6] += 1
                        elif enrollment_log['category'] == 'problem':
                            assert enrollment_log['week'] >= 0
                            if enrollment_log['week'] < 6:
                                problem_weeks[int(enrollment_log['week'])] += 1
                            else:
                                problem_weeks[6] += 1
                        elif enrollment_log['category'] == 'chapter':
                            assert enrollment_log['week'] >= 0
                            if enrollment_log['week'] < 6:
                                chapter_weeks[int(enrollment_log['week'])] += 1
                            else:
                                chapter_weeks[6] += 1
                        elif enrollment_log['category'] == 'sequential':
                            assert enrollment_log['week'] >= 0
                            if enrollment_log['week'] < 6:
                                sequential_weeks[int(enrollment_log['week'])] += 1
                            else:
                                sequential_weeks[6] += 1
                else:
                    if enrollment_logs['event'] in ['access', 'problem']:
                        enrollment_logs.set_value('event', enrollment_logs['event'] + '_' + enrollment_logs['source'])
                        assert enrollment_logs['event'] in [
                            'access_server',
                            'access_browser',
                            'problem_server',
                            'problem_browser'
                        ]
                    freqs = {}
                    freqs[enrollment_logs['event']] = 1
                    hour_zones = {_hour_zone(enrollment_logs['time'])[0]: 1}
                    hour_distri = [_get_hour(enrollment_logs['time'])]
                    video_weeks, problem_weeks, \
                        chapter_weeks, sequential_weeks = \
                        _extract_weeks(enrollment_logs)

                freqs_sum = sum(freqs.values())
                assert freqs_sum >= 0

                for event, freq in freqs.items():
                    assert freqs_sum > 0
                    features[event] = freq
                    features['normal_' + event] = freq / freqs_sum

                for zone in ZONE_FEATURE_NAMES:
                    features[zone] = hour_zones.get(zone, 0)

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

                features['longest_cont_hours'] = longest_cont_hours

                user_logs.reset_index(inplace=True)
                date_log_count = user_logs.shape[0]
                enrollment_log_count = user_logs[user_logs['enrollment_id'] == enrollment_id].shape[0]
                act_ratio = enrollment_log_count / date_log_count
                features['act_ratio'] = act_ratio

                login_hours = user_logs['time'].map(_get_hour).unique().shape[0]
                features['login_hours'] = login_hours

                total_course = user_logs['course_id'].unique().shape[0]
                features['total_course'] = total_course

                total_hour_zones = user_logs['time'].map(_hour_zone).value_counts(sort=False, normalize=True).to_dict()
                for zone in ZONE_FEATURE_NAMES:
                    features[zone + '_total'] = total_hour_zones.get(zone, 0)

                for i in range(7):
                    features['video_' + str(i)] = video_weeks[i]
                    features['problem_' + str(i)] = problem_weeks[i]
                    features['chapter_' + str(i)] = chapter_weeks[i]
                    features['sequential_' + str(i)] = sequential_weeks[i]
        # if total_course > 0:
        #     dropout_stat = user_logs[['enrollment_id', 'dropout']].drop_duplicates()['dropout'].value_counts().to_dict()
        #     dropout_count = dropout_stat.get(1, 0)
        #     no_dropout_count = dropout_stat.get(0, 0)
        #     unknow_dropout_count = total_course - (dropout_count + no_dropout_count)
        #     dropout_prob = (dropout_count + 0.5 * unknow_dropout_count) / total_course
        # else:
        #     dropout_prob = 0.5

        # features['dropout_prob'] = dropout_prob
        return features


def _extract_enrollment_event_freq(args):
    index, log_df = args[0], args[1]
    enrollment_id, dates, username = index[0], index[1], index[2]
    event_freq_df = DataFrame(index=range(30), columns=['enrollment_id', 'date', 'username'])
    event_freq_df['enrollment_id'] = [enrollment_id for i in range(30)]
    event_freq_df['date'] = dates
    event_freq_df['username'] = [username for i in range(30)]

    del dates
    del index

    log_df = log_df.xs(username, level='username')
    event_freq_df = event_freq_df.apply(_extract_date_event_freq, axis=1, args=(log_df,))
    first_day = event_freq_df[event_freq_df['hour_count'] != 0].index[0]
    last_day = event_freq_df[event_freq_df['hour_count'] != 0].index[-1]
    event_freq_df['first_day'] = first_day
    event_freq_df['last_day'] = last_day

    return event_freq_df


def extract_time_series_features(enrollment_df, log_df, enrollment_ids):
    log_df.set_index(['enrollment_id', 'date', 'username'], inplace=True)

    enrollment_df = enrollment_df[['date', 'username', 'course_id']]
    enrollment_df.index = enrollment_df.index.droplevel(1)

    print('start')
    records = [(i, enrollment_df.loc[i, 'date'].values, enrollment_df.loc[i, 'username'].iloc[0]) for i in enrollment_ids]
    # records = [(i, enrollment_df.loc[i, 'date'].values, enrollment_df.loc[i, 'username'].iloc[0]) for i in enrollment_df.index.unique()]
    print('records')

    log_df.sortlevel(2, inplace=True)

    del log_df['start_date']
    del log_df['end_date']
    del log_df['object']
    del log_df['train']

    if setting['multiprocessing']:
        pool = Pool(setting['njob'])
        partial_event_freq_df = pool.map(
            _extract_enrollment_event_freq,
            zip(records, repeat(log_df)))
    else:
        partial_event_freq_df = map(
            _extract_enrollment_event_freq,
            zip(records, repeat(log_df)))

    event_freq_df = pd.concat(partial_event_freq_df)
    event_freq_df.reset_index(drop=True, inplace=True)
    enrollment_df.reset_index(inplace=True)
    enrollment_df = enrollment_df.join(event_freq_df)

    enrollment_df.dropna(inplace=True)

    if setting['multiprocessing']:
        pool.close()
        pool.join()

    return enrollment_df


def gen_course_edges(enrollment_df):
    enrollment_df = enrollment_df.reset_index()
    enrollment_df.drop_duplicates('enrollment_id', inplace=True)
    enrollment_df.reset_index(drop=True, inplace=True)
    user_course = enrollment_df[['username', 'course_id']]
    user_course_group = user_course.groupby('username')

    graph = defaultdict(int)

    # skip users's course index who only have 1 course
    skip_user_course_index = {}

    for index in user_course_group.groups.values():
        assert len(index) > 0
        if len(index) < 2:
            skip_user_course_index[index[0]] = True
            continue

        for edge in combinations(
                sorted(user_course.loc[index, 'course_id'].values), 2):
            graph[edge] += 1

    course_union_size = defaultdict(int)
    course_user_group = user_course.groupby('course_id')

    for comb in graph:
        assert len(course_user_group.groups[comb[0]]) > 0
        assert len(course_user_group.groups[comb[1]]) > 0
        course_union_size[comb] = \
            len(list(filter(lambda x: x not in skip_user_course_index,
                course_user_group.groups[comb[0]]))) + \
            len(list(filter(lambda x: x not in skip_user_course_index,
                course_user_group.groups[comb[1]])))

    ratio_graph = defaultdict(int)

    for edge, count in graph.items():
        assert course_union_size[edge] > 0
        ratio_graph[edge] = count / course_union_size[edge]

    return list(filter(lambda x: ratio_graph[x] >= 0.045, ratio_graph))


def store_course_edges():
    enrollment_df, truth_df, log_df, course_df, object_df = load_df()
    course_edges = gen_course_edges(enrollment_df)
    mapping = {}

    for i, node in enumerate(set(flatten(course_edges)), 1):
        mapping[node] = i

    with open('course_node_mapping.txt', 'w') as f:
        for node in mapping:
            f.write("%d %s\n" % (mapping[node], node))

    with open('course_edges.txt', 'w') as f:
        for edge in course_edges:
            f.write("%s %s\n" % (mapping[edge[0]], mapping[edge[1]]))


def store_course_embedding(course_dim):
    call([
        'deepwalk',
        '--input', 'course_edges.txt',
        '--format', 'edgelist',
        '--representation-size', str(course_dim),
        '--output', 'course_ebd.txt'
    ])


def load_course_embedding(course_dim):
    graph_mapping = {}
    graph_vectors = {}

    if not (os.path.isfile('course_node_mapping.txt') and
            os.path.isfile('course_edges.txt')):
        store_course_edges()

    with open('course_node_mapping.txt') as f:
        for line in f:
            line = line.split()
            i, course_id = int(line[0]), line[1]
            graph_mapping[i] = course_id

    if not os.path.isfile('course_ebd.txt'):
        store_course_embedding(course_dim)

    with open('course_ebd.txt') as f:
        next(f)
        for line in f:
            line = line.split()
            i = int(line[0])
            vectors = list(map(float, line[1:]))
            # graph_vectors.append([graph_mapping[i]] + vectors)
            graph_vectors[graph_mapping[i]] = vectors

    return graph_vectors


# def append_graph_features(feature_df):
#     COURSE_DIM = 4
#     columns = ['course_id']
#     for i in range(COURSE_DIM):
#         columns.append('course_%d' % i)

#     graph_vector = load_course_embedding()
#     # indexes_columns = list(graph_vector.items())
#     graph_vector_df = DataFrame(
#         graph_vector,
#         columns=columns
#     )

#     # graph_vector_df.reset_index(inplace=True)
#     # graph_vector_df['course_id'] = graph_vector_df['index']
#     # del graph_vector_df['index']

#     feature_df = feature_df.merge(graph_vector_df, how='left', on='course_id')

#     return feature_df


def _extract_enrollment_features_proc(args):
    log_df, course_df, enrollment_id, first_start_date, course_dropout_rates, truth_df = \
        args[0], args[1], args[2], args[3], args[4], args[5]

    nb_feature = 170

    working_hard_features = [
        'total_hours',
        'cross_course_hours',
        'avg_hours',
        'max_hours',
        'min_hours',
        'longest_cont_hours',
        'avg_cont_hours',
        'min_cont_hours',
        'max_rest',
        'min_rest',
        'mean_rest',
        'longgest_streak',
        'current_streak',
        'mean_streak',
    ]

    season_features = [
        'start_dayofyear',
        'first_dayofyear',
        'last_dayofyear',
    ]

    behavior_features = [
        'access_server',
        'access_browser',
        'discussion',
        'nagivate',
        'page_close',
        'problem_server',
        'problem_browser',
        'video',
        'wiki',
        'access_server_ratio',
        'access_browser_ratio',
        'discussion_ratio',
        'nagivate_ratio',
        'page_close_ratio',
        'problem_server_ratio',
        'problem_browser_ratio',
        'video_ratio',
        'wiki_ratio',
    ]

    object_features = [
        'chapter_0',
        'chapter_1',
        'chapter_2',
        'chapter_3',
        'chapter_4',
        'chapter_5',
        'chapter_6',
        'sequential_0',
        'sequential_1',
        'sequential_2',
        'sequential_3',
        'sequential_4',
        'sequential_5',
        'sequential_6',
        'video_0',
        'video_1',
        'video_2',
        'video_3',
        'video_4',
        'video_5',
        'video_6',
        'problem_0',
        'problem_1',
        'problem_2',
        'problem_3',
        'problem_4',
        'problem_5',
        'problem_6',
        'chapter_0_ratio',
        'chapter_1_ratio',
        'chapter_2_ratio',
        'chapter_3_ratio',
        'chapter_4_ratio',
        'chapter_5_ratio',
        'chapter_6_ratio',
        'sequential_0_ratio',
        'sequential_1_ratio',
        'sequential_2_ratio',
        'sequential_3_ratio',
        'sequential_4_ratio',
        'sequential_5_ratio',
        'sequential_6_ratio',
        'video_0_ratio',
        'video_1_ratio',
        'video_2_ratio',
        'video_3_ratio',
        'video_4_ratio',
        'video_5_ratio',
        'video_6_ratio',
        'problem_0_ratio',
        'problem_1_ratio',
        'problem_2_ratio',
        'problem_3_ratio',
        'problem_4_ratio',
        'problem_5_ratio',
        'problem_6_ratio',
        'chapter_count',
        'sequential_count',
        'video_count',
        'problem_count',
        'chapter_ratio',
        'sequential_ratio',
        'video_object_ratio',
        'problem_ratio',
    ]

    event_count_features = [
        'event_count',
        'previous_event_count',
        'meanwhile_event_count',
        'future_event_count',
        'previous_avg_event_count',
        'meanwhile_avg_event_count',
        'future_avg_event_count',
        'avg_day_event_count',
        'min_day_event_count',
        'max_day_event_count',
    ]

    week_features = [
        'mon_hours',
        'tue_hours',
        'wed_hours',
        'thu_hours',
        'fri_hours',
        'sat_hours',
        'sun_hours',
        'mon_ratio',
        'tue_ratio',
        'wed_ratio',
        'thu_ratio',
        'fri_ratio',
        'sat_ratio',
        'sun_ratio',
    ]

    user_features = [
        'total_course_count',
        'previous_course_count',
        'meanwhile_course_count',
        'future_course_count',
        'event_count_ratio',  # compare to other courses (past and future)
        'user_first_day',
        'user_last_day',
    ]

    time_series_features = [
        'late_day',
        'leave_early_day',
        'day_count',
        'day_from_first_start',
        'day_range',
        'week1_event_ratio',
        'week2_event_ratio',
        'week3_event_ratio',
        'week4_event_ratio',
        'week5_event_ratio',
        'week1_event_count',
        'week2_event_count',
        'week3_event_count',
        'week4_event_count',
        'week5_event_count',
        'week1_hours',
        'week2_hours',
        'week3_hours',
        'week4_hours',
        'week5_hours',
        'week1_hours_ratio',
        'week2_hours_ratio',
        'week3_hours_ratio',
        'week4_hours_ratio',
        'week5_hours_ratio',
        'z0',
        'z1',
        'z2',
        'z3',
        'z4',
        'z5',
        'z0_ratio',
        'z1_ratio',
        'z2_ratio',
        'z3_ratio',
        'z4_ratio',
        'z5_ratio',
        'next_10_day_event_count',
        'next_10_day_hours',
    ]

    prior_features = [
        'dropout_rate',
    ]

    feature_series = Series(
        0,
        index=(
            working_hard_features +
            season_features +
            behavior_features +
            object_features +
            event_count_features +
            week_features +
            user_features +
            time_series_features +
            prior_features
        ),
        dtype='float64')

    print('enrollment %d' % enrollment_id)
    # import pdb; pdb.set_trace()

    feature_map = {}

    course_id = log_df.loc[enrollment_id, 'course_id']
    if isinstance(course_id, Series):
        course_id = course_id.iloc[0]

    username = log_df.loc[enrollment_id, 'username']
    if isinstance(username, Series):
        username = username.iloc[0]

    # import pdb; pdb.set_trace()
    start_date = log_df.loc[enrollment_id, 'start_date']
    if isinstance(start_date, Series):
        start_date = start_date.iloc[0]

    start_dayofyear = start_date.dayofyear
    day_from_first_start = (start_date - first_start_date).days

    feature_map['start_dayofyear'] = start_dayofyear
    feature_map['day_from_first_start'] = day_from_first_start

    user_log_df = log_df[log_df['username'] == username]
    all_course_ids = user_log_df['course_id'].unique()

    feature_map['total_course_count'] = all_course_ids.shape[0]

    end_date = log_df.loc[enrollment_id, 'end_date']
    if isinstance(end_date, Series):
        end_date = end_date.iloc[0]

    previous_course_count = 0
    meanwhile_course_count = 0
    future_course_count = 0
    previous_event_count = 0
    meanwhile_event_count = 0
    future_event_count = 0
    for i_course_id in all_course_ids:
        if course_df.loc[i_course_id, 'start_date'] < start_date:
            previous_event_count += user_log_df[user_log_df['course_id'] == i_course_id].shape[0]
            previous_course_count += 1
        elif end_date >= course_df.loc[i_course_id, 'start_date'] >= start_date:
            meanwhile_event_count += user_log_df[user_log_df['course_id'] == i_course_id].shape[0]
            meanwhile_course_count += 1
        else:  # future course
            future_event_count += user_log_df[user_log_df['course_id'] == i_course_id].shape[0]
            future_course_count += 1

    # previous_course_count = log(2 + previous_course_count)
    # meanwhile_course_count = log(2 + meanwhile_course_count)
    feature_map['previous_event_count'] = previous_event_count
    feature_map['meanwhile_event_count'] = meanwhile_event_count
    feature_map['future_event_count'] = future_event_count

    feature_map['previous_course_count'] = previous_course_count
    feature_map['meanwhile_course_count'] = meanwhile_course_count
    feature_map['future_course_count'] = future_course_count

    if previous_course_count > 0:
        previous_avg_event_count = previous_event_count / previous_course_count
    else:
        previous_avg_event_count = 0

    if meanwhile_course_count > 0:
        meanwhile_avg_event_count = meanwhile_event_count / meanwhile_course_count
    else:
        meanwhile_avg_event_count = 0

    if future_course_count > 0:
        future_avg_event_count = future_event_count / future_course_count
    else:
        future_avg_event_count = 0

    feature_map['previous_avg_event_count'] = previous_avg_event_count
    feature_map['meanwhile_avg_event_count'] = meanwhile_avg_event_count
    feature_map['future_avg_event_count'] = future_avg_event_count

    # previous_avg_event_count = log(2 + previous_avg_event_count)

    # import pdb; pdb.set_trace()
    user_first_day = (user_log_df['time'].min() - first_start_date).days
    user_last_day = (user_log_df['time'].max() - first_start_date).days

    feature_map['user_first_day'] = user_first_day
    feature_map['user_last_day'] = user_last_day

    date_series = log_df.loc[enrollment_id, 'date']
    if isinstance(date_series, Series):
        first_day = date_series.iloc[0]
        last_day = date_series.iloc[-1]

        dates = date_series.unique()
        rests = []
        aday = np.timedelta64(1, 'D')

        day_count = dates.shape[0]

        longgest_streak = 0
        tmp_streak = 0
        streaks = []
        for i in range(1, len(dates)):
            if int((dates[i] - dates[i - 1]) / aday) == 1:
                tmp_streak += 1
                if tmp_streak > longgest_streak:
                    longgest_streak = tmp_streak
            else:
                if tmp_streak > 0:
                    streaks.append(tmp_streak)
                tmp_streak = 0

        if len(streaks) == 0:
            if longgest_streak > 0:
                mean_streak = longgest_streak
            else:
                mean_streak = 0
        else:
            mean_streak = mean(streaks)

        current_streak = 0
        expand_dates = list(map(lambda x: x + 0 * aday, dates)) + [start_date + 29 * aday]
        for i in range(len(expand_dates) - 1, 0, -1):
            if int((expand_dates[i] - expand_dates[i - 1]) / aday) <= 1:
                current_streak += 1
            else:
                break

        if len(dates) > 1:
            for i in range(1, len(dates)):
                rests.append(int((dates[i] - dates[i - 1]) / aday))

            max_rest = max(rests) - 1
            min_rest = min(rests) - 1
            mean_rest = mean(array(rests) - 1)
        else:
            max_rest = 0
            min_rest = 0
            mean_rest = 0

    elif isinstance(date_series, Timestamp):
        first_day = date_series
        last_day = date_series
        max_rest = 0
        min_rest = 0
        mean_rest = 0
        longgest_streak = 0
        current_streak = 0
        mean_streak = 0
        day_count = 1
    else:
        raise NotImplementedError()

    feature_map['late_day'] = (first_day - start_date).days
    feature_map['leave_early_day'] = (last_day - start_date).days
    feature_map['first_dayofyear'] = first_day.dayofyear
    feature_map['last_dayofyear'] = last_day.dayofyear
    feature_map['day_range'] = (last_day - first_day).days + 1

    feature_map['max_rest'] = max_rest
    feature_map['min_rest'] = min_rest
    feature_map['mean_rest'] = mean_rest

    feature_map['longgest_streak'] = longgest_streak
    feature_map['current_streak'] = current_streak
    feature_map['mean_streak'] = mean_streak

    feature_map['day_count'] = day_count

    enrollment_logs = log_df.loc[enrollment_id]

    if isinstance(enrollment_logs, Series):
        event_freqs = {}
        normal_event_freqs = {}
        event_freqs[enrollment_logs['event']] = 1
        normal_event_freqs[enrollment_logs['event']] = 1
        video_weeks, problem_weeks, chapter_weeks, sequential_weeks = \
            _extract_weeks(enrollment_logs)
        event_count = 1
        avg_day_event_count = 1
        # import pdb; pdb.set_trace()
    else:
        events = enrollment_logs.apply(_separate_source, axis=1)['event']
        event_freqs = events.value_counts().to_dict()
        normal_event_freqs = events.value_counts(normalize=True).to_dict()

        event_count = enrollment_logs.shape[0]
        avg_day_event_count = event_count / day_count
        # event_count = log(event_count + 1)

        # import pdb; pdb.set_trace()

        video_weeks = [0] * 7
        problem_weeks = [0] * 7
        chapter_weeks = [0] * 7
        sequential_weeks = [0] * 7

        for _, enrollment_log in enrollment_logs[['category', 'week']].iterrows():
            if enrollment_log['category'] == 'video':
                assert enrollment_log['week'] >= 0
                if enrollment_log['week'] < 6:
                    video_weeks[int(enrollment_log['week'])] += 1
                else:
                    video_weeks[6] += 1
            elif enrollment_log['category'] == 'problem':
                assert enrollment_log['week'] >= 0
                if enrollment_log['week'] < 6:
                    problem_weeks[int(enrollment_log['week'])] += 1
                else:
                    problem_weeks[6] += 1
            elif enrollment_log['category'] == 'chapter':
                assert enrollment_log['week'] >= 0
                if enrollment_log['week'] < 6:
                    chapter_weeks[int(enrollment_log['week'])] += 1
                else:
                    chapter_weeks[6] += 1
            elif enrollment_log['category'] == 'sequential':
                assert enrollment_log['week'] >= 0
                if enrollment_log['week'] < 6:
                    sequential_weeks[int(enrollment_log['week'])] += 1
                else:
                    sequential_weeks[6] += 1

    feature_map['event_count'] = event_count
    feature_map['avg_day_event_count'] = avg_day_event_count

    event_types = [
        'access_server',
        'access_browser',
        'discussion',
        'nagivate',
        'page_close',
        'problem_server',
        'problem_browser',
        'video',
        'wiki'
    ]

    for event_type in event_types:
        feature_map[event_type] = event_freqs.get(event_type, 0)
        feature_map[event_type + '_ratio'] = normal_event_freqs.get(event_type, 0)

    # for i, value in enumerate(graph_vectors[course_id]):
    #     feature_series.set_value('course_%d' % i, value)

    feature_map['chapter_count'] = 0
    feature_map['sequential_count'] = 0
    feature_map['video_count'] = 0
    feature_map['problem_count'] = 0

    for i in range(7):
        feature_map['chapter_%d' % i] = chapter_weeks[i]
        feature_map['chapter_count'] += chapter_weeks[i]

    for i in range(7):
        feature_map['sequential_%d' % i] = sequential_weeks[i]
        feature_map['sequential_count'] += sequential_weeks[i]

    for i in range(7):
        feature_map['video_%d' % i] = video_weeks[i]
        feature_map['video_count'] += video_weeks[i]

    for i in range(7):
        feature_map['problem_%d' % i] = problem_weeks[i]
        feature_map['problem_count'] += problem_weeks[i]

    object_count = (
        feature_map['chapter_count'] +
        feature_map['sequential_count'] +
        feature_map['video_count'] +
        feature_map['problem_count'])

    if object_count > 0:
        feature_map['chapter_ratio'] = feature_map['chapter_count'] / object_count
        feature_map['sequential_ratio'] = feature_map['sequential_count'] / object_count
        feature_map['video_object_ratio'] = feature_map['video_count'] / object_count
        feature_map['problem_ratio'] = feature_map['problem_count'] / object_count
    else:
        feature_map['chapter_ratio'] = 0
        feature_map['sequential_ratio'] = 0
        feature_map['video_object_ratio'] = 0
        feature_map['problem_ratio'] = 0

    for i in range(7):
        if feature_map['chapter_count'] > 0:
            feature_map['chapter_%d_ratio' % i] = chapter_weeks[i] / feature_map['chapter_count']
        else:
            feature_map['chapter_%d_ratio' % i] = 0

    for i in range(7):
        if feature_map['sequential_count'] > 0:
            feature_map['sequential_%d_ratio' % i] = sequential_weeks[i] / feature_map['sequential_count']
        else:
            feature_map['sequential_%d_ratio' % i] = 0

    for i in range(7):
        if feature_map['video_count'] > 0:
            feature_map['video_%d_ratio' % i] = video_weeks[i] / feature_map['video_count']
        else:
            feature_map['video_%d_ratio' % i] = 0

    for i in range(7):
        if feature_map['problem_count'] > 0:
            feature_map['problem_%d_ratio' % i] = problem_weeks[i] / feature_map['problem_count']
        else:
            feature_map['problem_%d_ratio' % i]  = 0

    hours = set()
    weekdays = [0] * 7
    week_event_count = [0] * 5
    week_hours = [0] * 5
    zones = defaultdict(int)
    day_event_count = defaultdict(int)
    time_series = log_df.loc[enrollment_id, 'time']

    if isinstance(time_series, Timestamp):
        time_series = [time_series]

    for timestamp in time_series:
        week = ((timestamp - start_date).days) // 7
        week_event_count[week] += 1
        hour = (timestamp.year, timestamp.month, timestamp.day, timestamp.hour)

        hours_len = len(hours)
        hours.add(hour)

        day = (timestamp.year, timestamp.month, timestamp.day)
        day_event_count[day] += 1

        zones[_hour_zone(timestamp)] += 1

        assert len(hours) <= hours_len + 1
        if len(hours) > hours_len:
            weekdays[timestamp.dayofweek] += 1
            week_hours[week] += 1

    feature_map['min_day_event_count'] = min(day_event_count.values())
    feature_map['max_day_event_count'] = max(day_event_count.values())

    for i in range(5):
        feature_map['week%d_event_count' % (i + 1)] = week_event_count[i]
        feature_map['week%d_event_ratio' % (i + 1)] = week_event_count[i] / event_count
        feature_map['week%d_hours' % (i + 1)] = week_hours[i]
        feature_map['week%d_hours_ratio' % (i + 1)] = week_hours[i] / len(hours)

    for zone in ['z0', 'z1', 'z2', 'z3', 'z4', 'z5']:
        feature_map[zone] = zones[zone]
        feature_map[zone + '_ratio'] = zones[zone] / event_count

    feature_map['total_hours'] = len(hours)

    weekday_keys = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    for i, day in enumerate(weekday_keys):
        feature_map[day + '_hours'] = weekdays[i]
        feature_map[day + '_ratio'] = weekdays[i] / feature_map['total_hours']

    day_hours = defaultdict(int)
    cont_hours = []
    gap = True
    hours_list = sorted(list(hours))
    for i, hour in enumerate(hours_list, 1):
        day_hours[hour[:3]] += 1
        if i < len(hours_list):
            if (datetime(*hours_list[i]) - datetime(*hours_list[i - 1])).seconds == 3600:
                if gap:
                    cont_hours.append(1)
                    gap = False
                else:
                    cont_hours[-1] += 1
            else:
                gap = True

    if len(cont_hours) > 0:
        feature_map['longest_cont_hours'] = max(cont_hours)
        feature_map['avg_cont_hours'] = np.mean(cont_hours)
        feature_map['min_cont_hours'] = min(cont_hours)
    else:
        feature_map['longest_cont_hours'] = 0
        feature_map['avg_cont_hours'] = 0
        feature_map['min_cont_hours'] = 0

    feature_map['avg_hours'] = sum(day_hours.values()) / len(day_hours.keys())
    feature_map['max_hours'] = max(day_hours.values())
    feature_map['min_hours'] = min(day_hours.values())

    user_time_series = user_log_df['time']

    if isinstance(user_time_series, Timestamp):
        user_time_series = [user_time_series]

    cross_course_hours = set()
    cross_course_event_count = 0

    next_10_day_event_count = 0
    next_10_day_hours = set()

    for timestamp in user_time_series:
        hour = (timestamp.year, timestamp.month, timestamp.day, timestamp.hour)
        cross_course_hours.add(hour)
        cross_course_event_count += 1

        after_start_date = (timestamp.normalize() - start_date).days - 29

        if 1 <= after_start_date <= 10:
            next_10_day_event_count += 1
            next_10_day_hours.add(hour)

    user_enrollment_ids = user_log_df.index.unique()
    course_dropout_rate = course_dropout_rates[course_id]

    try:
        user_dropout_rate = truth_df.loc[user_enrollment_ids, 'dropout'].dropna().value_counts(normalize=True).to_dict().get(1, 0)
        feature_map['dropout_rate'] = (user_dropout_rate + course_dropout_rate) / 2
    except:
        feature_map['dropout_rate'] = course_dropout_rate

    feature_map['cross_course_hours'] = len(cross_course_hours)
    feature_map['event_count_ratio'] = event_count / cross_course_event_count

    feature_map['next_10_day_event_count'] = next_10_day_event_count
    feature_map['next_10_day_hours'] = len(next_10_day_hours)

    assert len(feature_map.keys()) == nb_feature

    for feature_name, feature_value in feature_map.items():
        feature_series.set_value(feature_name, feature_value)

    assert feature_series.shape[0] == nb_feature
    # import ipdb; ipdb.set_trace()

    # feature_series.set_value('late_day', (first_day - start_date).days)
    # feature_series.set_value('leave_early_day', (last_day - start_date).days)
    # feature_series.set_value('first_dayofyear', first_day.dayofyear)
    # feature_series.set_value('last_dayofyear', last_day.dayofyear)
    # feature_series.set_value('start_dayofyear', start_dayofyear)
    # feature_series.set_value('max_rest', max_rest)
    # feature_series.set_value('min_rest', min_rest)
    # feature_series.set_value('mean_rest', mean_rest)
    # feature_series.set_value('longgest_streak', longgest_streak)
    # feature_series.set_value('current_streak', current_streak)
    # feature_series.set_value('mean_streak', mean_streak)
    # feature_series.set_value('day_count', day_count)
    # feature_series.set_value('event_count', event_count)
    # feature_series.set_value('day_from_first_start', day_from_first_start)
    # feature_series.set_value('previous_course_count', previous_course_count)
    # feature_series.set_value('meanwhile_course_count', meanwhile_course_count)
    # feature_series.set_value('future_course_count', meanwhile_course_count)
    # feature_series.set_value('previous_event_count', previous_event_count)
    # feature_series.set_value('meanwhile_event_count', meanwhile_event_count)
    # feature_series.set_value('future_event_count', future_course_count)
    # feature_series.set_value('user_first_day', user_first_day)
    # feature_series.set_value('user_last_day', user_last_day)
    # feature_series.set_value('previous_avg_event_count', previous_avg_event_count)
    # feature_series.set_value('meanwhile_avg_event_count', meanwhile_avg_event_count)
    # feature_series.set_value('future_avg_event_count', future_avg_event_count)
    # feature_series.set_value('avg_day_event_count', avg_day_event_count)

    return feature_series


def append_course_repr(feature_df, enrollment_course_series, log_df):
    print('append course repr')
    onehot = {cat: i for i, cat in enumerate(enrollment_course_series.cat.categories)}
    enrollment_course_series = enrollment_course_series.map(onehot)
    enrollment_course_values = enrollment_course_series.values.reshape(enrollment_course_series.shape[0], 1)
    one_hot_encoder = OneHotEncoder(sparse=False).fit(enrollment_course_values)
    course_one_hot = one_hot_encoder.transform(enrollment_course_values)
    course_one_hot_df = DataFrame(course_one_hot)
    course_one_hot_df.columns = ['course_one_hot_%d' % i for i in range(39)]

    feature_df.reset_index(inplace=True)
    feature_df = pd.concat([feature_df, course_one_hot_df], axis=1)

    # import ipdb; ipdb.set_trace()
    log_df['course_id'] = log_df.index.get_level_values('course_id')
    log_df['username'] = log_df.index.get_level_values('username')

    # usernames = []

    enrollment_ids = list(map(int, feature_df['enrollment_id'].values))

    print('collect username')

    # for enrollment_id in enrollment_ids:
    #     print(enrollment_id)
    #     username = log_df.loc[enrollment_id, 'username'].iloc[0]
    #     usernames.append(username)

    usernames = log_df['username'].unique()

    user_dropout_courses = {}
    user_courses = {}

    print('collect user course')

    # import ipdb; ipdb.set_trace()
    del log_df['course_id']
    del log_df['username']
    user_enrollment_df = log_df.reset_index()
    user_enrollment_df.drop_duplicates('enrollment_id', inplace=True)
    user_enrollment_df.set_index('username', inplace=True)
    user_enrollment_df.sort_index(inplace=True)

    for username in usernames:
        print(username)
        # user_logs = log_df.xs(username, level='username')
        # user_logs.drop_duplicates('course_id', inplace=True)
        # import ipdb; ipdb.set_trace()
        this_user_course_df = user_enrollment_df.loc[username, ['course_id', 'dropout']]

        if isinstance(this_user_course_df, Series):
            if this_user_course_df['dropout'] == 1:
                dropout_courses = array([this_user_course_df['course_id']])
            else:
                dropout_courses = array([])
        else:
            dropout_courses_df = this_user_course_df[this_user_course_df['dropout'] == 1]
            if isinstance(dropout_courses_df, str):
                dropout_courses = array([dropout_courses_df])
            else:
                dropout_courses = dropout_courses_df['course_id'].values

        user_dropout_courses[username] = dropout_courses

        courses_df = user_enrollment_df.loc[username, 'course_id']

        if isinstance(courses_df, str):
            courses = array([courses_df])
        else:
            courses = courses_df.values

        user_courses[username] = courses

        # import ipdb; ipdb.set_trace()

    # import ipdb; ipdb.set_trace()
    user_dropout_courses_df = DataFrame(0, index=enrollment_ids, columns=['user_dropout_course_one_hot_%d' % i for i in range(39)], dtype='float64')
    # user_dropout_courses_df.columns = ['user_dropout_course_one_hot_%d' % i for i in range(39)]

    user_courses_df = DataFrame(0, index=enrollment_ids, columns=['user_course_one_hot_%d' % i for i in range(39)], dtype='float64')
    # user_courses_df.columns = ['user_course_one_hot_%d' % i for i in range(39)]
    # import ipdb; ipdb.set_trace()

    print('append features')

    log_df.reset_index(inplace=True)
    log_df.drop_duplicates('enrollment_id', inplace=True)
    log_df.set_index('enrollment_id', inplace=True)

    # log_df['username'] = log_df.index.get_level_values('username')
    # log_df['course_id'] = log_df.index.get_level_values('course_id')

    for enrollment_id in enrollment_ids:
        print(enrollment_id)
        # enrollment_id = int(enrollment_id)
        # import ipdb; ipdb.set_trace()
        # import ipdb; ipdb.set_trace()
        enrollment_course_encode = onehot[log_df.loc[enrollment_id, 'course_id']]
        enrollment_user = log_df.loc[enrollment_id, 'username']
        # import ipdb; ipdb.set_trace()
        sorted_user_courses = sorted(user_courses[enrollment_user])
        user_courses_vector = zeros(39)
        for course_id in sorted_user_courses:
            course_id = onehot[course_id]
            user_courses_vector += one_hot_encoder.transform(course_id)[0]

        user_courses_df.loc[enrollment_id] = user_courses_vector

        # import ipdb; ipdb.set_trace()

        sorted_user_dropout_courses = sorted(user_dropout_courses[enrollment_user])
        user_dropout_courses_vector = zeros(39)
        for course_id in sorted_user_dropout_courses:
            # import ipdb; ipdb.set_trace()
            course_id = onehot[course_id]
            if course_id != enrollment_course_encode:
                user_dropout_courses_vector += one_hot_encoder.transform(course_id)[0]

        user_dropout_courses_df.loc[enrollment_id] = user_dropout_courses_vector

    user_courses_df.reset_index(inplace=True)
    user_dropout_courses_df.reset_index(inplace=True)
    feature_df = pd.concat([feature_df, user_courses_df, user_dropout_courses_df], axis=1)
    feature_df.set_index('enrollment_id', inplace=True)

    del feature_df['index']

    return feature_df


def append_course_repr2(feature_df, log_df):
    print('append course repr2')

    log_df['course_id'] = log_df.index.get_level_values('course_id')
    course_ids = log_df['course_id'].unique()

    graph_vectors = {}
    graph_mapping = {i: k for i, k in enumerate(sorted(course_ids), 1)}
    # username_map = {k: i for i, k in enumerate(sorted(usernames), 40)}

    # import ipdb; ipdb.set_trace()

    with open('user-course-ebd', 'r') as f:
        next(f)
        for line in f:
            line = line.split()
            i = int(line[0])
            if i > 39:
                continue
            vectors = list(map(float, line[1:]))
            # import ipdb; ipdb.set_trace()
            graph_vectors[graph_mapping[i]] = vectors

    # import ipdb; ipdb.set_trace()

    enrollment_ids = feature_df.index.values

    course_graph_df = DataFrame(0, index=enrollment_ids, columns=['course_graph_%d' % i for i in range(16)], dtype='float64')

    print('append features')

    del log_df['course_id']
    log_df.reset_index(inplace=True)
    log_df.drop_duplicates('enrollment_id', inplace=True)
    log_df.set_index('enrollment_id', inplace=True)

    for enrollment_id in enrollment_ids:
        print(enrollment_id)

        course_id = log_df.loc[enrollment_id, 'course_id']
        course_graph_df.loc[enrollment_id] = graph_vectors[course_id]

    # import ipdb; ipdb.set_trace()

    course_graph_df.reset_index(inplace=True)
    feature_df.reset_index(inplace=True)
    feature_df = pd.concat([feature_df, course_graph_df], axis=1)
    feature_df.set_index('enrollment_id', inplace=True)

    del feature_df['index']

    # import ipdb; ipdb.set_trace()

    return feature_df


def append_user_repr(feature_df, log_df):
    print('append course repr2')

    log_df['username'] = log_df.index.get_level_values('username')
    usernames = log_df['username'].unique()

    graph_vectors = {}
    graph_mapping = {i: k for i, k in enumerate(sorted(usernames), 5927)}
    # username_map = {k: i for i, k in enumerate(sorted(usernames), 40)}

    # import ipdb; ipdb.set_trace()

    with open('user-object-ebd', 'r') as f:
        next(f)
        for line in f:
            line = line.split()
            i = int(line[0])
            if i < 5927:
                continue
            vectors = list(map(float, line[1:]))
            # import ipdb; ipdb.set_trace()
            graph_vectors[graph_mapping[i]] = vectors

    # import ipdb; ipdb.set_trace()

    enrollment_ids = feature_df.index.values

    user_graph_df = DataFrame(0, index=enrollment_ids, columns=['user_graph_%d' % i for i in range(64)], dtype='float64')

    print('append features')

    del log_df['username']
    log_df.reset_index(inplace=True)
    log_df.drop_duplicates('enrollment_id', inplace=True)
    log_df.set_index('enrollment_id', inplace=True)

    for enrollment_id in enrollment_ids:
        print(enrollment_id)

        username = log_df.loc[enrollment_id, 'username']
        user_graph_df.loc[enrollment_id] = graph_vectors.get(username, zeros(64))

    # import ipdb; ipdb.set_trace()

    user_graph_df.reset_index(inplace=True)
    feature_df.reset_index(inplace=True)
    feature_df = pd.concat([feature_df, user_graph_df], axis=1)
    feature_df.set_index('enrollment_id', inplace=True)

    del feature_df['index']

    # import ipdb; ipdb.set_trace()

    return feature_df


def gen_user_course_edges(log_df):
    # import ipdb; ipdb.set_trace()
    course_ids = log_df.index.get_level_values('course_id').unique()
    usernames = log_df.index.get_level_values('username').unique()

    user_courses = {}

    user_enrollment_df = log_df.reset_index()
    user_enrollment_df.drop_duplicates('enrollment_id', inplace=True)
    user_enrollment_df.set_index('username', inplace=True)
    user_enrollment_df.sort_index(inplace=True)

    for username in usernames:
        print(username)

        this_user_course_series = user_enrollment_df.loc[username, 'course_id']

        if isinstance(this_user_course_series, str):
            courses = [this_user_course_series]
        else:
            courses = this_user_course_series.values

        user_courses[username] = courses

    course_id_map = {k: i for i, k in enumerate(sorted(course_ids), 1)}
    username_map = {k: i for i, k in enumerate(sorted(usernames), 40)}

    # import ipdb; ipdb.set_trace()

    with open('user-graph', 'w') as f:

        for username, course_ids in user_courses.items():
            username_enc = username_map[username]

            for course_id in course_ids:
                course_id_enc = course_id_map[course_id]
                f.write('%d %d\n' % (course_id_enc, username_enc))


def gen_user_object_edges(log_df, object_df):
    # import ipdb; ipdb.set_trace()

    object_series = object_df['object'].value_counts()
    usernames = log_df.index.get_level_values('username').unique()
    object_ids = object_series[object_series == 1].index

    user_objects = {}

    user_enrollment_df = log_df.reset_index()
    # user_enrollment_df.drop_duplicates('enrollment_id', inplace=True)
    user_enrollment_df.set_index('username', inplace=True)
    user_enrollment_df.sort_index(inplace=True)

    for username in usernames:
        print(username)

        this_user_object_series = user_enrollment_df.loc[username, 'object']

        if isinstance(this_user_object_series, str):
            if this_user_object_series in object_ids:
                objects = [this_user_object_series]
        elif(this_user_object_series, Series):
            my_this_user_object_series = this_user_object_series[this_user_object_series.isin(object_ids)]
            objects = my_this_user_object_series.unique()
        else:
            NotImplementedError()

        user_objects[username] = objects

    # import ipdb; ipdb.set_trace()

    object_id_map = {k: i for i, k in enumerate(sorted(object_ids), 1)}
    username_map = {k: i for i, k in enumerate(sorted(usernames), object_ids.shape[0] + 1)}

    # import ipdb; ipdb.set_trace()

    with open('user-object-graph', 'w') as f:

        for username, my_object_ids in user_objects.items():
            username_enc = username_map[username]

            for my_object_id in my_object_ids:
                my_object_id_enc = object_id_map[my_object_id]
                f.write('%d %d\n' % (my_object_id_enc, username_enc))


def extend_features(feature_df, enrollment_df, log_df):

    enrollment_df['index'] = enrollment_df.index.get_level_values('enrollment_id')
    enrollment_df.index = enrollment_df.index.droplevel('day')
    enrollment_df.drop_duplicates('index', inplace=True)
    enrollment_course_series = enrollment_df['course_id'].astype('category')

    feature_df = append_course_repr(feature_df, enrollment_course_series, log_df)

    return feature_df


def extract_enrollment_features(log_df, course_df, truth_df, enrollment_df, enrollment_ids):
    print('extract enrollment features')

    log_df.reset_index(inplace=True)
    log_df.set_index('enrollment_id', drop=True, inplace=True)
    log_df['username'] = log_df['username'].astype('category')
    log_df['course_id'] = log_df['course_id'].astype('category')

    first_start_date = course_df['start_date'].min()

    course_dim = 8
    graph_vectors = load_course_embedding(course_dim)

    # enrollment_ids = log_df.index.unique()[:10000]
    # enrollment_ids = log_df.index.unique()

    # import pdb; pdb.set_trace()
    # feature_df.index.names = ['enrollment_id']
    log_df['index'] = log_df.index
    course_dropout_df = log_df.drop_duplicates('index')[['course_id', 'dropout']]
    course_dropout_df.set_index('course_id', inplace=True)

    course_dropout_rates = {}

    for course_id in course_dropout_df.index.categories:
        course_dropout_rates[course_id] = course_dropout_df.loc[course_id, 'dropout'].dropna().value_counts(normalize=True).to_dict()[1]

    records = [(log_df, course_df, enrollment_id, first_start_date, course_dropout_rates, truth_df, graph_vectors, course_dim) for enrollment_id in enrollment_ids]

    if setting['multiprocessing']:
        pool = Pool(setting['njob'])
        feature_series = pool.map(_extract_enrollment_features_proc, records)
    else:
        feature_series = map(_extract_enrollment_features_proc, records)

    if setting['multiprocessing']:
        pool.close()
        pool.join()

    feature_df = pd.concat(feature_series, axis=1).T

    feature_df.index = enrollment_ids
    feature_df.index.names = ['enrollment_id']

    print('done')
    # import pdb; pdb.set_trace()

    return feature_df


# def extract_future(feature_df, log_df, course_df, enrollment_id):
#     username = log_df.loc[enrollment_id, 'username']
#     if isinstance(username, Series):
#         username = username.iloc[0]

#     # import pdb; pdb.set_trace()
#     start_date = log_df.loc[enrollment_id, 'start_date']
#     if isinstance(start_date, Series):
#         start_date = start_date.iloc[0]

#     end_date = log_df.loc[enrollment_id, 'end_date']
#     if isinstance(end_date, Series):
#         end_date = end_date.iloc[0]

#     user_log_df = log_df[log_df['username'] == username]
#     all_course_ids = user_log_df['course_id'].unique()

#     previous_course_count = 0
#     meanwhile_course_count = 0
#     future_course_count = 0
#     future_event_count = 0
#     previous_event_count = 0

#     for i_course_id in all_course_ids:
#         if course_df.loc[i_course_id, 'start_date'] < start_date:
#             previous_event_count += user_log_df[user_log_df['course_id'] == i_course_id].shape[0]
#             previous_course_count += 1
#             course_df.loc[i_course_id, 'start_date']
#         elif end_date >= course_df.loc[i_course_id, 'start_date'] >= start_date:
#             meanwhile_course_count += 1
#         else:  # future course
#             future_course_count += 1
#             future_event_count += user_log_df[user_log_df['course_id'] == i_course_id].shape[0]


def cal_sample_weight(log_df, enrollment_ids):
    sample_weight_df = DataFrame(0, index=enrollment_ids, columns=['weight'], dtype='float64')
    sample_weight_df.index.names = ['enrollment_id']

    log_df['date'] = log_df.index.get_level_values('date')

    log_df.index = log_df.index.droplevel('username')
    log_df.index = log_df.index.droplevel('course_id')
    log_df.index = log_df.index.droplevel('date')

    for enrollment_id in enrollment_ids:
        dates = log_df.loc[enrollment_id, 'date']

        if isinstance(dates, Series):
            date_count = dates.unique().shape[0]
        else:
            date_count = 1
        # import pdb; pdb.set_trace()
        if isinstance(dates, Series):
            year = dates.iloc[0].year
            month = dates.iloc[0].month
        else:
            year = dates.year
            month = dates.month

        # count_weight1 = pow(1.09, 31 - date_count)
        # count_weight2 = pow(1.09, date_count)
        # count_weight = (count_weight1 + count_weight2) / (pow(1.09, 16) + pow(1.09, 15))
        count_weight = pow(1.025, date_count - 1)

        # c = 1
        # if year == 2014:
        #     if month == 1:
        #         time_weight = c * pow(1.12, 3)
        #     elif month == 5:
        #         time_weight = c * pow(1.12, 4)
        #     elif month == 6:
        #         time_weight = c * pow(1.12, 5)
        #     elif month == 7:
        #         time_weight = c * pow(1.12, 6)
        # elif year == 2013:
        #     if month == 12:
        #         time_weight = c * pow(1.12, 4)
        #     elif month == 11:
        #         time_weight = c * pow(1.12, 5)
        #     elif month == 10:
        #         time_weight = c * pow(1.12, 6)

        # # time_weight += 1

        is_train = log_df.loc[enrollment_id, 'train']

        if isinstance(is_train, Series):
            is_train = is_train.iloc[0]

        # if is_train:
        #     is_dropout = log_df.loc[enrollment_id, 'dropout']
        #     if isinstance(is_dropout, Series):
        #         is_dropout = is_dropout.iloc[0]

        #     if is_dropout:
        #         class_weight = 1.5
        #     else:
        #         class_weight = 1
        # else:
        #     class_weight = 0

        # weight = (count_weight + time_weight + class_weight) / 3
        weight = count_weight

        sample_weight_df.set_value(enrollment_id, 'weight', weight)
        sample_weight_df.set_value(enrollment_id, 'train', is_train)

    # import pdb; pdb.set_trace()

    sample_weight_df['train'] = sample_weight_df['train'].astype('bool')

    return sample_weight_df

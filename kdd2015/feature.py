# -*- coding: utf-8 -*-
from __future__ import print_function, division

from itertools import groupby, repeat, combinations
from collections import Counter, defaultdict
from multiprocessing import Pool
from subprocess import call
import os.path

# from dateutil import parser
import pandas as pd
from pandas import date_range, DataFrame, Series, IndexSlice, MultiIndex
from numpy import array, vstack, random, zeros, array_split
from keras.preprocessing import sequence
from more_itertools import chunked, flatten, first
# from numba import jit

from .data import load_df
from .analyze import time_bound, fetch_user

COURSE_DIM = 4
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


def append_time_series_features(enrollment_df, log_df):
    log_df.set_index(['enrollment_id', 'date', 'username'], inplace=True)

    enrollment_df = enrollment_df[['date', 'username', 'course_id']]
    enrollment_df.index = enrollment_df.index.droplevel(1)

    records = [(i, enrollment_df.loc[i, 'date'].values, enrollment_df.loc[i, 'username'].iloc[0]) for i in enrollment_df.index.unique()]

    log_df.sortlevel(2, inplace=True)

    del log_df['start_date']
    del log_df['end_date']
    del log_df['object']
    del log_df['train']

    pool = Pool(10)
    partial_event_freq_df = pool.map(
        _extract_enrollment_event_freq,
        zip(records, repeat(log_df)))

    event_freq_df = pd.concat(partial_event_freq_df)
    event_freq_df.reset_index(drop=True, inplace=True)
    enrollment_df.reset_index(inplace=True)
    enrollment_df = enrollment_df.join(event_freq_df)

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


def store_course_embedding():
    call([
        'deepwalk',
        '--input', 'course_edges.txt',
        '--format', 'edgelist',
        '--representation-size', str(COURSE_DIM),
        '--output', 'course_ebd.txt'
    ])


def load_course_embedding():
    graph_mapping = {}
    graph_vectors = []

    if not (os.path.isfile('course_node_mapping.txt') and
            os.path.isfile('course_edges.txt')):
        store_course_edges()

    with open('course_node_mapping.txt') as f:
        for line in f:
            line = line.split()
            i, course_id = int(line[0]), line[1]
            graph_mapping[i] = course_id

    if not os.path.isfile('course_ebd.txt'):
        store_course_embedding()

    with open('course_ebd.txt') as f:
        next(f)
        for line in f:
            line = line.split()
            i = int(line[0])
            vectors = list(map(float, line[1:]))
            graph_vectors.append([graph_mapping[i]] + vectors)

    return graph_vectors


def append_graph_features(feature_df):
    COURSE_DIM = 4
    columns = ['course_id']
    for i in range(COURSE_DIM):
        columns.append('course_%d' % i)

    graph_vector = load_course_embedding()
    # indexes_columns = list(graph_vector.items())
    graph_vector_df = DataFrame(
        graph_vector,
        columns=columns
    )

    # graph_vector_df.reset_index(inplace=True)
    # graph_vector_df['course_id'] = graph_vector_df['index']
    # del graph_vector_df['index']

    feature_df = feature_df.merge(graph_vector_df, how='left', on='course_id')

    return feature_df

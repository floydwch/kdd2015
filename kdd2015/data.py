# -*- coding: utf-8 -*-
# from datetime import datetime
# from functools import lru_cache
# from operator import __add__
from math import log
from datetime import datetime
import os.path
import logging

from pandas import MultiIndex, DataFrame, Series, HDFStore
from pandas.tseries.offsets import DateOffset
# from more_itertools import flatten
import pandas as pd
import numpy as np
from scipy.stats.mstats import zscore
# from treelib import Tree
# from numpy import datetime64
import h5py
# from ipdb import set_trace

logging.basicConfig(filename='data.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_csv():
    log_train = pd.read_csv('log_train.csv', parse_dates=[1])
    log_test = pd.read_csv('log_test.csv', parse_dates=[1])
    enrollment_train = pd.read_csv('enrollment_train.csv')
    enrollment_test = pd.read_csv('enrollment_test.csv')
    truth_df = pd.read_csv('truth_train.csv', header=None)

    log_train['train'] = True
    log_test['train'] = False

    truth_df.columns = ['enrollment_id', 'dropout']

    log_df = pd.concat([log_train, log_test], ignore_index=True)
    # log_df = log_train
    log_df = pd.merge(log_df, truth_df, how='left', on='enrollment_id')
    raw_enrollment_df = pd.concat(
        [enrollment_train, enrollment_test], ignore_index=True)
    # raw_enrollment_df = enrollment_train
    log_df = pd.merge(log_df, raw_enrollment_df, how='left', on='enrollment_id')

    raw_enrollment_df.set_index('enrollment_id', inplace=True)

    log_df['date'] = log_df['time'].map(pd.datetools.normalize_date)

    log_df['source'] = log_df['source'].astype('category')
    log_df['event'] = log_df['event'].astype('category')
    log_df['object'] = log_df['object'].astype('category')
    log_df['username'] = log_df['username'].astype('category')
    log_df['course_id'] = log_df['course_id'].astype('category')

    selected_indices = ['enrollment_id', 'username', 'course_id', 'date']
    log_df.set_index(selected_indices, inplace=True)

    log_df.sortlevel(inplace=True)

    course_groups = log_df.groupby(level='course_id')

    start_dates = course_groups['time'].agg(np.min).map(
        pd.datetools.normalize_date)
    start_dates.name = 'start_date'
    start_dates = DataFrame(start_dates)

    end_dates = course_groups['time'].agg(np.max).map(
        pd.datetools.normalize_date)
    end_dates.name = 'end_date'
    end_dates = DataFrame(end_dates)

    course_ids = course_groups.groups.keys()
    course_df = DataFrame({'course_id': list(course_ids)})

    course_df.set_index('course_id', inplace=True)

    course_df = course_df.merge(
        start_dates, left_index=True, right_index=True)
    course_df = course_df.merge(
        end_dates, left_index=True, right_index=True)

    log_df['start_date'] = start_dates.reindex(
        log_df.index, level='course_id')

    log_df['end_date'] = end_dates.reindex(
        log_df.index, level='course_id')

    enrollment_ids = log_df.index.get_level_values('enrollment_id').unique()
    days = np.array(range(30))

    index = MultiIndex.from_product(
        [enrollment_ids, days],
        names=['enrollment_id', 'day'])

    enrollment_df = DataFrame(index=index)
    enrollment_df = enrollment_df.join(raw_enrollment_df)

    def cal_date(index):
        if not hasattr(cal_date, '_start_dates'):
            cal_date._start_dates = {}

        if not hasattr(cal_date, '_date_offsets'):
            cal_date._date_offsets = [DateOffset(i) for i in range(30)]

        if not hasattr(cal_date, '_course_ids'):
            cal_date._course_ids = {}

        if index[0] not in cal_date._course_ids:
            cal_date._course_ids[index[0]] = \
                enrollment_df.loc[index, 'course_id']

        course_id = cal_date._course_ids[index[0]]

        if course_id not in cal_date._start_dates:
            cal_date._start_dates[course_id] = \
                course_df.loc[course_id, 'start_date']

        start_date = cal_date._start_dates[course_id]
        return start_date + cal_date._date_offsets[index[1]]

    enrollment_df['date'] = enrollment_df.index.map(cal_date)

    object_df = pd.read_csv('object.csv')

    return enrollment_df, truth_df, log_df, course_df, object_df


def calibrate_object(raw_object_df, log_df):
    log_df['course_id'] = log_df.index.get_level_values('course_id')
    object_df = log_df.sortlevel('date')
    object_df = object_df[['course_id', 'event', 'object', 'start_date']]
    object_df.drop_duplicates(inplace=True)

    del log_df['course_id']

    raw_object_df['week'] = [-10000] * raw_object_df.shape[0]

    raw_object_df['start'] = raw_object_df['start'].map(
        lambda x: pd.tslib.NaTType() if x == 'null' else pd.datetools.parse(x))

    raw_object_df.rename(
        columns={'module_id': 'object', 'course_id': 'raw_object_course_id'},
        inplace=True
    )

    object_df['date'] = object_df.index.get_level_values('date')

    object_df['course_id'] = object_df['course_id'].astype('category')

    object_df.set_index('course_id', inplace=True)

    object_df['event'] = object_df['event'].astype('category')

    for course_id in object_df.index.get_level_values('course_id').unique():
        course_object_df = raw_object_df.loc[
            (raw_object_df['raw_object_course_id'] == course_id),
            ['category', 'object', 'children', 'start']
        ]

        course_start = pd.datetools.normalize_date(course_object_df[course_object_df['category'] == 'course']['start'].values[0])

        vertical_df = raw_object_df[raw_object_df['category'] == 'vertical']
        vertical_df.set_index('object', inplace=True)

        chapter_starts = []
        chapter_weeks = []
        for _, chapter in raw_object_df.loc[(raw_object_df['raw_object_course_id'] == course_id) & (raw_object_df['category'] == 'chapter')].iterrows():
            if isinstance(chapter['start'], int) or isinstance(course_start, int):
                import pdb; pdb.set_trace()

            if isinstance(chapter['start'], pd.tslib.NaTType):
                chapter_starts.append(chapter['start'])
                chapter_weeks.append(0)
            else:
                chapter_starts.append(course_start if chapter['start'] < course_start else chapter['start'])
                week = (chapter_starts[-1] - course_start).days // 7
                chapter_weeks.append(week + 1)

        raw_object_df.loc[(raw_object_df['raw_object_course_id'] == course_id) & (raw_object_df['category'] == 'chapter'), 'start'] = chapter_starts
        raw_object_df.loc[(raw_object_df['raw_object_course_id'] == course_id) & (raw_object_df['category'] == 'chapter'), 'week'] = chapter_weeks

        seq_parents = []
        seq_starts = []
        seq_weeks = []
        for _, seq in raw_object_df.loc[(raw_object_df['raw_object_course_id'] == course_id) & (raw_object_df['category'] == 'sequential')].iterrows():
            done = False
            for _, chapter in raw_object_df.loc[(raw_object_df['raw_object_course_id'] == course_id) & (raw_object_df['category'] == 'chapter')].iterrows():
                chapter_children = chapter['children']

                if not isinstance(chapter_children, str):
                    if np.isnan(chapter_children):
                        continue
                    else:
                        logger.debug('chapter %s \'s children are illegal' % chapter['object'][0])

                if seq['object'] in chapter['children']:
                    seq_parents.append(chapter['object'])
                    if not isinstance(seq['start'], pd.tslib.NaTType):
                        seq['start'] = pd.datetools.normalize_date(seq['start'])
                        seq_starts.append(
                            chapter['start'] if seq['start'] < chapter['start']
                            else seq['start']
                        )
                        week = (seq_starts[-1] - course_start).days // 7
                        seq_weeks.append(week + 1)
                    else:
                        seq_starts.append(seq['start'])
                        seq_weeks.append(0)
                        logger.info('seq %s \'s time is nat' % seq['object'])
                    done = True
                    break
            if not done:
                seq_parents.append(None)
                seq_starts.append(pd.tslib.NaTType())
                seq_weeks.append(0)
                logger.info('seq %s has no parent' % seq['object'])

        try:
            assert len(seq_parents) == raw_object_df.loc[(raw_object_df['raw_object_course_id'] == course_id) & (raw_object_df['category'] == 'sequential')].shape[0]
        except:
            import pdb; pdb.set_trace()

        raw_object_df.loc[(raw_object_df['raw_object_course_id'] == course_id) & (raw_object_df['category'] == 'sequential'), 'start'] = seq_starts
        raw_object_df.loc[(raw_object_df['raw_object_course_id'] == course_id) & (raw_object_df['category'] == 'sequential'), 'week'] = seq_weeks

        video_parents = []
        video_starts = []
        video_weeks = []

        for _, video in raw_object_df.loc[(raw_object_df['raw_object_course_id'] == course_id) & (raw_object_df['category'] == 'video')].iterrows():
            done = False

            for _, seq in raw_object_df.loc[(raw_object_df['raw_object_course_id'] == course_id) & (raw_object_df['category'] == 'sequential')].iterrows():
                seq_children = seq['children']

                if not isinstance(seq_children, str):
                    if np.isnan(seq_children):
                        continue
                    else:
                        logger.debug('seq %s \'s children are illegal' % seq['object'][0])

                vertical_ids = seq_children.split()
                for vertical_id in vertical_ids:

                    try:
                        children = vertical_df.loc[vertical_id, 'children']
                    except KeyError:
                        logger.debug('vertical %s not in the object.csv' % vertical_id)
                        continue

                    if isinstance(children, str):
                        pass
                    elif isinstance(children, Series):
                        children = children.iloc[0]
                    else:
                        assert np.isnan(children)
                        continue

                    if video['object'] in children:
                        video_parents.append(vertical_id)
                        video_starts.append(seq['start'])
                        if not isinstance(video_starts[-1], pd.tslib.NaTType):
                            week = (video_starts[-1] - course_start).days // 7
                            video_weeks.append(week + 1)
                        else:
                            video_weeks.append(0)
                        done = True
                        break

                if done:
                    break

            if not done:
                video_parents.append(None)
                video_starts.append(pd.tslib.NaTType())
                video_weeks.append(0)
                logger.info('video %s has no parent' % video['object'])

        assert len(video_parents) == raw_object_df.loc[(raw_object_df['raw_object_course_id'] == course_id) & (raw_object_df['category'] == 'video')].shape[0]

        raw_object_df.loc[(raw_object_df['raw_object_course_id'] == course_id) & (raw_object_df['category'] == 'video'), 'start'] = video_starts
        raw_object_df.loc[(raw_object_df['raw_object_course_id'] == course_id) & (raw_object_df['category'] == 'video'), 'week'] = video_weeks

        problem_parents = []
        problem_starts = []
        problem_weeks = []

        for _, problem in raw_object_df.loc[(raw_object_df['raw_object_course_id'] == course_id) & (raw_object_df['category'] == 'problem')].iterrows():
            done = False

            for _, seq in raw_object_df.loc[(raw_object_df['raw_object_course_id'] == course_id) & (raw_object_df['category'] == 'sequential')].iterrows():

                seq_children = seq['children']

                if not isinstance(seq_children, str):
                    if np.isnan(seq_children):
                        continue
                    else:
                        logger.debug('seq %s \'s children are illegal' % seq['object'][0])

                vertical_ids = seq_children.split()

                for vertical_id in vertical_ids:

                    try:
                        children = vertical_df.loc[vertical_id, 'children']
                    except KeyError:
                        logger.debug('vertical %s not in the object.csv' % vertical_id)
                        continue

                    if isinstance(children, str):
                        pass
                    elif isinstance(children, Series):
                        children = children.iloc[0]
                    else:
                        assert np.isnan(children)
                        continue

                    if problem['object'] in children:
                        problem_parents.append(vertical_id)
                        problem_starts.append(seq['start'])
                        if not isinstance(problem_starts[-1], pd.tslib.NaTType):
                            week = (problem_starts[-1] - course_start).days // 7
                            problem_weeks.append(week + 1)
                        else:
                            problem_weeks.append(0)
                        done = True
                        break

                if done:
                    break

            if not done:
                problem_parents.append(None)
                problem_starts.append(pd.tslib.NaTType())
                problem_weeks.append(0)
                logger.info('problem %s has no parent' % problem['object'])


        assert len(problem_parents) == raw_object_df.loc[(raw_object_df['raw_object_course_id'] == course_id) & (raw_object_df['category'] == 'problem')].shape[0]

        raw_object_df.loc[(raw_object_df['raw_object_course_id'] == course_id) & (raw_object_df['category'] == 'problem'), 'start'] = problem_starts
        raw_object_df.loc[(raw_object_df['raw_object_course_id'] == course_id) & (raw_object_df['category'] == 'problem'), 'week'] = problem_weeks

    object_df = object_df.merge(raw_object_df, how='left', on='object')

    object_df['start'] = object_df['start'].map(pd.to_datetime)

    object_df['object'] = object_df['object'].astype('category')
    object_df['category'] = object_df['category'].astype('category')

    return object_df


def load_df():
    if not (os.path.isfile('enrollment_df.pickle') and
            os.path.isfile('truth_df.pickle') and
            os.path.isfile('log_df.pickle') and
            os.path.isfile('course_df.pickle') and
            os.path.isfile('object_df.pickle')):

        enrollment_df, truth_df, log_df, course_df, object_df = load_csv()

        object_df = calibrate_object(object_df, log_df)

        enrollment_df.to_pickle('enrollment_df.pickle')
        truth_df.to_pickle('truth_df.pickle')
        log_df.to_pickle('log_df.pickle')
        course_df.to_pickle('course_df.pickle')
        object_df.to_pickle('object_df.pickle')

    else:
        enrollment_df = pd.read_pickle('enrollment_df.pickle')
        truth_df = pd.read_pickle('truth_df.pickle')
        log_df = pd.read_pickle('log_df.pickle')
        course_df = pd.read_pickle('course_df.pickle')
        object_df = pd.read_pickle('object_df.pickle')

    return enrollment_df, truth_df, log_df, course_df, object_df


def to_submission(result):
    submission = pd.read_csv('sampleSubmission.csv', header=None)
    submission[1] = result
    now = datetime.now()
    submission.to_csv('%d-%d:%d:%d-submission.csv' % (
        now.month, now.day, now.hour, now.minute),
        header=None, index=False
    )


def load_feature(extend):
    from .feature import extract_time_series_features, extract_enrollment_features, cal_sample_weight, extend_features, append_course_repr2, append_user_repr

    with HDFStore('features.h5') as feature_store:

        if 'enrollment_ids' not in feature_store:
            enrollment_df, truth_df, log_df, course_df, object_df = load_df()

            # enrollment_ids = np.sort(np.random.choice(
            #         enrollment_df.index.get_level_values('enrollment_id').unique(),
            #         12000, False))
            # import pdb; pdb.set_trace()

            enrollment_ids = enrollment_df.index.get_level_values('enrollment_id').unique()

            feature_store['enrollment_ids'] = Series(enrollment_ids)

        else:
            enrollment_ids = feature_store['enrollment_ids'].values

        if 'time_series_feature_df' not in feature_store:
            enrollment_df, truth_df, log_df, course_df, object_df = load_df()

            del course_df
            del truth_df

            if not (os.path.isfile('log_object_df.pickle')):

                log_df.reset_index(inplace=True)
                object_df.drop_duplicates('object', inplace=True)

                log_df = log_df.merge(
                    object_df[['object', 'category', 'week']],
                    how='left', on='object')

                log_df['object'] = log_df['object'].astype('category')
                log_df['category'] = log_df['category'].astype('category')
                log_df['event'] = log_df['event'].astype('category')
                log_df['source'] = log_df['source'].astype('category')

                log_df['event'].cat.add_categories([
                    'access_server',
                    'access_browser',
                    'problem_server',
                    'problem_browser'
                ], inplace=True)

                del object_df

                selected_indices = [
                    'enrollment_id', 'username', 'course_id', 'date'
                ]
                log_df.set_index(selected_indices, inplace=True)

                log_df.to_pickle('log_object_df.pickle')
            else:
                del object_df
                del log_df

                log_df = pd.read_pickle('log_object_df.pickle')

            log_df.reset_index(inplace=True)

            # import pdb; pdb.set_trace()
            time_series_feature_df = extract_time_series_features(
                enrollment_df, log_df, enrollment_ids)

            # import pdb; pdb.set_trace()

            feature_store['time_series_feature_df'] = time_series_feature_df
        else:
            time_series_feature_df = feature_store['time_series_feature_df']

        print('done')

    with HDFStore('features.h5') as feature_store:
        if 'enrollment_feature_df' not in feature_store:
            enrollment_df, truth_df, log_df, course_df, object_df = load_df()

            if not (os.path.isfile('log_object_df.pickle')):

                log_df.reset_index(inplace=True)
                object_df.drop_duplicates('object', inplace=True)

                log_df = log_df.merge(
                    object_df[['object', 'category', 'week']],
                    how='left', on='object')

                log_df['object'] = log_df['object'].astype('category')
                log_df['category'] = log_df['category'].astype('category')
                log_df['event'] = log_df['event'].astype('category')
                log_df['source'] = log_df['source'].astype('category')

                log_df['event'].cat.add_categories([
                    'access_server',
                    'access_browser',
                    'problem_server',
                    'problem_browser'
                ], inplace=True)

                del object_df

                selected_indices = [
                    'enrollment_id', 'username', 'course_id', 'date'
                ]
                log_df.set_index(selected_indices, inplace=True)

                log_df.to_pickle('log_object_df.pickle')
            else:
                del object_df
                del log_df

                log_df = pd.read_pickle('log_object_df.pickle')

            log_df.reset_index(inplace=True)

            enrollment_feature_df = extract_enrollment_features(
                log_df, course_df, truth_df, enrollment_df, enrollment_ids)

            feature_store['enrollment_feature_df'] = enrollment_feature_df
        else:
            enrollment_feature_df = feature_store['enrollment_feature_df']

        if extend == 1:
            enrollment_df, truth_df, log_df, course_df, object_df = load_df()
            enrollment_feature_df = extend_features(enrollment_feature_df, enrollment_df, log_df)
            feature_store['enrollment_feature_df'] = enrollment_feature_df

        if extend == 2:
            enrollment_df, truth_df, log_df, course_df, object_df = load_df()
            feature_store['old_enrollment_feature_df'] = enrollment_feature_df
            enrollment_feature_df = append_course_repr2(enrollment_feature_df, log_df)
            feature_store['enrollment_feature_df'] = enrollment_feature_df

        if extend == 3:
            enrollment_df, truth_df, log_df, course_df, object_df = load_df()
            feature_store['old_old_enrollment_feature_df'] = enrollment_feature_df
            enrollment_feature_df = append_user_repr(enrollment_feature_df, log_df)
            feature_store['enrollment_feature_df'] = enrollment_feature_df

    with HDFStore('features.h5') as feature_store:
        if 'sample_weight_df' not in feature_store:
            enrollment_df, truth_df, log_df, course_df, object_df = load_df()
            sample_weight_df = cal_sample_weight(log_df, enrollment_ids)
            feature_store['sample_weight_df'] = sample_weight_df
        else:
            sample_weight_df = feature_store['sample_weight_df']

    enrollment_df, truth_df, log_df, course_df, object_df = load_df()
    left_df = enrollment_feature_df.reset_index()
    # left_df.rename(columns={b'enrollment_id': 'enrollment_id'}, inplace=True)
    analysis_df = pd.merge(left_df, truth_df, on='enrollment_id')

    # enrollment_feature_df = extend_features(enrollment_feature_df)

    return time_series_feature_df, enrollment_feature_df, enrollment_ids, analysis_df, sample_weight_df


def load_data():
    if not os.path.isfile('data.h5'):
        enrollment_df, truth_df, log_df, course_df, object_df = load_df()
        time_series_feature_df, enrollment_feature_df, enrollment_ids, analysis_df, sample_weight_df = load_feature(False)

        print('feature loaded')

        # time_series_feature_df.set_index('enrollment_id', inplace=True)
        truth_df.set_index('enrollment_id', inplace=True)

        # import ipdb; ipdb.set_trace()

        normal_enrollment_feature_df = enrollment_feature_df.apply(zscore, axis=0)

        # time_series_feature_truth_df = time_series_feature_df.join(truth_df)
        enrollment_feature_truth_df = enrollment_feature_df.join(truth_df)

        # time_series_feature_truth_df = time_series_feature_df.join(truth_df)
        normal_enrollment_feature_truth_df = normal_enrollment_feature_df.join(truth_df)

        # import ipdb; ipdb.set_trace()

        del time_series_feature_df
        del enrollment_feature_df

        print('feature_truth joined')

        # train_time_series_df = time_series_feature_truth_df.dropna()
        # test_time_series_df = time_series_feature_truth_df[
            # time_series_feature_truth_df['dropout'].isnull()
        # ]

        train_enrollment_df = enrollment_feature_truth_df.dropna()
        test_enrollment_df = enrollment_feature_truth_df[
            enrollment_feature_truth_df['dropout'].isnull()
        ]

        train_normal_enrollment_df = normal_enrollment_feature_truth_df.dropna()
        test_normal_enrollment_df = normal_enrollment_feature_truth_df[
            normal_enrollment_feature_truth_df['dropout'].isnull()
        ]

        # del time_series_feature_truth_df
        # del enrollment_feature_truth_df

        masked_time_series_features = [
            # 'access_server',
            # 'access_browser',
            # 'discussion',
            # 'nagivate',
            # 'page_close',
            # 'problem_server',
            # 'problem_browser',
            # 'video',
            # 'wiki',
            # 'normal_access_server',
            # 'normal_access_browser',
            # 'normal_discussion',
            # 'normal_nagivate',
            # 'normal_page_close',
            # 'normal_problem_server',
            # 'normal_problem_browser',
            # 'normal_video',
            # 'normal_wiki',
            # 'z0',
            # 'z1',
            # 'z2',
            # 'z3',
            # 'z4',
            # 'z5',
            # 'hour_count',
            # 'longest_cont_hours',
            # 'act_ratio',
            # 'login_hours',
            # 'z0_total',
            # 'z1_total',
            # 'z2_total',
            # 'z3_total',
            # 'z4_total',
            # 'z5_total',
            # 'total_course',
            # 'chapter_0',
            # 'chapter_1',
            # 'chapter_2',
            # 'chapter_3',
            # 'chapter_4',
            # 'chapter_5',
            # 'chapter_6',
            # 'sequential_0',
            # 'sequential_1',
            # 'sequential_2',
            # 'sequential_3',
            # 'sequential_4',
            # 'sequential_5',
            # 'sequential_6',
            # 'video_0',
            # 'video_1',
            # 'video_2',
            # 'video_3',
            # 'video_4',
            # 'video_5',
            # 'video_6',
            # 'problem_0',
            # 'problem_1',
            # 'problem_2',
            # 'problem_3',
            # 'problem_4',
            # 'problem_5',
            # 'problem_6',
            'first_day',
            'last_day'
        ]

        masked_enrollment_features = [
            # 'course_0',
            # 'course_1',
            # 'course_2',
            # 'course_3',
            # 'first_day',
            # 'last_day',
        ]

        # for column in ['username', 'course_id', 'date', 'dropout'] + \
        #         masked_time_series_features:

            # del train_time_series_df[column]
            # del test_time_series_df[column]

        for column in ['dropout'] + masked_enrollment_features:
            del train_enrollment_df[column]
            del test_enrollment_df[column]

        # import pdb; pdb.set_trace()

        # x_time_series_train = np.array(
        #     np.split(
        #         train_time_series_df.values,
        #         len(train_time_series_df.index.unique())
        #     )
        # )
        # del train_time_series_df

        # import pdb; pdb.set_trace()

        x_enrollment_train = train_enrollment_df.values
        del train_enrollment_df

        x_normal_enrollment_train = train_normal_enrollment_df.values
        del train_normal_enrollment_df

        # x_train[:, :, :9] = np.vectorize(log)(x_train[:, :, :9] + 1)

        # x_train = np.array([reduce(__add__, x_train[:, i:i + 5]) for i in range(0, x_train.shape[0], 5)])

        # squeeze_timestep = 15

        # x_train = (x_train[:, 0::3] + x_train[:, 1::3] + x_train[:, 2::3]) / 3 #+ x_train[:, 3::5] + x_train[:, 4::5]
        # x_train = x_train[:, 20:]

        # squeezed_x_train = x_train[:, :squeeze_timestep]
        # for i in range(squeeze_timestep, 30, squeeze_timestep):
        #     squeezed_x_train += x_train[:, i:i+squeeze_timestep]

        # x_time_series_test = np.array(
        #     np.split(
        #         test_time_series_df.values,
        #         len(test_time_series_df.index.unique())
        #     )
        # )
        # del test_time_series_df

        x_enrollment_test = test_enrollment_df.values
        del test_enrollment_df

        x_normal_enrollment_test = test_normal_enrollment_df.values
        del test_normal_enrollment_df

        # x_test[:, :, :9] = np.vectorize(log)(x_test[:, :, :9] + 1)

        # x_test = (x_test[:, 0::3] + x_test[:, 1::3] + x_test[:, 2::3]) / 3 #+ x_test[:, 3::5] + x_test[:, 4::5]

        # x_test = x_test[:, 20:]
        # squeezed_x_test = x_test[:, :squeeze_timestep]
        # for i in range(squeeze_timestep, 30, squeeze_timestep):
        #     squeezed_x_test += x_test[:, i:i+squeeze_timestep]
        # x_test = np.array([reduce(__add__, x_test[:, i:i + 5]) for i in range(0, x_test.shape[0], 5)])

        # x_train = squeezed_x_train
        # x_test = squeezed_x_test

        # y_train = truth_df.values.flatten()[:x_enrollment_train.shape[0]]
        y_train = truth_df['dropout'].values

        del truth_df

        with h5py.File('data.h5', 'w') as h5f:
            print('write to h5')
            # h5f.create_dataset('x_time_series_train', data=x_time_series_train)
            # h5f.create_dataset('x_time_series_test', data=x_time_series_test)
            h5f.create_dataset('x_enrollment_train', data=x_enrollment_train)
            h5f.create_dataset('x_enrollment_test', data=x_enrollment_test)
            h5f.create_dataset('x_normal_enrollment_train', data=x_normal_enrollment_train)
            h5f.create_dataset('x_normal_enrollment_test', data=x_normal_enrollment_test)
            h5f.create_dataset('y_train', data=y_train)
            h5f.create_dataset('enrollment_ids', data=enrollment_ids)
            print('done')
    else:
        time_series_feature_df, enrollment_feature_df, enrollment_ids, analysis_df, sample_weight_df = load_feature(False)

        with h5py.File('data.h5', 'r') as h5f:
            # x_time_series_train = h5f['x_time_series_train'][:]
            # x_time_series_test = h5f['x_time_series_test'][:]
            x_enrollment_train = h5f['x_enrollment_train'][:]
            x_enrollment_test = h5f['x_enrollment_test'][:]
            x_normal_enrollment_train = h5f['x_normal_enrollment_train'][:]
            x_normal_enrollment_test = h5f['x_normal_enrollment_test'][:]
            y_train = h5f['y_train'][:]
            enrollment_ids = h5f['enrollment_ids'][:]

    # return x_time_series_train, x_enrollment_train, y_train, x_time_series_test, x_enrollment_test
    return x_enrollment_train, x_normal_enrollment_train, x_enrollment_test, x_normal_enrollment_test, y_train, enrollment_ids, sample_weight_df


def clear_features(feature_set):
    with HDFStore('features.h5') as feature_store:
        if feature_set == 'time_series':
            del feature_store['time_series_feature_df']
        elif feature_set == 'enrollment':
            del feature_store['enrollment_feature_df']
        elif feature_set == 'enrollment_ids':
            del feature_store['enrollment_ids']
        elif feature_set == 'sample_weight_df':
            del feature_store['sample_weight_df']
        else:
            raise NotImplementedError()

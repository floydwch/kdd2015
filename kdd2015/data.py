# -*- coding: utf-8 -*-
# from datetime import datetime
# from functools import lru_cache
# from operator import __add__
from datetime import datetime
import os.path

from pandas import MultiIndex, DataFrame
from pandas.tseries.offsets import DateOffset
import pandas as pd
import numpy as np
# from numpy import datetime64
import h5py
# from ipdb import set_trace

from .feature import df2array, append_features


# @lru_cache(maxsize=None)
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
    log_df = pd.merge(log_df, truth_df, how='left', on='enrollment_id')
    raw_enrollment_df = pd.concat(
        [enrollment_train, enrollment_test], ignore_index=True)
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
                course_df.loc[course_id]['start_date']

        start_date = cal_date._start_dates[course_id]
        return start_date + cal_date._date_offsets[index[1]]

    enrollment_df['date'] = enrollment_df.index.map(cal_date)

    log_df['course_id'] = log_df.index.get_level_values('course_id')
    object_df = log_df.sortlevel('date')
    object_df = object_df[['course_id', 'event', 'object']]
    object_df.drop_duplicates(inplace=True)

    del log_df['course_id']

    raw_object_df = pd.read_csv('object.csv')

    raw_object_df.rename(
        columns={'module_id': 'object', 'course_id': 'raw_object_course_id'},
        inplace=True
    )

    # raw_object_df.set_index('module_id', inplace=True)

    object_df = object_df.merge(raw_object_df, how='left', on='object')

    return enrollment_df, truth_df, log_df, course_df, object_df


def load_raw():
    if not (os.path.isfile('enrollment_df.pickle') and
            os.path.isfile('truth_df.pickle') and
            os.path.isfile('log_df.pickle') and
            os.path.isfile('course_df.pickle') and
            os.path.isfile('object_df.pickle')):

        enrollment_df, truth_df, log_df, course_df, object_df = load_csv()

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


def load_feature():
    if not os.path.isfile('feature_df.pickle'):
        # if not (os.path.isfile('enrollment_df.pickle') or
        #         os.path.isfile('truth_df.pickle') or
        #         os.path.isfile('log_df.pickle') or
        #         os.path.isfile('course_df.pickle')):

            # enrollment_df, truth_df, log_df, course_df, object_df = load_csv()

        # else:
        enrollment_df, truth_df, log_df, course_df, object_df = load_raw()

        feature_df = append_features(enrollment_df, log_df)
        feature_df.to_pickle('feature_df.pickle')

    else:
        feature_df = pd.read_pickle('feature_df.pickle')

    return feature_df


def load_data():
    if not os.path.isfile('data.h5'):
        enrollment_df, truth_df, log_df, course_df = load_raw()
        feature_df = load_feature()

        feature_df.set_index('enrollment_id', inplace=True)
        truth_df.set_index('enrollment_id', inplace=True)
        feature_truth_df = feature_df.join(truth_df)

        train_df = feature_truth_df.dropna()
        test_df = feature_truth_df[feature_truth_df['dropout'].isnull()]

        masked_features = [
            # 'access',
            # 'discussion',
            # 'nagivate',
            # 'page_close',
            # 'problem',
            # 'video',
            # 'wiki',
            # 'normal_access',
            # 'normal_discussion',
            # 'normal_nagivate',
            # 'normal_page_close',
            # 'normal_problem',
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
            'first_day',  # couse the training failed
            'last_day'
        ]

        for column in ['username', 'date', 'dropout'] + masked_features:
            del train_df[column]
            del test_df[column]


        x_train = np.array(np.split(train_df.values, len(train_df.index.unique())))
        # x_train = np.array([reduce(__add__, x_train[:, i:i + 5]) for i in range(0, x_train.shape[0], 5)])

        # squeeze_timestep = 15

        # x_train = (x_train[:, 0::3] + x_train[:, 1::3] + x_train[:, 2::3]) / 3 #+ x_train[:, 3::5] + x_train[:, 4::5]
        # x_train = x_train[:, 20:]

        # squeezed_x_train = x_train[:, :squeeze_timestep]
        # for i in range(squeeze_timestep, 30, squeeze_timestep):
        #     squeezed_x_train += x_train[:, i:i+squeeze_timestep]

        x_test = np.array(np.split(test_df.values, len(test_df.index.unique())))

        # x_test = (x_test[:, 0::3] + x_test[:, 1::3] + x_test[:, 2::3]) / 3 #+ x_test[:, 3::5] + x_test[:, 4::5]

        # x_test = x_test[:, 20:]
        # squeezed_x_test = x_test[:, :squeeze_timestep]
        # for i in range(squeeze_timestep, 30, squeeze_timestep):
        #     squeezed_x_test += x_test[:, i:i+squeeze_timestep]
        # x_test = np.array([reduce(__add__, x_test[:, i:i + 5]) for i in range(0, x_test.shape[0], 5)])

        # x_train = squeezed_x_train
        # x_test = squeezed_x_test

        y_train = truth_df.values.flatten()

        # set_trace()

        # x_train, y_train, x_test = df2array(log_df)

        with h5py.File('data.h5', 'w') as h5f:
            h5f.create_dataset('x_train', data=x_train)
            h5f.create_dataset('y_train', data=y_train)
            h5f.create_dataset('x_test', data=x_test)
    else:
        with h5py.File('data.h5', 'r') as h5f:
            x_train = h5f['x_train'][:]
            y_train = h5f['y_train'][:]
            x_test = h5f['x_test'][:]

    return x_train, y_train, x_test

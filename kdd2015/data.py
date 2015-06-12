# -*- coding: utf-8 -*-
# from datetime import datetime
# from functools import lru_cache
from datetime import datetime
import os.path

from pandas import MultiIndex, DataFrame
from pandas.tseries.offsets import DateOffset
import pandas as pd
import numpy as np
# from numpy import datetime64
import h5py
from ipdb import set_trace

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

    enrollment_df = append_features(enrollment_df, log_df, course_df)

    enrollment_df.to_pickle('enrollment_df.pickle')
    truth_df.to_pickle('truth_df.pickle')
    log_df.to_pickle('log_df.pickle')
    course_df.to_pickle('course_df.pickle')

    return enrollment_df, truth_df, log_df, course_df


def load_pickle():
    enrollment_df = pd.read_pickle('enrollment_df.pickle')
    truth_df = pd.read_pickle('truth_df.pickle')
    log_df = pd.read_pickle('log_df.pickle')
    course_df = pd.read_pickle('course_df.pickle')
    return enrollment_df, truth_df, log_df, course_df


def to_submission(result):
    submission = pd.read_csv('sampleSubmission.csv', header=None)
    submission[1] = result
    now = datetime.now()
    submission.to_csv('%d-%d:%d:%d-submission.csv' % (
        now.month, now.day, now.hour, now.minute),
        header=None, index=False
    )


def load_data():
    if not (os.path.isfile('log_df.pickle') or
            os.path.isfile('truth_df.pickle') or
            os.path.isfile('enrollment_df.pickle')):
        enrollment_df, truth_df, log_df, course_df = load_csv()
    else:
        enrollment_df, truth_df, log_df, course_df = load_pickle()

    if not os.path.isfile('data.h5'):
        x_train, y_train, x_test = df2array(log_df)

        with h5py.File('data.h5', 'w') as h5f:
            h5f.create_dataset('x_train', data=x_train)
            h5f.create_dataset('y_train', data=y_train)
            h5f.create_dataset('x_test', data=x_test)
    else:
        with h5py.File('data.h5', 'r') as h5f:
            x_train = h5f['x_train'][:]
            y_train = h5f['y_train'][:]
            x_test = h5f['x_test'][:]

    return log_df, truth_df, x_train, y_train, x_test

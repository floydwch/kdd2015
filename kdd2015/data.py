# -*- coding: utf-8 -*-
# from datetime import datetime
# from functools import lru_cache
from datetime import datetime
import os.path

from pandas import MultiIndex, DataFrame
import pandas as pd
import numpy as np
# from numpy import datetime64
import h5py
from ipdb import set_trace

from .feature import df2array


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

    logs_df = pd.concat([log_train, log_test], ignore_index=True)
    logs_df = pd.merge(logs_df, truth_df, how='left', on='enrollment_id')
    enrollments_df = pd.concat(
        [enrollment_train, enrollment_test], ignore_index=True)
    logs_df = pd.merge(logs_df, enrollments_df, how='left', on='enrollment_id')

    del enrollments_df

    logs_df['date'] = logs_df['time'].map(pd.datetools.normalize_date)

    logs_df['source'] = logs_df['source'].astype('category')
    logs_df['event'] = logs_df['event'].astype('category')
    logs_df['object'] = logs_df['object'].astype('category')
    logs_df['username'] = logs_df['username'].astype('category')
    logs_df['course_id'] = logs_df['course_id'].astype('category')

    selected_indices = ['enrollment_id', 'username', 'course_id', 'date']
    logs_df = logs_df.set_index(selected_indices)

    logs_df.sortlevel(inplace=True)

    course_groups = logs_df.groupby(level='course_id')

    start_dates = course_groups['time'].agg(np.min).map(
        pd.datetools.normalize_date)
    start_dates.name = 'start_dates'
    start_dates = DataFrame(start_dates)

    end_dates = course_groups['time'].agg(np.max).map(
        pd.datetools.normalize_date)
    end_dates.name = 'end_dates'
    end_dates = DataFrame(end_dates)

    logs_df['start_date'] = start_dates.reindex(
        logs_df.index, level='course_id')

    logs_df['end_date'] = end_dates.reindex(
        logs_df.index, level='course_id')

    enrollment_ids = logs_df.index.get_level_values('enrollment_id').unique()
    days = np.array(range(30))

    index = MultiIndex.from_product(
        [enrollment_ids, days],
        names=['enrollment_id', 'day'])

    enrollments_df = DataFrame(index=index)

    logs_df.to_pickle('logs_df.pickle')
    truth_df.to_pickle('truth_df.pickle')
    enrollments_df.to_pickle('enrollments_df.pickle')

    return logs_df, truth_df, enrollments_df


def load_pickle():
    logs_df = pd.read_pickle('logs_df.pickle')
    truth_df = pd.read_pickle('truth_df.pickle')
    enrollments_df = pd.read_pickle('enrollments_df.pickle')
    return logs_df, truth_df, enrollments_df


def to_submission(result):
    submission = pd.read_csv('sampleSubmission.csv', header=None)
    submission[1] = result
    now = datetime.now()
    submission.to_csv('%d-%d:%d:%d-submission.csv' % (
        now.month, now.day, now.hour, now.minute),
        header=None, index=False
    )


def load_data():
    if not (os.path.isfile('logs_df.pickle') or
            os.path.isfile('truth_df.pickle') or
            os.path.isfile('enrollments_df.pickle')):
        logs_df, truth_df, enrollments_df = load_csv()
    else:
        logs_df, truth_df, enrollments_df = load_pickle()

    if not os.path.isfile('data.h5'):
        x_train, y_train, x_test = df2array(logs_df)

        with h5py.File('data.h5', 'w') as h5f:
            h5f.create_dataset('x_train', data=x_train)
            h5f.create_dataset('y_train', data=y_train)
            h5f.create_dataset('x_test', data=x_test)
    else:
        with h5py.File('data.h5', 'r') as h5f:
            x_train = h5f['x_train'][:]
            y_train = h5f['y_train'][:]
            x_test = h5f['x_test'][:]

    return logs_df, truth_df, x_train, y_train, x_test

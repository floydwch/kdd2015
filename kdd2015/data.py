# -*- coding: utf-8 -*-
# from datetime import datetime
# from functools import lru_cache
from datetime import datetime
import os.path

import pandas as pd
# from numpy import datetime64
import h5py

from .feature import df2array


# @lru_cache(maxsize=None)
def load_csv():
    log_train = pd.read_csv('log_train.csv', parse_dates=[1])
    log_test = pd.read_csv('log_test.csv', parse_dates=[1])
    enrollment_train = pd.read_csv('enrollment_train.csv')
    enrollment_test = pd.read_csv('enrollment_test.csv')
    df_ans = pd.read_csv('truth_train.csv', header=None)

    log_train['train'] = True
    log_test['train'] = False

    df_ans.columns = ['enrollment_id', 'dropout']
    # df_ans = df_ans.set_index('enrollment_id')

    df = pd.concat([log_train, log_test], ignore_index=True)
    df = pd.merge(df, df_ans, how='left', on='enrollment_id')
    enrollments = pd.concat(
        [enrollment_train, enrollment_test], ignore_index=True)
    df = pd.merge(df, enrollments, how='left', on='enrollment_id')
    # df = pd.merge(df, enrollments, on='enrollment_id')

    return df, df_ans, enrollments


def to_submission(result):
    submission = pd.read_csv('sampleSubmission.csv', header=None)
    submission[1] = result
    now = datetime.now()
    submission.to_csv('%d-%d:%d:%d-submission.csv' % (
        now.month, now.day, now.hour, now.minute),
        header=None, index=False
    )


def load_data():
    df, df_ans, enrollments = load_csv()

    if not os.path.isfile('data.h5'):
        x_train, y_train, x_test = df2array(df, df_ans, enrollments)

        with h5py.File('data.h5', 'w') as h5f:
            h5f.create_dataset('x_train', data=x_train)
            h5f.create_dataset('y_train', data=y_train)
            h5f.create_dataset('x_test', data=x_test)
    else:
        with h5py.File('data.h5', 'r') as h5f:
            x_train = h5f['x_train'][:]
            y_train = h5f['y_train'][:]
            x_test = h5f['x_test'][:]

    return df, df_ans, x_train, y_train, x_test

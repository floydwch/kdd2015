# -*- coding: utf-8 -*-
# from ptpdb import set_trace
# from dateutil import parser
# import pandas as pd
import numpy as np


def fetch_enrollment(logs_df, enrollment_id):
    return logs_df[logs_df['enrollment_id'] == enrollment_id]


def fetch_user(logs_df, username):
    # enrollments_df = enrollments_df.set_index('username')
    # enrollments = enrollments_df.loc[username]['enrollment_id']
    # logs_df = logs_df.set_index('enrollment_id')
    # logs_df = logs_df.loc[enrollments]
    return logs_df[logs_df['username'] == username]
    # set_trace()


def time_bound(logs_df, enrollments_df, course_id):
    # set_trace()
    enrollments_df = enrollments_df.set_index('course_id')
    enrollments = enrollments_df.loc[course_id]['enrollment_id'].values
    times = logs_df[logs_df['enrollment_id'].isin(enrollments)]['time']
    # set_trace()
    # import pdb; pdb.set_trace()
    start_time = times.min().normalize()
    end_time = times.max().normalize()
    # if course_id == '81UZtt1JJwBFYMj5u38WNKCSVA4IJSDv':
        # set_trace()
    return start_time, end_time


def all_time_bound():
    from .data import load_csv

    logs_df, ans_df, enrollments_df = load_csv()
    enrollments_df = enrollments_df.set_index('enrollment_id', False)
    enrollments = logs_df.groupby('enrollment_id').groups

    skip = set()

    for enrollment_id in enrollments:
        course_id = enrollments_df.loc[enrollment_id]['course_id']

        if course_id in skip:
            continue
        else:
            skip.add(course_id)

        start_time, end_time = time_bound(logs_df, enrollments_df, course_id)
        print(course_id, start_time, end_time, (end_time - start_time).days)

    print('len:', len(skip))


def errors(x_train, y_train, predicts, index, on):
    from .data import load_df
    enrollment_df, truth_df, log_df, course_df, object_df = load_df()

    abs_errors = np.abs(y_train - predicts)
    uncertain_errors = 1 - np.abs(abs_errors - 0.5)
    print('uncertain_errors(top 30) on %s' % on)
    print(uncertain_errors.argsort()[::-1][:30])
    print('abs_errors(top 30) on %s' % on)
    print(abs_errors.argsort()[::-1][:30])

    import pdb; pdb.set_trace()
    # truth_df.iloc[120]

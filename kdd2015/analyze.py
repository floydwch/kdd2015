# -*- coding: utf-8 -*-
# from ptpdb import set_trace
# from dateutil import parser


def fetch(logs_df, enrollment_id):
    enrollments = logs_df.groupby('enrollment_id').groups
    return logs_df.iloc[enrollments[enrollment_id]]


def time_bound(logs_df, enrollments_df, course_id):
    # set_trace()
    enrollments_df = enrollments_df.set_index('course_id')
    enrollments = enrollments_df.loc[course_id]['enrollment_id'].values
    times = logs_df[logs_df['enrollment_id'].isin(enrollments)]['time']
    # set_trace()
    # import pdb; pdb.set_trace()
    start_time = times.min().normalize()
    end_time = times.max().normalize()
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

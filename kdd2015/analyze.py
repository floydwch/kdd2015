# -*- coding: utf-8 -*-
# from ptpdb import set_trace
from dateutil import parser


def fetch(logs_df, enrollment_id):
    enrollments = logs_df.groupby('enrollment_id').groups
    return logs_df.iloc[enrollments[enrollment_id]]


def time_bound(logs_df, enrollments_df, course_id):
    # set_trace()
    enrollments_df = enrollments_df.set_index('course_id', False)
    enrollments = enrollments_df.loc[course_id]['enrollment_id'].values
    times = logs_df[logs_df['enrollment_id'].isin(enrollments)]['time'].map(
        lambda x: parser.parse(x[:10]))
    start_time = times.min()
    end_time = times.max()
    return start_time, end_time

# -*- coding: utf-8 -*-


def fetch(df, enrollment_id):
    enrollments = df.groupby('enrollment_id').groups
    return df.iloc[enrollments[enrollment_id]]

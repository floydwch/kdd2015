# -*- coding: utf-8 -*-


def indexing(df):
    index_items = df['event'].value_counts().index.tolist()
    index_map = dict(map(reversed, enumerate(index_items)))

    df['event'] = df['event'].map(index_map)

    return df

# -*- coding: utf-8 -*-
import gc
import pandas as pd

gc.disable()


pd.set_option('display.max_colwidth', 50)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', None)

setting = {
    'multiprocessing': False,
    'njob': 32
}

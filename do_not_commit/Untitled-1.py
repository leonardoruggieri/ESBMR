# %%
import pandas as pd
import numpy as np
import time
from functools import partial
import multiprocessing
from importlib import reload
# Custom modules
import multiprocessing_functions
# %%


def make_this(df: np.array) -> np.array:
    tic = time.time()
    df[:] = df[:] + 1
    toc = time.time()
    print(f"Time elapsed: {toc-tic}")
    return df


# %%
reload(multiprocessing_functions)
df = np.zeros(shape=(50, 50))
# %%
df = pd.DataFrame(df)
# %%
make_this(df)
# %%
reload(multiprocessing_functions)


def make_this_multiproc(df,
                        n_proc=multiprocessing.cpu_count()):

    df = multiprocessing_functions.multi_func_df(df=df,
                                                 func=make_this,
                                                 n_procs=n_proc)
    return df


# %%
make_this_multiproc(df)

# %%


def func(df):
    return df
# %%


def func_multi(df,
               n_proc=multiprocessing.cpu_count()):

    df = multiprocessing_functions.multi_func_df(df=df,
                                                 func=func,
                                                 n_procs=n_proc)
    return df


# %%
df
# %%
func_multi(df)

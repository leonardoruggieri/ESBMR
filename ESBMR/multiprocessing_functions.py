#########################################################################################################
# Env setting
#########################################################################################################

# Libraries
import pandas as pd
import numpy as np
import time 

from multiprocessing import Pool
from functools import partial


#########################################################################################################
# Multiprocessing Functions
#########################################################################################################

# def start_process():
#     print('Starting', multiprocessing.current_process().name)


def multi_func_list(raw_list,
                    func,
                    n_procs):
    """Multiprocessing application of a function to each element of a list

    Parameters
    ----------
    raw_list : list
        original list to transform
    func : function
        function to apply
    n_procs : int
        number of machine's processors

    Returns
    -------
    processed_list : list
        processed list
    """
    pool = Pool(processes=n_procs)  # , initializer=start_process)
    processed_list = pool.map(func, [i for i in raw_list])
    pool.close()
    pool.terminate()
    pool.join()

    return processed_list


def multi_func_df(df: pd.DataFrame,
                  func,
                  n_procs) -> pd.DataFrame:
    """Multiprocessing application of a function to each row of a pd.DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        original pd.DataFrame
    func : function
        function to apply
    n_procs : int
        number of machine's processors

    Returns
    -------
    processed_df : pd.DataFrame
        processed pd.DataFrame
    """
    split_df = np.array_split(df, n_procs)

    pool = Pool(processes=n_procs)
    processed_df = pd.concat(pool.map(func, split_df))
    pool.close()
    pool.join()
    return processed_df


def multi_func_np(df: np.array,
                  func,
                  n_procs) -> np.array:
    """Multiprocessing application of a function to each row of a np.array

    Parameters
    ----------
    df : np.array
        original np.array
    func : function
        function to apply
    n_procs : int
        number of machine's processors

    Returns
    -------
    processed_df : np.array
        processed np.array
    """
    split_df = np.array_split(df, n_procs)
    pool = Pool(processes=n_procs)
    
    processed_df = np.concatenate((pool.map(func, split_df)), axis = 0)
    
    pool.close()
    pool.join()

    return processed_df

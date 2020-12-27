import pandas as pd
import numpy as np


def make_bins(df: pd.DataFrame,
              q=[0, .75, .85, .89, .95, 1]) -> pd.DataFrame:
    """[summary]

    Parameters
    ----------
    df : pd.DataFrame
        Transactions matrix.
    q : list, optional
        Quantiles to make bins, by default [0, .75, .85, .89, .95, 1]

    Returns
    -------
    df : pd.DataFrame
        Transactions matrix with an additional column for bins.
    """

    df['bins'] = pd.qcut(df['conteggio'], q=q, labels=False,
                         precision=0)
    df['bins'] = df['bins'] + 1

    return df


def make_bins_adjacency(df: np.array,
                        q=[0, .75, .85, .89, .95, 1]) -> np.array:
    """[summary]

    Parameters
    ----------
    df : np.array
        Adjacency matrix.
    q : list, optional
        Quantiles to make bins, by default [0, .75, .85, .89, .95, 1]

    Returns
    -------
    df : np.array
        Adjacency matrix with an additional column for bins.
    """

    dim = df.shape
    df = df.flatten()

    df = pd.DataFrame(df)

    df['bins'] = 0
    df['bins'][df.iloc[:, 0] > 0] = pd.qcut(df.iloc[:, 0], q=q, labels=False,
                                            precision=0)
    df['bins'][df['bins'] > 0] = df['bins'] + 1

    df = np.array(df)[:, 1].reshape(dim)

    df = df.astype(int)

    return df


def cross_tab_bins(df: pd.DataFrame):
    """
    Returns an explicit feedback adjacency matrix startng from the transactions dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The transaction dataframe

    Returns
    -------
    expl : pd.DataFrame
        The adjacency matrix, in explicit feedback format.
    """

    expl = pd.crosstab(df['id_car'], df['nm_nome_cleaned'], df['bins'], aggfunc='first').fillna(0)
    expl = np.array(expl)

    return expl


def cross_tab(df: pd.DataFrame):
    """
    Returns an explicit feedback adjacency matrix startng from the transactions dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The transaction dataframe

    Returns
    -------
    expl : pd.DataFrame
        The adjacency matrix, in explicit feedback format.
    """

    expl = pd.crosstab(df['id_car'], df['nm_nome_cleaned'], df['conteggio'], aggfunc='first').fillna(0)
    
    id_car = expl.index.values  # saving the id_car strings for later use
    nm_nome_cleaned = expl.columns.values  # saving the merhant name strings for later use
    
    expl = np.array(expl)

    return expl, id_car, nm_nome_cleaned


def cross_tab_pd(df: pd.DataFrame):
    """
    Returns an explicit feedback adjacency matrix startng from the transactions dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The transaction dataframe

    Returns
    -------
    expl : pd.DataFrame
        The adjacency matrix, in explicit feedback format.
    """

    expl = pd.crosstab(df['id_car'], df['nm_nome_cleaned'], df['conteggio'], aggfunc='first').fillna(0)

    return expl


def adjacency_matrix(df):
    '''
    It builds a count-type adjacency matrix starting from single user/item interactions
    ---------
    Parameter:
    df : numpy.array (2D)
    '''
    row_name = np.unique(df[:, 0], return_index=False)
    col_name = np.unique(df[:, 1], return_index=False)

    y = np.zeros(shape=(np.unique(df[:, 0]).shape[0], np.unique(df[:, 1]).shape[0]))

    for it in range(df.shape[0]):
        y[np.argwhere(row_name == df[it, 0])[0][0], np.argwhere(col_name == df[it, 1])[0][0]] = 1

    return y

#########################################################################################################
# Env setting
#########################################################################################################

# Libraries
import os
import yaml
import re
import unidecode
import pandas as pd
from typing import List
from functools import partial
from itertools import groupby
import warnings
warnings.filterwarnings('ignore')

# Import custom scripts
import feat_eng
import multiprocessing_functions

# Load config variables
conf_path = os.path.join(os.getcwd(), "..", "environment", "conf.yml")
with open(conf_path, "r", encoding="utf-8") as config_file:
    cfg = yaml.load(config_file, Loader=yaml.FullLoader)

# Set parameters
MIN_WORD_LENGTH = 2
MAX_CONSECUTIVE_CHARS = 4


#########################################################################################################
# NOME_INSEGNA CLEANING FUNCTIONS
#########################################################################################################


def remove_special_char(x: str) -> str:
    """Remove special chars, like punctuation (except ".")

    Parameters
    ----------
    x : str
        Input string

    Returns
    -------
    str
        Output string, without special chars
    """
    return re.sub("[^. \w]", " ", unidecode.unidecode(str(x).replace("_", " ").upper()))


def remove_numbers(x: str) -> str:
    """Remove numbers

    Parameters
    ----------
    x : str
        Input string

    Returns
    -------
    str
        Output string, without numbers
    """
    return "".join([i for i in x if not i.isdigit()])


def remove_short_words(x: str) -> str:
    """Remove words that are shorter than MIN_WORD_LENGTH

    Parameters
    ----------
    x : str
        Input string

    Returns
    -------
    str
        Output string, without words that are shorter than MIN_WORD_LENGTH
    """
    return " ".join([word for word in x.split(" ") if len(word) > MIN_WORD_LENGTH])


def remove_consecutive_char_words(x: str) -> str:
    """Remove words containg more than MAX_CONSECUTIVE_CHARS consecutive chars

    Parameters
    ----------
    x : str
        Input string

    Returns
    -------
    str
        Output string, without words containg more than MAX_CONSECUTIVE_CHARS consecutive chars
    """
    groups = (len(list(values)) for _, values in groupby(x))
    if all(y < MAX_CONSECUTIVE_CHARS for y in groups):
        return x
    else:
        return ""


def remove_words_from_blacklist(x: str,
                                custom_black_list: List[str] = []) -> str:
    """Remove words from blacklist

    Parameters
    ----------
    x : str
        Input string
    custom_black_list : List[str], optional
        custom blacklist of words to remove from nome_insegna, to be added to the cfg blacklist, by default []
    Returns
    -------
    str
        Output string, without words from blacklist
    """
    return " ".join([word for word in x.split(" ")
                     if word.replace(".", "") not in cfg["LAKA_BLACKLIST"] + custom_black_list])


def remove_multi_dots(x: str) -> str:
    """Remove multi-dots

    Parameters
    ----------
    x : str
        Input string

    Returns
    -------
    str
        Output string, without multi-dots
    """
    return re.sub('\.+', '.', x.replace(". ", "").replace(" .", ""))


def remove_spaces(x: str) -> str:
    """Remove multiple blank spaces and strip (spaces and dots) string

    Parameters
    ----------
    x : str
        Input string

    Returns
    -------
    str
        Output string, without multiple blank spaces and strip (spaces and dots) string
    """
    return re.sub(' +', ' ', x.strip().strip("."))


def clean_str(x: str,
              custom_black_list: List[str] = []) -> str:
    """Clean string

        Steps:
        1. Remove special chars, like punctuation (except ".")
        2. Remove numbers (they are not likely to be laka keywords, they are likely to be codes)
        3. Remove words that are shorter than MIN_WORD_LENGTH
        4. Remove words containg more than MAX_CONSECUTIVE_CHARS consecutive chars (they are likely to be codes)
        5. Remove words from blacklist
        6. Remove multi-dots
        7. Remove multiple blank spaces and strip (spaces and dots) string

    Parameters
    ----------
    x : str
        input str to be cleaned
    custom_black_list : List[str], optional
        custom blacklist of words to remove from nome_insegna, to be added to the cfg blacklist, by default []

    Returns
    -------
    str
        cleaned string
    """

    cleaned_x = remove_special_char(x=x)                                            # 1. Remove special chars
    cleaned_x = remove_numbers(x=cleaned_x)                                         # 2. Remove words containing numbers
    cleaned_x = remove_short_words(x=cleaned_x)                                     # 3. Remove short words
    cleaned_x = remove_consecutive_char_words(x=cleaned_x)                          # 4. Remove consecutive char words
    cleaned_x = remove_words_from_blacklist(x=cleaned_x,
                                            custom_black_list=custom_black_list)    # 5. Remove words from blacklist
    cleaned_x = remove_multi_dots(x=cleaned_x)                                      # 6. Remove multi-dots
    cleaned_x = remove_spaces(x=cleaned_x)                                          # 7. Remove blank spaces and strip
    cleaned_x = remove_short_words(x=cleaned_x)                                     # 3. Remove short words

    return cleaned_x


def clean_insegna_df(df: pd.DataFrame,
                     nome_insegna: str,
                     custom_black_list: List[str] = []) -> pd.DataFrame:
    """Clean nome_insegna

    Parameters
    ----------
    df : pd.DataFrame
        input pd.DataFrame
    nome_insegna : str
        column name of insegna name
    custom_black_list : List[str], optional
        custom blacklist of words to remove from nome_insegna, to be added to the cfg blacklist, by default []

    Returns
    -------
    pd.DataFrame
        input pd.DataFrame, with 1 more columns:
        nome_insegna_cleaned : nome_insegna after cleaning phase
    """

    clean_str_partial = partial(clean_str,
                                custom_black_list=custom_black_list)

    df.loc[~df[nome_insegna].isnull(), nome_insegna + "_cleaned"] = df[nome_insegna].apply(clean_str_partial)
    df[nome_insegna + "_cleaned"].fillna("", inplace=True)
    return df


def clean_insegna(df: pd.DataFrame,
                  nome_insegna: str,
                  custom_black_list: List[str] = [],
                  n_procs: int = 1) -> pd.DataFrame:
    """Clean nome_insegna, in a multiprocessing fashion

    Parameters
    ----------
    df : pd.DataFrame
        input pd.DataFrame
    nome_insegna : str
        column name of insegna name
    custom_black_list : List[str], optional
        custom blacklist of words to remove from nome_insegna, to be added to the cfg blacklist, by default []
    n_procs : int, optional
        number of cores, by default 1

    Returns
    -------
    pd.DataFrame
        input pd.DataFrame, with 1 more columns:
        nome_insegna_cleaned : nome_insegna after cleaning phase
    """

    partial_clean_insegna_df = partial(clean_insegna_df,
                                       custom_black_list=custom_black_list,
                                       nome_insegna=nome_insegna)
    cleaned_df = multiprocessing_functions.multi_func_df(df=df,
                                                         func=partial_clean_insegna_df,
                                                         n_procs=n_procs)
    return cleaned_df

''' Load de novo sequence data '''

import pandas as pd
import numpy as np
import re

import postnovo.config as config
import postnovo.utils as utils

#import config
#import utils

from collections import OrderedDict
from os.path import basename
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


def load_files():

    alg_basename_dfs_dict = OrderedDict()

    if config.novor_files:
        alg_basename_dfs_dict['novor'] = OrderedDict.fromkeys(
            [basename(novor_file) for novor_file in config.novor_files])
        for i, novor_file in enumerate(config.novor_files):
            utils.verbose_print('loading', basename(novor_file))
            check_file_fragment_mass_tol(novor_file, config.frag_mass_tols[i])
            if i == 0:
                find_precursor_mass_tol(novor_file)

            alg_basename_dfs_dict['novor'][basename(novor_file)] = load_novor_file(novor_file)
    
    if config.peaks_files:
        alg_basename_dfs_dict['peaks'] = OrderedDict.fromkeys(
            [basename(peaks_file) for peaks_file in config.peaks_files])
        for i, peaks_file in enumerate(config.peaks_files):
            utils.verbose_print('loading', basename(peaks_file))
            alg_basename_dfs_dict['peaks'][basename(peaks_file)] = load_peaks_file(peaks_file)

    if config.pn_files:
        alg_basename_dfs_dict['pn'] = OrderedDict.fromkeys(
            [basename(pn_file) for pn_file in config.pn_files])
        for i, pn_file in enumerate(config.pn_files):
            utils.verbose_print('loading', basename(pn_file))
            alg_basename_dfs_dict['pn'][basename(pn_file)] = load_pn_file(pn_file)

    utils.verbose_print('cleaning up input data')
    alg_basename_dfs_dict = filter_shared_scans(alg_basename_dfs_dict)

    return alg_basename_dfs_dict

def load_novor_file(novor_file):
    
    novor_df = pd.read_csv(novor_file, skiprows = 20, index_col = False)

    novor_df.dropna(axis = 1, how = 'all', inplace = True)
    novor_df.columns = [name.strip() for name in novor_df.columns]
    novor_df.drop(['# id', 'err(data-denovo)', 'score'], axis = 1, inplace = True)

    novor_df.columns = ['scan', 'retention time', 'm/z',
                        'charge', 'novor seq mass',
                        'seq mass error', 'seq',
                        'aa score']

    novor_df['rank'] = 0
    new_col_order = [novor_df.columns[0]] + [novor_df.columns[-1]] + novor_df.columns[1:-1].tolist()
    novor_df = novor_df[new_col_order]

    novor_df['seq'] = novor_df['seq'].str.strip()
    novor_df['aa score'] = novor_df['aa score'].str.strip()
    novor_df[['scan', 'retention time', 'm/z',
              'charge', 'novor seq mass',
              'seq mass error']] = novor_df[[
                  'scan', 'retention time',
                  'm/z', 'charge', 'novor seq mass',
                  'seq mass error']].apply(pd.to_numeric)

    novor_df['retention time'] /= config.seconds_in_min

    novor_df['novor seq mass'] = (novor_df['novor seq mass'] +
                                  config.proton_mass * novor_df['charge'])
    
    novor_df['seq'] = novor_df['seq'].apply(
        lambda seq: config.novor_seq_sub_fn(string = seq))

    novor_df['aa score'] = novor_df['aa score'].apply(
        lambda score_string: score_string.split('-')).apply(
            lambda score_string_list: list(map(int, score_string_list)))
    novor_df['avg aa score'] = novor_df['aa score'].apply(
        lambda score_list: sum(score_list) / len(score_list))
    #novor_df['aa score'] = novor_df['aa score'].apply(
    #    lambda score_list: ' '.join(score_list))

    novor_df.set_index(['scan', 'rank'], inplace = True)

    return novor_df

def check_file_fragment_mass_tol(novor_file, user_mass_tol):

    file_mass_tol = pd.read_csv(novor_file, nrows = 12).iloc[11][0]
    if user_mass_tol not in file_mass_tol:
        raise AssertionError(
            'Order of mass tol args does not correspond to order of Novor files')

def find_precursor_mass_tol(novor_file):
    precursor_mass_tol_info_str = pd.read_csv(novor_file, nrows = 13).iloc[12][0]
    try:
        config.precursor_mass_tol[0] = float(re.search('(?<=# precursorErrorTol = )(.*)(?=ppm)', precursor_mass_tol_info_str).group(0))
    except ValueError:
        pass

def load_peaks_file(peaks_file):
    pass

def load_pn_file(pn_file):
    
    col_names = ['rank', 'rank score', 'pn score', 'n-gap', 'c-gap', '[m+h]', 'charge', 'seq']
    pn_df = pd.read_csv(pn_file, sep = '\t', names = col_names, comment = '#')

    pn_df = pn_df[~(pn_df['rank'].shift(-1).str.contains('>>') & pn_df['rank'].str.contains('>>'))]
    nonnumeric_scan_col = pd.to_numeric(
        pn_df['rank'], errors = 'coerce').apply(np.isnan)
    pn_df['retained rows'] = (((nonnumeric_scan_col.shift(-1) - nonnumeric_scan_col) == -1)
                              | (nonnumeric_scan_col == False))
    pn_df.drop(pn_df[pn_df['retained rows'] == False].index, inplace = True)
    pn_df.drop('retained rows', axis = 1, inplace = True)
    
    pn_df.reset_index(drop = True, inplace = True)
    pn_df['group'] = np.nan
    nonnumeric_scan_col = pd.to_numeric(
        pn_df['rank'], errors = 'coerce').apply(np.isnan)
    nonnumeric_indices = pn_df['rank'][nonnumeric_scan_col].index.tolist()
    pn_df['group'][nonnumeric_indices] = pn_df['group'][nonnumeric_indices].index
    pn_df['group'].fillna(method = 'ffill', inplace = True)

    grouped = pn_df.groupby('group')
    scan_start_substr = 'scans: "'
    scan_end_substr = '" \('
    try:
        pn_df['scan'] = grouped['rank'].first().apply(
            lambda str: re.search(
                scan_start_substr + '(.*)' + scan_end_substr, str).group(1))
    except TypeError as e:
        raise Exception('scan # not found on \">>\" header line.\n\
        This is likely because the mgf file used in PN was not generated properly.\n\
        raw -> mgf file conversion in Proteome Discoverer reports scan numbers properly.')
    pn_df['scan'] = pn_df['scan'].apply(float)
    pn_df['scan'].fillna(method = 'ffill', inplace = True)
        
    sqs_start_substr = 'SQS '
    sqs_end_substr = '\)'
    pn_df['sqs'] = grouped['rank'].first().apply(
        lambda str: re.search(
            sqs_start_substr + '(.*)' + sqs_end_substr, str).group(1))
    pn_df['sqs'] = pn_df['sqs'].apply(float)
    pn_df['sqs'].fillna(method = 'ffill', inplace = True)

    pn_df.drop('group', axis = 1, inplace = True)

    pn_df['[m+h]'] = (pn_df['[m+h]'] + (pn_df['charge'] - 1) * config.proton_mass) / pn_df['charge']
    pn_df.rename(columns = {'[m+h]': 'm/z'}, inplace = True)

    pn_df['seq'].replace(
        to_replace = np.nan, value = '', inplace = True)
    pn_df['seq'] = pn_df['seq'].apply(
        lambda seq: config.pn_seq_sub_fn(string = seq))

    pn_df_cols = ['scan', 'rank', 'm/z',
                  'charge', 'n-gap', 'c-gap',
                  'seq', 'rank score', 'pn score',
                  'sqs']

    pn_df = pn_df.reindex_axis(
        sorted(pn_df.columns,
               key = lambda old_col: pn_df_cols.index(old_col)), axis = 1)

    pn_df.drop(grouped['rank'].first().index, inplace = True)

    pn_df[['scan', 'rank']] = pn_df[['scan', 'rank']].astype(int)
    pn_df.set_index(['scan', 'rank'], inplace = True)

    return pn_df

def filter_shared_scans(alg_basename_dfs_dict):

    for tol in config.frag_mass_tols:

        for alg0, alg1 in combinations(config.alg_list, 2):
            if (tol in config.alg_tols_dict[alg0].keys()
                and tol in config.alg_tols_dict[alg1].keys()):

                df_name0 = config.alg_tols_dict[alg0][tol]
                df_name1 = config.alg_tols_dict[alg1][tol]
                df0 = alg_basename_dfs_dict[alg0][df_name0]
                df1 = alg_basename_dfs_dict[alg1][df_name1]

                common = df0.iloc[:, :1].join(
                    df1.iloc[:, :1], lsuffix = '_l', rsuffix = '_r')
                df0 = df0[df0.index.get_level_values(0).isin(
                    common.index.get_level_values(0))]
                df1 = df1[df1.index.get_level_values(0).isin(
                    common.index.get_level_values(0))]

    return alg_basename_dfs_dict
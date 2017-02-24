from config import (proton_mass, seconds_in_min,
                    novor_dropped_chars, pn_dropped_chars,
                    accepted_algs)
from utils import verbose_print

import pandas as pd
import numpy as np
import re

from collections import OrderedDict
from os.path import basename
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

def load_files(user_args):

    alg_list = []

    if 'novor_files' in user_args:
        alg_list.append('novor')
        novor_dfs = OrderedDict.fromkeys(
            [basename(novor_file) for novor_file in user_args['novor_files']])
        for i, novor_file in enumerate(user_args['novor_files']):
            verbose_print('loading', basename(novor_file))

            check_file_mass_tol(novor_file, user_args['novor_tols'][i])
            novor_dfs[basename(novor_file)] = load_novor_file(novor_file)

            novor_tols = OrderedDict(
                zip(user_args['novor_tols'], novor_dfs.keys()))
    else:
        novor_dfs = OrderedDict()
        novor_tols = OrderedDict()
    
    if 'peaks_files' in user_args:
        alg_list.append('peaks')
        peaks_dfs = OrderedDict.fromkeys(
            [basename(peaks_file) for peaks_file in user_args['peaks_files']])
        for i, peaks_file in enumerate(user_args['peaks_files']):
            verbose_print('loading', basename(peaks_file))

            peaks_dfs[basename(peaks_file)] = load_peaks_file(peaks_file)

        peaks_tols = OrderedDict(
            zip(user_args['peaks_tols'], peaks_dfs.keys()))
    else:
        peaks_dfs = OrderedDict()
        peaks_tols = OrderedDict()

    if 'pn_files' in user_args:
        alg_list.append('pn')
        pn_dfs = OrderedDict.fromkeys(
            [basename(pn_file) for pn_file in user_args['pn_files']])
        for i, pn_file in enumerate(user_args['pn_files']):
            verbose_print('loading', basename(pn_file))

            pn_dfs[basename(pn_file)] = load_pn_file(pn_file)

        pn_tols = OrderedDict(
            zip(user_args['pn_tols'], pn_dfs.keys()))

    else:
        pn_dfs = OrderedDict()
        pn_tols = OrderedDict()

    alg_df_name_dict = OrderedDict([('novor', novor_dfs), ('peaks', peaks_dfs), ('pn', pn_dfs)])
    alg_tol_dict = OrderedDict([('novor', novor_tols), ('peaks', peaks_tols), ('pn', pn_tols)])

    verbose_print('cleaning up input data')
    alg_df_name_dict, tol_df_name_dict = filter_shared_scans(alg_df_name_dict, alg_tol_dict)

    return alg_list, alg_df_name_dict, tol_df_name_dict, alg_tol_dict

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

    novor_df['retention time'] /= seconds_in_min

    novor_df['novor seq mass'] = (novor_df['novor seq mass'] +
                                  proton_mass * novor_df['charge'])
    
    novor_df['seq'] = novor_df['seq'].apply(
        lambda seq: seq.translate(novor_dropped_chars))

    novor_df['aa score'] = novor_df['aa score'].apply(
        lambda score_string: score_string.split('-')).apply(
            lambda score_string_list: list(map(int, score_string_list)))
    novor_df['avg aa score'] = novor_df['aa score'].apply(
        lambda score_list: sum(score_list) / len(score_list))
    #novor_df['aa score'] = novor_df['aa score'].apply(
    #    lambda score_list: ' '.join(score_list))

    novor_df.set_index(['scan', 'rank'], inplace = True)

    return novor_df

def check_file_mass_tol(novor_file, user_mass_tol):

    file_mass_tol = pd.read_csv(novor_file, nrows = 12).iloc[11][0]
    if user_mass_tol not in file_mass_tol:
        raise AssertionError(
            'Order of mass tol args does not correspond to order of Novor files')

    return file_mass_tol

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

    pn_df['[m+h]'] = (pn_df['[m+h]'] + (pn_df['charge'] - 1) * proton_mass) / pn_df['charge']
    pn_df.rename(columns = {'[m+h]': 'm/z'}, inplace = True)

    pn_df['seq'].replace(
        to_replace = np.nan, value = '', inplace = True)
    pn_df['seq'] = pn_df['seq'].apply(
        lambda seq: seq.translate(pn_dropped_chars))

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

def filter_shared_scans(alg_df_name_dict, alg_tol_dict):

    tol_list = sorted(list(set(alg_tol_dict['novor'].keys()).union(
        alg_tol_dict['peaks'].keys()).union(
            alg_tol_dict['pn'].keys())))

    tol_df_name_dict = OrderedDict()

    for tol in tol_list:

        tol_df_name_dict[tol] = []

        for alg1, alg2 in combinations(alg_tol_dict.keys(), 2):
            if (tol in alg_tol_dict[alg1].keys()
                and tol in alg_tol_dict[alg2].keys()):

                df_name1 = alg_tol_dict[alg1][tol]
                df_name2 = alg_tol_dict[alg2][tol]
                df1 = alg_df_name_dict[alg1][df_name1]
                df2 = alg_df_name_dict[alg2][df_name2]

                tol_df_name_dict[tol] +=\
                    [(accepted_algs.index(alg1), df_name1)] +\
                    [(accepted_algs.index(alg2), df_name2)]

                common = df1.iloc[:, :1].join(
                    df2.iloc[:, :1], lsuffix = '_l', rsuffix = '_r')
                df1 = df1[df1.index.get_level_values(0).isin(
                    common.index.get_level_values(0))]
                df2 = df2[df2.index.get_level_values(0).isin(
                    common.index.get_level_values(0))]

        tol_df_name_dict[tol] = [df_name for (alg_index, df_name)
                                 in sorted(tol_df_name_dict[tol])]

    return alg_df_name_dict, tol_df_name_dict
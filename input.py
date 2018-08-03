''' Load de novo sequence data '''

import multiprocessing
import numpy as np
import pandas as pd
import re
import sys

from collections import OrderedDict
from functools import partial
from itertools import combinations
from os.path import basename
import warnings
warnings.filterwarnings('ignore')

if 'postnovo' in sys.modules:
    import postnovo.config as config
    import postnovo.utils as utils
else:
    import config
    import utils

def load_files():

    ## input_df_dict format =
    ## OrderedDict('novor': OrderedDict('0.2': df, ..., '0.7': df), 'pn': ..., 'deepnovo': ...)
    input_df_dict = OrderedDict()

    if config.globals['novor_fps']:
        input_df_dict['novor'] = OrderedDict()

        ## Single-threaded
        #for i, novor_file in enumerate(config.globals['novor_fps']):
        #    utils.verbose_print('loading', basename(novor_file))
        #    input_df_dict['novor'][
        #        config.globals['frag_mass_tols'][i]
        #    ] = load_novor_file(novor_file)

        # Multi-threaded
        mp_pool = multiprocessing.Pool(config.globals['cpus'])
        utils.verbose_print('loading Novor files')
        novor_dfs = mp_pool.map(load_novor_file, config.globals['novor_fps'])
        mp_pool.close()
        mp_pool.join()
        for i, novor_df in enumerate(novor_dfs):
            input_df_dict['novor'][
                config.globals['frag_mass_tols'][i]
            ] = novor_df

    if config.globals['pn_fps']:
        input_df_dict['pn'] = OrderedDict()

        ## Single-threaded
        #for i, pn_file in enumerate(config.globals['pn_fps']):
        #    utils.verbose_print('loading', basename(pn_file))
        #    input_df_dict['pn'][
        #        config.globals['frag_mass_tols'][i]
        #    ] = load_pn_file(pn_file)

        # Multi-threaded
        mp_pool = multiprocessing.Pool(config.globals['cpus'])
        utils.verbose_print('loading PepNovo+ files')
        pn_dfs = mp_pool.map(load_pn_file, config.globals['pn_fps'])
        mp_pool.close()
        mp_pool.join()
        for i, pn_df in enumerate(pn_dfs):
            input_df_dict['pn'][
                config.globals['frag_mass_tols'][i]
            ] = pn_df

    if config.globals['deepnovo_fps']:
        input_df_dict['deepnovo'] = OrderedDict()

        ## Single-threaded
        #for i, path in enumerate(config.globals['deepnovo_fps']):
        #    utils.verbose_print('loading', basename(path))
        #    input_df_dict['deepnovo'][config.frag_mass_tols[i]] = load_deepnovo_file(path)

        # Multi-threaded: do not use, since there is currently multithreading within load_deepnovo_file
        mp_pool = multiprocessing.Pool(config.globals['cpus'])
        utils.verbose_print('loading DeepNovo files')
        deepnovo_dfs = mp_pool.map(load_deepnovo_file, config.globals['deepnovo_fps'])
        mp_pool.close()
        mp_pool.join()
        for i, deepnovo_df in enumerate(deepnovo_dfs):
            input_df_dict['deepnovo'][config.frag_mass_tols[i]] = deepnovo_df

    utils.verbose_print('cleaning up input data')
    input_df_dict = filter_shared_scans(input_df_dict)

    return input_df_dict

def load_novor_file(novor_file):
    
    novor_df = pd.read_csv(novor_file, skiprows = 20, index_col = False)

    novor_df.dropna(axis = 1, how = 'all', inplace = True)
    novor_df.columns = [name.strip() for name in novor_df.columns]
    novor_df.drop(['# id', 'err(data-denovo)', 'score'], axis = 1, inplace = True)

    novor_df.columns = [
        'scan', 
        'retention time', 
        'm/z', 
        'charge', 
        'novor seq mass', 
        'seq mass error', 
        'seq', 
        'aa score'
    ]

    novor_df['rank'] = 0
    new_col_order = [novor_df.columns[0]] + [novor_df.columns[-1]] + novor_df.columns[1:-1].tolist()
    novor_df = novor_df[new_col_order]

    novor_df['seq'] = novor_df['seq'].str.strip()
    novor_df['aa score'] = novor_df['aa score'].str.strip()
    novor_df[
        [
            'scan', 
            'retention time', 
            'm/z', 
            'charge', 
            'novor seq mass', 
            'seq mass error'
        ]
    ] = novor_df[
        [
            'scan', 
            'retention time', 
            'm/z', 
            'charge', 
            'novor seq mass', 
            'seq mass error'
        ]
    ].apply(pd.to_numeric)

    novor_df['retention time'] /= config.seconds_in_min

    novor_df['novor seq mass'] = (
        novor_df['novor seq mass'] + config.proton_mass * novor_df['charge']
    )
    
    novor_df['seq'] = novor_df['seq'].apply(
        lambda seq: utils.remove_mod_chars(seq = seq))

    novor_df['aa score'] = novor_df['aa score'].apply(
        lambda score_string: score_string.split('-')).apply(
            lambda score_string_list: list(map(int, score_string_list)))
    novor_df['avg aa score'] = novor_df['aa score'].apply(
        lambda score_list: sum(score_list) / len(score_list))

    novor_df.set_index(['scan', 'rank'], inplace = True)

    return novor_df

def load_pn_file(pn_file):
    
    col_names = [
        'rank', 
        'rank score', 
        'pn score', 
        'n-gap', 
        'c-gap', 
        '[m+h]', 
        'charge', 
        'seq'
    ]
    pn_df = pd.read_csv(pn_file, sep = '\t', names = col_names, comment = '#')

    pn_df = pn_df[~(pn_df['rank'].shift(-1).str.contains('>>') & pn_df['rank'].str.contains('>>'))]
    nonnumeric_scan_col = pd.to_numeric(pn_df['rank'], errors='coerce').apply(np.isnan)
    pn_df['retained rows'] = (
        ((nonnumeric_scan_col.shift(-1) - nonnumeric_scan_col) == -1) 
        | (nonnumeric_scan_col == False)
    )
    pn_df.drop(pn_df[pn_df['retained rows'] == False].index, inplace = True)
    pn_df.drop('retained rows', axis=1, inplace = True)
    
    pn_df.reset_index(drop = True, inplace = True)
    pn_df['group'] = np.nan
    nonnumeric_scan_col = pd.to_numeric(pn_df['rank'], errors='coerce').apply(np.isnan)
    nonnumeric_indices = pn_df['rank'][nonnumeric_scan_col].index.tolist()
    pn_df['group'][nonnumeric_indices] = pn_df['group'][nonnumeric_indices].index
    pn_df['group'].fillna(method = 'ffill', inplace = True)

    # mgf files generated by certain programs include double quotes in spectrum header lines
    # These are removed from the mgf.out headers to allow scan and SQS to be easily extracted
    if '"' in pn_df.ix[0]['rank']:
        pn_df['rank'] = pn_df['rank'].apply(lambda x: x.replace('"', ''))

    grouped = pn_df.groupby('group')
    scan_start_substr = 'scans: '
    scan_end_substr = ' \('
    try:
        pn_df['scan'] = grouped['rank'].first().apply(
            lambda str: re.search(
                scan_start_substr + '([0-9]+)' + scan_end_substr, str).group(1)
            )
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
            sqs_start_substr + '(.*)' + sqs_end_substr, str
        ).group(1)
    )
    pn_df['sqs'] = pn_df['sqs'].apply(float)
    pn_df['sqs'].fillna(method = 'ffill', inplace = True)

    pn_df.drop('group', axis = 1, inplace = True)

    pn_df['[m+h]'] = (pn_df['[m+h]'] + (pn_df['charge'] - 1) * config.proton_mass) \
        / pn_df['charge']
    pn_df.rename(columns = {'[m+h]': 'm/z'}, inplace = True)

    pn_df['seq'].replace(
        to_replace = np.nan, value = '', inplace = True
    )
    pn_df['seq'] = pn_df['seq'].apply(
        lambda seq: utils.remove_mod_chars(seq=seq)
    )

    pn_df_cols = [
        'scan', 
        'rank', 
        'm/z', 
        'charge', 
        'n-gap', 
        'c-gap', 
        'seq', 
        'rank score', 
        'pn score', 
        'sqs'
    ]

    pn_df = pn_df.reindex_axis(
        sorted(
            pn_df.columns, 
            key = lambda old_col: pn_df_cols.index(old_col)
        ), axis = 1
    )

    pn_df.drop(grouped['rank'].first().index, inplace = True)

    pn_df[['scan', 'rank']] = pn_df[['scan', 'rank']].astype(int)
    pn_df.set_index(['scan', 'rank'], inplace = True)

    return pn_df

def load_deepnovo_file(path):
    
    deepnovo_table = pd.read_csv(path, sep='\t', header=0)
    deepnovo_table = deepnovo_table[['scan', 'output_seq', 'AA_probability']]
    deepnovo_table.columns = ['scan', 'seq', 'aa score']

    # Example:
    # 'Mmod,V,D,V,A,Q,H,P,N,I,R' -> 'MmodVDVAQHPNIR' -> 'MVDVAQHPNIR'
    deepnovo_table['seq'] = deepnovo_table['seq'].apply(
        lambda s: ''.join(s.split(',')).replace('mod', '')
        )
    # Example:
    # '0.67 1.0 1.0 1.0 1.0 0.28 1.0 0.1 0.0 0.85' -> [0.67, 1.0, ...]
    deepnovo_table['aa score'] = deepnovo_table['aa score'].apply(
        lambda s: list(map(float, s.split(' ')))
        )
    deepnovo_table['avg aa score'] = deepnovo_table['aa score'].apply(
        lambda s: sum(s) / len(s)
        )

    # deepnovo predicts both Leu and Ile:
    # Consider these residues to be the same and remove redundant lower-ranking seqs
    groupby_deepnovo_table = deepnovo_table.groupby('scan')
    scan_tables = [scan_table.reset_index(drop=True) for _, scan_table in groupby_deepnovo_table]

    # Single-threaded
    dereplicated_scan_tables = []
    for scan_table in scan_tables:
        dereplicated_scan_tables.append(dereplicate_deepnovo_scan_tables(scan_table))

    ## Multiprocessing
    #mp_pool = multiprocessing.Pool(config.globals['cpus'])
    #dereplicated_scan_tables = mp_pool.map(dereplicate_deepnovo_scan_tables, scan_tables)
    #mp_pool.close()
    #mp_pool.join()

    deepnovo_table_dereplicated = pd.concat(dereplicated_scan_tables)

    # Add column of seq rank
    scan_tables = [scan_table.reset_index() for _, scan_table in deepnovo_table_dereplicated.groupby('scan')]
    deepnovo_table_with_ranks = pd.concat(scan_tables).rename(columns={'index': 'rank'})

    deepnovo_table = deepnovo_table_with_ranks[['scan', 'rank', 'seq', 'aa score', 'avg aa score']]
    deepnovo_table['scan'] = deepnovo_table['scan'].apply(int)
    deepnovo_table.set_index(['scan', 'rank'], inplace=True)

    return deepnovo_table

def dereplicate_deepnovo_scan_tables(scan_table):

    scan_table_seqs = scan_table['seq'].tolist()
    scan_table_seqs_all_leu = [seq.replace('I', 'L') for seq in scan_table_seqs]

    # There are occasionally fewer than 20 results per peptide
    retained_rows = list(range(len(scan_table_seqs)))
    for i, seq1 in enumerate(scan_table_seqs_all_leu):
        if retained_rows[i] != -1:
            for j, seq2 in enumerate(scan_table_seqs_all_leu[i + 1:]):
                if retained_rows[j] != -1:
                    if seq1 == seq2:
                        retained_rows[i + j + 1] = -1
    
    scan_table['seq'] = scan_table_seqs_all_leu
    dereplicated_scan_table = scan_table.ix[[i for i in retained_rows if i != -1]]
    dereplicated_scan_table.index = range(len(dereplicated_scan_table))
    return dereplicated_scan_table

def filter_shared_scans(input_df_dict):
    for tol in config.globals['frag_mass_tols']:
        # Get the set of scans retained from the first alg
        common = set(input_df_dict[config.globals['algs'][0]][tol].index.get_level_values(0).tolist())
        # Loop through the other algs and only retain the scans in common
        for alg in config.globals['algs'][1:]:
            common = common.intersection(set(input_df_dict[alg][tol].index.get_level_values(0).tolist()))
        common = list(common)
        for alg in config.globals['algs']:
            input_df = input_df_dict[alg][tol]
            input_df_dict[alg][tol] = input_df[input_df.index.get_level_values(0).isin(common)]

    return input_df_dict
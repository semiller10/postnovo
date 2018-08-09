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
        #    utils.verbose_print('Loading', basename(novor_file))
        #    input_df_dict['novor'][
        #        config.globals['frag_mass_tols'][i]
        #    ] = load_novor_file(novor_file)

        # Multi-threaded
        mp_pool = multiprocessing.Pool(config.globals['cpus'])
        utils.verbose_print('Loading Novor files')
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
        #    utils.verbose_print('Loading', basename(pn_file))
        #    input_df_dict['pn'][
        #        config.globals['frag_mass_tols'][i]
        #    ] = load_pn_file(pn_file)

        # Multi-threaded
        mp_pool = multiprocessing.Pool(config.globals['cpus'])
        utils.verbose_print('Loading PepNovo+ files')
        pn_dfs = mp_pool.map(load_pn_file, config.globals['pn_fps'])
        mp_pool.close()
        mp_pool.join()
        for i, pn_df in enumerate(pn_dfs):
            input_df_dict['pn'][
                config.globals['frag_mass_tols'][i]
            ] = pn_df

    if config.globals['deepnovo_fps']:
        input_df_dict['deepnovo'] = OrderedDict()

        ##Single-threaded
        #for i, path in enumerate(config.globals['deepnovo_fps']):
        #    utils.verbose_print('Loading', basename(path))
        #    input_df_dict['deepnovo'][config.globals['frag_mass_tols'][i]] = load_deepnovo_file(path)

        #Multi-threaded: do not use, since there is currently multithreading within load_deepnovo_file
        mp_pool = multiprocessing.Pool(config.globals['cpus'])
        utils.verbose_print('Loading DeepNovo files')
        deepnovo_dfs = mp_pool.map(load_deepnovo_file, config.globals['deepnovo_fps'])
        mp_pool.close()
        mp_pool.join()
        for i, deepnovo_df in enumerate(deepnovo_dfs):
            input_df_dict['deepnovo'][config.globals['frag_mass_tols'][i]] = deepnovo_df

    utils.verbose_print('Cleaning up input data')
    input_df_dict = filter_shared_spectra(input_df_dict)

    return input_df_dict

def load_novor_file(novor_file):
    
    #Determine the number of file header lines.
    file_header_lines = 0
    with open(novor_file) as handle:
        for i, line in enumerate(handle.readlines()):
            if line[:4] == '# id':
                file_header_lines = i
                break

    novor_df = pd.read_csv(novor_file, skiprows=file_header_lines, index_col=False)

    novor_df.dropna(axis=1, how='all', inplace=True)
    novor_df.columns = [name.strip() for name in novor_df.columns]
    novor_df.drop(['err(data-denovo)', 'score'], axis=1, inplace=True)

    novor_df.columns = [
        'spec_id', 
        'scan', 
        'retention time', 
        'm/z', 
        'charge', 
        'novor seq mass', 
        'seq mass error', 
        'seq', 
        'aa score'
    ]

    #REMOVE
    #spec_ids = []
    #for scan, rt in zip(novor_df['scan'].tolist(), novor_df['retention time'].tolist()):
    #    rounded_rt = round(rt, 1)
    #    try:
    #        spec_ids.append(config.globals['rt_scan_spec_id_dict'][(str(scan), str(rounded_rt))])
    #    except KeyError:
    #        try:
    #            spec_ids.append(config.globals['rt_scan_spec_id_dict'][(str(scan), str(round(rounded_rt + 0.1, 1)))])
    #        except KeyError:
    #            spec_ids.append(config.globals['rt_scan_spec_id_dict'][(str(scan), str(round(rounded_rt - 0.1, 1)))])

    #novor_df['spec_id'] = spec_ids
    #novor_df['retention time'] = novor_df['retention time'].apply(str)

    novor_df['rank'] = 0
    new_col_order = ['spec_id'] + ['rank'] + novor_df.columns[1:-1].tolist()
    novor_df = novor_df[new_col_order]

    novor_df['seq'] = novor_df['seq'].str.strip()
    novor_df['aa score'] = novor_df['aa score'].str.strip()
    novor_df[
        [
            'spec_id', 
            'retention time', 
            'm/z', 
            'charge', 
            'novor seq mass', 
            'seq mass error'
        ]
    ] = novor_df[
        [
            'spec_id', 
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
        lambda seq: utils.remove_mod_chars(seq = seq)
    )

    novor_df['aa score'] = novor_df['aa score'].apply(
        lambda score_str: score_str.split('-')).apply(
            lambda score_strs: list(map(int, score_strs)))
    novor_df['avg aa score'] = novor_df['aa score'].apply(
        lambda scores: sum(scores) / len(scores))

    novor_df.set_index(['spec_id', 'rank'], inplace = True)

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
    pn_df = pd.read_csv(pn_file, sep='\t', names=col_names, comment='#')

    #Remove spectrum blocks without any de novo sequence ids.
    pn_df = pn_df[~(pn_df['rank'].shift(-1).str.contains('>>') & pn_df['rank'].str.contains('>>'))]
    nonnumeric_scan_col = pd.to_numeric(pn_df['rank'], errors='coerce').apply(np.isnan)
    pn_df['retained rows'] = (
        ((nonnumeric_scan_col.shift(-1) - nonnumeric_scan_col) == -1) 
        | (nonnumeric_scan_col == False)
    )
    pn_df.drop(pn_df[pn_df['retained rows'] == False].index, inplace=True)
    pn_df.drop('retained rows', axis=1, inplace=True)
    
    pn_df.reset_index(drop=True, inplace=True)
    pn_df['group'] = np.nan
    nonnumeric_scan_col = pd.to_numeric(pn_df['rank'], errors='coerce').apply(np.isnan)
    nonnumeric_indices = pn_df['rank'][nonnumeric_scan_col].index.tolist()
    pn_df['group'][nonnumeric_indices] = pn_df['group'][nonnumeric_indices].index
    pn_df['group'].fillna(method='ffill', inplace=True)

    #Extract seq ids, scans and SQS values from spectrum block headers.
    grouped = pn_df.groupby('group')
    spectrum_headers = grouped['rank'].first()
    pn_df['spec_id'] = spectrum_headers.apply(
        lambda s: int(s[s.index('"; SpectrumID: "') + 16: s.index('"; scans: "')])
    )
    pn_df['spec_id'].fillna(method='ffill', inplace=True)
    pn_df['scan'] = spectrum_headers.apply(
        lambda s: s[s.index('scans: "') + 8: s.index('" (SQS')]
    )
    pn_df['scan'].fillna(method='ffill', inplace=True)
    pn_df['sqs'] = spectrum_headers.apply(
        lambda s: float(s[s.index('" (SQS') + 6: -1])
    )
    pn_df['sqs'].fillna(method='ffill', inplace=True)
    pn_df.drop('group', axis=1, inplace=True)

    pn_df['[m+h]'] = (pn_df['[m+h]'] + (pn_df['charge'] - 1) * config.proton_mass) \
        / pn_df['charge']
    pn_df.rename(columns={'[m+h]': 'm/z'}, inplace=True)

    pn_df['seq'].replace(
        to_replace=np.nan, value='', inplace=True
    )
    pn_df['seq'] = pn_df['seq'].apply(
        lambda seq: utils.remove_mod_chars(seq=seq)
    )

    pn_df_cols = [
        'spec_id', 
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
            key=lambda old_col: pn_df_cols.index(old_col)
        ), 
        axis=1
    )

    pn_df.drop(grouped['rank'].first().index, inplace=True)

    pn_df[['spec_id', 'rank']] = pn_df[['spec_id', 'rank']].astype(int)
    pn_df.set_index(['spec_id', 'rank'], inplace = True)

    return pn_df

def load_deepnovo_file(path):
    
    deepnovo_df = pd.read_csv(path, sep='\t', header=0, dtype={'scan': str})
    deepnovo_df['spec_id'] = deepnovo_df['scan'].apply(
        lambda s: config.globals['scan_spec_id_dict'][s]
    )
    deepnovo_df = deepnovo_df[['spec_id', 'scan', 'output_seq', 'AA_probability']]
    deepnovo_df.columns = ['spec_id', 'scan', 'seq', 'aa score']

    #Example: 
    #'Mmod,V,D,V,A,Q,H,P,N,I,R' -> 'MmodVDVAQHPNIR' -> 'MVDVAQHPNIR'
    deepnovo_df['seq'] = deepnovo_df['seq'].apply(
        lambda s: ''.join(s.split(',')).replace('mod', '')
    )
    #Example:
    #'0.67 1.0 1.0 1.0 1.0 0.28 1.0 0.1 0.0 0.85' -> [0.67, 1.0, ...]
    deepnovo_df['aa score'] = deepnovo_df['aa score'].apply(
        lambda s: list(map(float, s.split(' ')))
    )
    deepnovo_df['avg aa score'] = deepnovo_df['aa score'].apply(
        lambda s: sum(s) / len(s)
    )

    #DeepNovo predicts both Leu and Ile: 
    #Consider these residues to be the same and remove redundant lower-ranking seqs.
    deepnovo_gb = deepnovo_df.groupby('spec_id')
    spec_dfs = [spec_df.reset_index(drop=True) for _, spec_df in deepnovo_gb]

    #Single-threaded
    derep_spec_dfs = []
    for spec_df in spec_dfs:
        derep_spec_dfs.append(dereplicate_deepnovo_spec_dfs(spec_df))

    ##Multiprocessing: only if DeepNovo file loading is also not multithreaded!
    #mp_pool = multiprocessing.Pool(config.globals['cpus'])
    #derep_spec_dfs = mp_pool.map(dereplicate_deepnovo_spec_dfs, spec_dfs)
    #mp_pool.close()
    #mp_pool.join()

    derep_deepnovo_df = pd.concat(derep_spec_dfs)

    #Add a seq rank column.
    spec_dfs = [spec_df.reset_index() for _, spec_df in derep_deepnovo_df.groupby('spec_id')]
    deepnovo_df_with_ranks = pd.concat(spec_dfs).rename(columns={'index': 'rank'})

    deepnovo_df = deepnovo_df_with_ranks[
        ['spec_id', 'scan', 'rank', 'seq', 'aa score', 'avg aa score']
    ]
    deepnovo_df.set_index(['spec_id', 'rank'], inplace=True)

    return deepnovo_df

def dereplicate_deepnovo_spec_dfs(spec_df):

    spec_seqs = spec_df['seq'].tolist()
    spec_seqs_all_leu = [seq.replace('I', 'L') for seq in spec_seqs]

    #There are occasionally fewer than 20 results per peptide.
    retained_rows = list(range(len(spec_seqs)))
    for i, seq1 in enumerate(spec_seqs_all_leu):
        if retained_rows[i] != -1:
            for j, seq2 in enumerate(spec_seqs_all_leu[i + 1:]):
                if retained_rows[i + j + 1] != -1:
                    if seq1 == seq2:
                        retained_rows[i + j + 1] = -1
    
    spec_df['seq'] = spec_seqs_all_leu
    derep_spec_df = spec_df.iloc[[i for i in retained_rows if i != -1]]
    derep_spec_df.index = range(len(derep_spec_df))
    return derep_spec_df

def filter_shared_spectra(input_df_dict):

    for tol in config.globals['frag_mass_tols']:
        #Get the set of spectra IDs retained from the first de novo alg.
        common = set(
            input_df_dict[config.globals['algs'][0]][tol].index.get_level_values(0).tolist()
        )
        #Retain spectra with predictions from all algs.
        for alg in config.globals['algs'][1:]:
            common = common.intersection(set(
                input_df_dict[alg][tol].index.get_level_values(0).tolist()
            ))
        common = list(common)
        for alg in config.globals['algs']:
            input_df = input_df_dict[alg][tol]
            input_df_dict[alg][tol] = input_df[input_df.index.get_level_values(0).isin(common)]

    return input_df_dict
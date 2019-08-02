''' Load Novor, PepNovo+, and DeepNovo de novo sequence data. '''

#Copyright 2018, Samuel E. Miller. All rights reserved.
#Postnovo is publicly available for non-commercial uses.
#Licensed under GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007.
#See postnovo/LICENSE.txt.

import config
import utils

import multiprocessing
import numpy as np
import pandas as pd
import re
import sys
import time
import warnings

from collections import OrderedDict
from itertools import combinations
from os.path import basename

warnings.filterwarnings('ignore')

def parse():
    '''
    Load de novo sequencing results from multiple tools specified in globals.

    Parameters
    ----------
    None

    Returns
    -------
    None
        DataFrame objects, each corresponding to the output of a de novo sequencing tool 
        run at a fragment mass parameterization, 
        are pickled as (temporary) files in the user-specified output directory.
    '''

    if 'Novor Output Filepaths' in config.globals:
        ##Single process
        #utils.verbose_print('Loading Novor files:')
        #for i, novor_fp in enumerate(config.globals['Novor Output Filepaths']):
        #    utils.verbose_print('Loading', basename(novor_fp))
        #    novor_df = load_novor_file(novor_fp)
        #    utils.save_pkl_objects(
        #        config.globals['Output Directory'], 
        #        **{'Novor.' + config.globals['MGF Filename'] + '.' + 
        #           config.globals['Fragment Mass Tolerances'][i] + '.pkl': novor_df})

        #Multiprocessing
        mp_pool = multiprocessing.Pool(config.globals['CPU Count'])
        utils.verbose_print('Loading Novor files')
        novor_dfs = mp_pool.map(load_novor_file, config.globals['Novor Output Filepaths'])
        mp_pool.close()
        mp_pool.join()
        for i, novor_df in enumerate(novor_dfs):
            utils.save_pkl_objects(
                config.globals['Output Directory'], 
                **{'Novor.' + config.globals['MGF Filename'] + '.' + 
                   config.globals['Fragment Mass Tolerances'][i] + '.pkl': novor_df})

    if 'PepNovo Output Filepaths' in config.globals:
        ##Single process
        #utils.verbose_print('Loading PepNovo+ files:')
        #for i, pn_fp in enumerate(config.globals['PepNovo Output Filepaths']):
        #    utils.verbose_print('Loading', basename(pn_fp))
        #    pn_df = load_pn_file(pn_fp)
        #    utils.save_pkl_objects(
        #        config.globals['Output Directory'], 
        #        **{'PepNovo.' + config.globals['MGF Filename'] + '.' + 
        #           config.globals['Fragment Mass Tolerances'][i] + '.pkl': pn_df})

        #Multiprocessing
        mp_pool = multiprocessing.Pool(config.globals['CPU Count'])
        utils.verbose_print('Loading PepNovo+ files')
        pn_dfs = mp_pool.map(load_pn_file, config.globals['PepNovo Output Filepaths'])
        mp_pool.close()
        mp_pool.join()
        for i, pn_df in enumerate(pn_dfs):
            utils.save_pkl_objects(
                config.globals['Output Directory'], 
                **{'PepNovo.' + config.globals['MGF Filename'] + '.' + 
                    config.globals['Fragment Mass Tolerances'][i] + '.pkl': pn_df})

    if 'DeepNovo Output Filepaths' in config.globals:
        #Single process
        utils.verbose_print('Loading DeepNovo files:')
        for i, deepnovo_fp in enumerate(config.globals['DeepNovo Output Filepaths']):
            utils.verbose_print('Loading', basename(deepnovo_fp))
            deepnovo_df = load_deepnovo_file(deepnovo_fp)
            utils.save_pkl_objects(
                config.globals['Output Directory'], 
                **{'DeepNovo.' + config.globals['MGF Filename'] + '.' + 
                    config.globals['Fragment Mass Tolerances'][i] + '.pkl': deepnovo_df})

        ##Multiprocessing: DO NOT USE, 
        #as there is currently multiprocessing within load_deepnovo_file (causing daemon error).
        #mp_pool = multiprocessing.Pool(config.globals['CPU Count'])
        #utils.verbose_print('Loading DeepNovo files')
        #deepnovo_dfs = mp_pool.map(
        #    load_deepnovo_file, config.globals['DeepNovo Output Filepaths'])
        #mp_pool.close()
        #mp_pool.join()
        #for i, deepnovo_df in enumerate(deepnovo_dfs):
        #    utils.save_pkl_objects(
        #        config.globals['Output Directory'], 
        #        **{'DeepNovo.' + config.globals['MGF Filename'] + '.' + 
        #            config.globals['Fragment Mass Tolerances'][i] + '.pkl': deepnovo_df})


    utils.verbose_print('Cleaning up input data')

    return

def load_novor_file(novor_fp):
    '''
    Load data from a Novor output file as a DataFrame.

    Parameters
    ----------
    novor_fp : str
        Filepath to Novor output file.

    Returns
    -------
    novor_df : pandas DataFrame
        Table of Novor data.
    '''

    #Determine the number of file header lines.
    #Find the identity and order of fixed and variable amino acid modifications: 
    #the order determines the symbol used for each modification in the reported sequences.
    file_header_lines = 0
    with open(novor_fp) as in_f:
        denovogui_fixed_mod_codes = []
        denovogui_variable_mod_codes = []
        for i, line in enumerate(in_f.readlines()):
            if line[:23] == '# fixedModifications = ':
                denovogui_fixed_mod_codes = \
                    line.rstrip().replace('# fixedModifications = ', '').split(', ')
            elif line[:26] == '# variableModifications = ':
                denovogui_variable_mod_codes = \
                    line.rstrip().replace('# variableModifications = ', '').split(', ')
            elif line[:4] == '# id':
                file_header_lines = i
                break

    novor_df = pd.read_csv(novor_fp, skiprows=file_header_lines, index_col=False)

    novor_df.dropna(axis=1, how='all', inplace=True)
    novor_df.columns = [name.strip() for name in novor_df.columns]
    #scanNum is redundant with spec_id due to Postnovo "format_mgf" setup.
    #err(data-denovo) = mz(data) - (pepMass(denovo) + z * mass(H+)) / z
    #This deduces the de novo M/Z value from the neutral de novo sequence mass.
    #ppm(1e6*err/(mz*z)) is dropped in favor of a slightly different value.
    novor_df.drop(
        ['scanNum', 'err(data-denovo)', 'ppm(1e6*err/(mz*z))'], 
        axis=1, 
        inplace=True)

    #Rename the columns.
    novor_df.columns = [
        'Spectrum ID', 
        'Retention Time', 
        'M/Z', 
        'Charge', 
        'Novor Peptide Mass', 
        'Novor Peptide Score', 
        'Sequence', 
        'Novor Amino Acid Scores']

    novor_df['Rank'] = 0
    novor_df['Sequence'] = novor_df['Sequence'].str.strip()
    novor_df['Novor Amino Acid Scores'] = novor_df['Novor Amino Acid Scores'].str.strip()

    #Ensure that TSV data is typed correctly.
    novor_df[[
        'Spectrum ID', 
        'Charge', 
        'Novor Peptide Mass', 
        'Novor Peptide Score']] = \
        novor_df[[
            'Spectrum ID', 
            'Charge', 
            'Novor Peptide Mass', 
            'Novor Peptide Score']].apply(pd.to_numeric)

    #Taking into account different fragment masses, 
    #Novor calculates the mass of the predicted peptide.
    novor_df['De Novo Peptide Ion Mass'] = (
        novor_df['Novor Peptide Mass'] + config.PROTON_MASS * novor_df['Charge'])
    novor_df.drop('Novor Peptide Mass', axis=1, inplace=True)
    #Calculate the error in the (full-length) peptide ion prediction.
    novor_df['De Novo Peptide Ion Mass Error (ppm)'] = \
        (10**6 * 
         (novor_df['De Novo Peptide Ion Mass'] - novor_df['M/Z'] * novor_df['Charge']) / 
         novor_df['De Novo Peptide Ion Mass'])
    novor_df['Sequence'] = novor_df['Sequence'].apply(
        lambda s: replace_mod_chars(s, config.novor_postnovo_mod_dict))
    #Encode de novo sequence amino acids as integers.
    novor_df['Encoded Sequence'] = novor_df['Sequence'].apply(lambda s: utils.encode_aas(s))
    #Record the length of the sequences.
    novor_df['Sequence Length'] = novor_df['Encoded Sequence'].apply(len)

    #Peptide-level score reflects peptide probability,  
    #whereas amino acid scores reflect amino acid probabilities.
    novor_df['Novor Amino Acid Scores'] = novor_df['Novor Amino Acid Scores'].apply(
        lambda score_str: score_str.split('-')).apply(
            lambda score_strs: list(map(int, score_strs)))
    novor_df['Novor Average Amino Acid Score'] = novor_df['Novor Amino Acid Scores'].apply(
        lambda scores: sum(scores) / len(scores))

    novor_df.set_index(['Spectrum ID', 'Rank'], inplace=True)
    #Rearrange the columns.
    novor_df = novor_df.reindex(columns=config.alg_cols_dict['Novor'])
    
    return novor_df

def load_pn_file(pn_fp):
    '''
    Load data from a PepNovo+ output file as a DataFrame.

    Parameters
    ----------
    pn_fp : str
        Filepath to PepNovo+ output file.

    Returns
    -------
    pn_fp : pandas DataFrame
        Table of PepNovo+ data.
    '''
    
    col_names = [
        'Rank', 
        'PepNovo Rank Score', 
        'PepNovo Score', 
        'PepNovo N-terminal Mass Gap', 
        'PepNovo C-terminal Mass Gap', 
        'Measured [M+H]', 
        'Charge', 
        'Sequence']
    pn_df = pd.read_csv(pn_fp, sep='\t', names=col_names, comment='#')

    #Remove spectrum blocks without any de novo sequence IDs.
    pn_df = pn_df[~(pn_df['Rank'].shift(-1).str.contains('>>') & pn_df['Rank'].str.contains('>>'))]
    nonnumeric_scan_col = pd.to_numeric(pn_df['Rank'], errors='coerce').apply(np.isnan)
    pn_df['retained rows'] = (
        ((nonnumeric_scan_col.shift(-1) - nonnumeric_scan_col) == -1) 
        | (nonnumeric_scan_col == False))
    pn_df.drop(pn_df[pn_df['retained rows'] == False].index, inplace=True)
    pn_df.drop('retained rows', axis=1, inplace=True)
    
    pn_df.reset_index(drop=True, inplace=True)
    pn_df['group'] = np.nan
    nonnumeric_scan_col = pd.to_numeric(pn_df['Rank'], errors='coerce').apply(np.isnan)
    nonnumeric_indices = pn_df['Rank'][nonnumeric_scan_col].index.tolist()
    pn_df['group'][nonnumeric_indices] = pn_df['group'][nonnumeric_indices].index
    pn_df['group'].fillna(method='ffill', inplace=True)

    #Extract spectrum IDs and SQS values from spectrum block headers.
    grouped = pn_df.groupby('group')
    spectrum_headers = grouped['Rank'].first()
    pn_df['Spectrum ID'] = spectrum_headers.apply(
        lambda s: int(s[s.index(', Index:') + 8: s.index(', Old index:')]))
    pn_df['Spectrum ID'].fillna(method='ffill', inplace=True)
    pn_df['PepNovo Spectrum Quality Score (SQS)'] = spectrum_headers.apply(
        lambda s: float(s[s.index('(SQS') + 4: -1]))
    pn_df['PepNovo Spectrum Quality Score (SQS)'].fillna(method='ffill', inplace=True)
    pn_df.drop('group', axis=1, inplace=True)

    pn_df['Sequence'].replace(to_replace=np.nan, value='', inplace=True)
    pn_df['Sequence'] = pn_df['Sequence'].apply(
        lambda seq: replace_mod_chars(seq, config.pn_postnovo_mod_dict))
    #Fixed modifications are not explicitly reported by PepNovo+, e.g., C+57 is just C.
    standard_aa_postnovo_fixed_mod_dict = OrderedDict(
        [(config.postnovo_mod_standard_aa_dict[mod], mod) 
         for mod in config.globals['Fixed Modifications']])
    pn_df['Sequence'] = pn_df['Sequence'].apply(
        lambda seq: replace_mod_chars(seq, standard_aa_postnovo_fixed_mod_dict))
    #Encode de novo sequence amino acids as integers.
    pn_df['Encoded Sequence'] = pn_df['Sequence'].apply(lambda s: utils.encode_aas(s))
    #Record the length of the sequences.
    pn_df['Sequence Length'] = pn_df['Encoded Sequence'].apply(len)

    pn_df.drop(grouped['Rank'].first().index, inplace=True)

    #Determine precursor mass and retention time directly from the MGF file.
    pn_df['Retention Time'] = pn_df['Spectrum ID'].apply(
        lambda s: config.mgf_info_dict[s]['Retention Time'])
    pn_df['M/Z'] = pn_df['Spectrum ID'].apply(lambda s: config.mgf_info_dict[s]['M/Z'])
    #The mass error in the sequence prediction is unknown, 
    #as PepNovo+ often predicts a partial-length sequence.

    pn_df[['Spectrum ID', 'Rank', 'Charge']] = pn_df[['Spectrum ID', 'Rank', 'Charge']].astype(int)
    pn_df.set_index(['Spectrum ID', 'Rank'], inplace=True)
    #Rearrange the columns.
    pn_df = pn_df.reindex(columns=config.alg_cols_dict['PepNovo'])

    return pn_df

def load_deepnovo_file(deepnovo_fp):
    '''
    Load data from a DeepNovo output file as a DataFrame.

    Parameters
    ----------
    deepnovo_fp : str
        Filepath to DeepNovo output file.

    Returns
    -------
    deepnovo_fp : pandas DataFrame
        Table of DeepNovo data.
    '''
    
    deepnovo_df = pd.read_csv(deepnovo_fp, sep='\t', header=0)
    deepnovo_df = deepnovo_df[['scan', 'output_seq', 'AA_probability']]
    deepnovo_df.columns = ['Spectrum ID', 'Sequence', 'DeepNovo Amino Acid Scores']

    ##Example: 
    ##'Mmod,V,D,V,A,Q,H,P,N,I,R' -> 'MmodVDVAQHPNIR' -> 'M+15.995VDVAQHPNIR'
    deepnovo_df['Sequence'] = deepnovo_df['Sequence'].apply(lambda s: ''.join(s.split(',')))
    deepnovo_df['Sequence'] = deepnovo_df['Sequence'].apply(
        lambda seq: replace_mod_chars(seq, config.deepnovo_config_postnovo_mod_dict))
    #Very rarely, DeepNovo assigns a modified amino acid at the beginning or end of the sequence 
    #that is not allowed by the parameters file, 
    #e.g., only Cmod and Mmod are allowed, but Nmod and Qmod appear.
    #Remove predictions with unallowed amino acids: 
    #at this point in the process, these are sequences containing "mod".
    deepnovo_df = deepnovo_df[deepnovo_df['Sequence'].apply(lambda s: 'mod' not in s)]
    ##Example:
    ##'0.67 1.0 1.0 1.0 1.0 0.28 1.0 0.1 0.0 0.85' -> [0.67, 1.0, ...]
    deepnovo_df['DeepNovo Amino Acid Scores'] = deepnovo_df['DeepNovo Amino Acid Scores'].apply(
        lambda s: list(map(float, s.split(' '))))
    deepnovo_df['DeepNovo Average Amino Acid Score'] = deepnovo_df[
        'DeepNovo Amino Acid Scores'].apply(lambda s: sum(s) / len(s))

    #DeepNovo predicts both Leu and Ile: 
    #Consider these residues to be the same and remove redundant lower-ranking seqs.
    deepnovo_gb = deepnovo_df.groupby('Spectrum ID')
    spec_dfs = [spec_df.reset_index(drop=True) for _, spec_df in deepnovo_gb]

    ##Single process
    #derep_spec_dfs = []
    #for spec_df in spec_dfs:
    #    derep_spec_dfs.append(dereplicate_deepnovo_spec_dfs(spec_df))

    #Multiprocessing: only if function call is not also multithreaded!
    mp_pool = multiprocessing.Pool(config.globals['CPU Count'])
    derep_spec_dfs = mp_pool.map(dereplicate_deepnovo_spec_dfs, spec_dfs)
    mp_pool.close()
    mp_pool.join()

    derep_deepnovo_df = pd.concat(derep_spec_dfs)

    #Add a sequence rank column.
    spec_dfs = [spec_df.reset_index() for _, spec_df in derep_deepnovo_df.groupby('Spectrum ID')]
    deepnovo_df_with_ranks = pd.concat(spec_dfs).rename(columns={'index': 'Rank'})
    deepnovo_df = deepnovo_df_with_ranks[[
        'Spectrum ID', 
        'Rank', 
        'Sequence', 
        'DeepNovo Amino Acid Scores', 
        'DeepNovo Average Amino Acid Score']]

    #Determine precursor mass and retention time directly from the MGF file.
    deepnovo_df['Retention Time'] = deepnovo_df['Spectrum ID'].apply(
        lambda s: config.mgf_info_dict[s]['Retention Time'])
    deepnovo_df['M/Z'] = deepnovo_df['Spectrum ID'].apply(lambda s: config.mgf_info_dict[s]['M/Z'])
    deepnovo_df['Charge'] = deepnovo_df['Spectrum ID'].apply(
        lambda s: config.mgf_info_dict[s]['Charge'])

    #Encode de novo sequence amino acids as integers.
    deepnovo_df['Encoded Sequence'] = deepnovo_df['Sequence'].apply(lambda s: utils.encode_aas(s))
    #Record the length of the seqs.
    deepnovo_df['Sequence Length'] = deepnovo_df['Encoded Sequence'].apply(len)

    deepnovo_df.set_index(['Spectrum ID', 'Rank'], inplace=True)
    #Rearrange the columns.
    deepnovo_df = deepnovo_df.reindex(columns=config.alg_cols_dict['DeepNovo'])

    return deepnovo_df

def replace_mod_chars(seq, mod_dict):
    '''
    Replace one set of symbols for modified amino acids by another.

    Parameters
    ----------
    seq : str
        Amino acid sequence symbolized by standard amino acid letters and modification substrings.
    mod_dict : dict
        Mapping of one set of modification symbols to another, e.g., {'C+57.021': 'Cmod'}.

    Returns
    -------
    new_seq : str
    '''

    new_seq = seq
    for old_mod_code, new_mod_code in mod_dict.items():
        new_seq = new_seq.replace(old_mod_code, new_mod_code)

    return new_seq

def dereplicate_deepnovo_spec_dfs(spec_df):
    '''
    Dereplicate de novo sequence predictions for a spectrum, replacing 'I' with 'L'.

    Parameters
    ----------
    spec_df : pandas DataFrame
        A table of de novo sequence predictions and associated data.

    Returns
    -------
    derep_spec_df : pandas DataFrame
        A potentially shorter version of the input table, with substitution of 'I' in sequences.
    '''

    spec_seqs = spec_df['Sequence'].tolist()
    spec_seqs_all_leu = [seq.replace('I', 'L') for seq in spec_seqs]

    #There are occasionally fewer than 20 results per peptide.
    retained_rows = list(range(len(spec_seqs)))
    for i, seq1 in enumerate(spec_seqs_all_leu):
        if retained_rows[i] != -1:
            for j, seq2 in enumerate(spec_seqs_all_leu[i + 1:]):
                if retained_rows[i + j + 1] != -1:
                    if seq1 == seq2:
                        retained_rows[i + j + 1] = -1
    
    spec_df['Sequence'] = spec_seqs_all_leu
    derep_spec_df = spec_df.iloc[[i for i in retained_rows if i != -1]]
    derep_spec_df.index = range(len(derep_spec_df))

    return derep_spec_df

##IN PROGRESS:
#OLD CODE TO LOAD PEAKS DATA, ORIGINALLY FROM CLASSIFIER MODULE
##Load Peaks output for comparison.
#peaks_fp = os.path.join(
#    config.globals['iodir'], 
#    config.globals['filename'] + '.0.05.peaks.csv')
#peaks_df = pd.read_csv(peaks_fp, header=0)
#peaks_df = peaks_df[['Scan', 'Peptide', 'ALC (%)', 'm/z']]
#peaks_df.columns = ['Scan', 'Sequence', 'score', 'mz']
#peaks_df['Scan'] = peaks_df['Scan'].apply(lambda s: int(s.split(':')[1]))

#index_dict = OrderedDict()
#with open(config.globals['mgf_fp']) as handle:
#    for line in handle.readlines():
#        if line[:6] == 'TITLE=':
#            spec_id = line.split('; SpectrumID: "')[1].split('"; scans: "')[0]
#        elif line[:8] == 'PEPMASS=':
#            pepmass = line.split('PEPMASS=')[1].rstrip()
#        elif line[:6] == 'SCANS=':
#            scan = line.split('SCANS=')[1].rstrip()
#        elif line == 'END IONS\n':
#            if scan in index_dict:
#                index_dict[scan].append((pepmass, spec_id))
#            else:
#                index_dict[scan] = [(pepmass, spec_id)]

#spec_ids = []
#for scan, peaks_mz in zip(peaks_df['Scan'].tolist(), peaks_df['mz'].tolist()):
#    mgf_entries = index_dict[scan]
#    if len(mgf_entries) == 1:
#        spec_ids.append(int(mgf_entries[0][1]))
#    elif len(mgf_entries) > 1:
#        for t in mgf_entries:
#            mgf_mz = float(t[0])
#            if (mgf_mz - 0.01) <= float(peaks_mz) <= (mgf_mz + 0.01):
#                spec_ids.append(int(t[1]))
#                break
#        else:
#            raise AssertionError('Peaks and mgf data do not match up')
#            print('Peaks: scan = ' + str(scan) + ', m/z = ' + peaks_mz)
#            print('mgf entries: ')
#            print(mgf_entries)
#    else:
#        raise AssertionError(str(scan) + ' was not found in the mgf file')
#peaks_df['Spectrum ID'] = spec_ids
#peaks_df = peaks_df.groupby('Spectrum ID', as_index=False).first()

#peaks_df['Sequence'] = peaks_df['Sequence'].apply(lambda s: utils.remove_mod_chars(seq=s))
#peaks_df['Sequence Length'] = peaks_df['Sequence'].apply(len)

#psm_without_denovo_df = db_search_ref.merge(
#    peaks_df, how='left', on='Spectrum ID')
#psm_without_denovo_df = psm_without_denovo_df[
#    pd.isnull(psm_without_denovo_df['Sequence'])]
#psm_without_denovo_df = psm_without_denovo_df[
#    psm_without_denovo_df['Reference Sequence'].apply(len) >= 
#    config.globals['Minimum Postnovo Sequence Length']]
#psm_without_denovo_predict_count_dict['Peaks'] = len(psm_without_denovo_df)
#psm_without_denovo_predict_aa_len_dict['Peaks'] = psm_without_denovo_df[
#    'Reference Sequence'].apply(len).sum()

#peaks_df = peaks_df[peaks_df['Sequence Length'] >= config.globals['Minimum Postnovo Sequence Length']]
#peaks_df = peaks_df.merge(db_search_ref, how='left', on='Spectrum ID')
#peaks_df['Reference Sequence'][peaks_df['Reference Sequence'].isnull()] = ''
#peaks_df['Reference Sequence Match'] = find_seqs_in_paired_seqs(
#    peaks_df['Sequence'].tolist(), peaks_df['Reference Sequence'].tolist())
#peaks_df = peaks_df.set_index('Spectrum ID')[
#    ['Scan', 'score', 'Sequence', 'Reference Sequence', 'Reference Sequence Match', 'Sequence Length']]
#peaks_df = peaks_df.sort_values('score', ascending=False)
#alg_score_accuracy_df_dict['Peaks'] = peaks_df
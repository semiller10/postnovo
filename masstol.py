''' Compare sequences predicted with different fragment mass tolerances '''

#Copyright 2018, Samuel E. Miller. All rights reserved.
#Postnovo is publicly available for non-commercial uses.
#Licensed under GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007.
#See postnovo/LICENSE.txt.

import config
import utils

from utils import find_subarray

import gc
import multiprocessing
import numpy as np
import pandas as pd
import sys

from functools import partial

def do_mass_tol_procedure(prediction_df):
    '''
    Determine fragment mass tolerance agreement features.

    Parameters
    ----------
    prediction_df : pandas DataFrame
        Must contain spectrum ID, de novo algorithm origin, 
        fragment mass tolerance, and encoded sequence columns.

    Returns
    -------
    prediction_df : pandas DataFrame
        One column for each fragment mass tolerance comparison has been added.
    '''

    prediction_df.reset_index(inplace=True)
    if 'index' in prediction_df.columns:
        prediction_df.drop('index', axis=1, inplace=True)
    global frag_mass_tols
    frag_mass_tols = config.globals['Fragment Mass Tolerances']
    prediction_df.set_index(['Spectrum ID'] + frag_mass_tols, inplace=True)
    prediction_df.sort_index(ascending=[True] + [False] * len(frag_mass_tols), inplace=True)
    global alg_origin_headers
    alg_origin_headers = config.globals['De Novo Algorithm Origin Headers']
    all_specs_df = prediction_df[['Encoded Sequence'] + alg_origin_headers]

    #Example: tol_group_keys = [(1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), ...]
    global tol_group_keys
    tol_group_keys = []
    for i, tol in enumerate(frag_mass_tols):
        tol_group_key = [0] * len(frag_mass_tols)
        tol_group_key[i] = 1
        tol_group_keys.append(tuple(tol_group_key))

    all_specs_gb = all_specs_df.groupby(level='Spectrum ID', sort=False)
    spec_dfs = [spec_df for _, spec_df in all_specs_gb]
    del(all_specs_gb)
    gc.collect()
    one_percent_number_seqs_per_cpu = \
        len(spec_dfs) / 100 / config.globals['CPU Count']

    ##Single process
    #tol_match_arrays = []
    #print_percent_progress_fn = partial(
    #    utils.print_percent_progress_singlethreaded, 
    #    procedure_str='Fragment mass tolerance comparison progress: ', 
    #    one_percent_total_count=one_percent_number_seqs_per_cpu)
    #child_initialize(spec_dfs, print_percent_progress_fn)
    #for spec_index in range(len(spec_dfs)):
    #    tol_match_arrays.append(find_mass_tol_matches_for_spec(spec_index))

    #Multiprocessing
    print_percent_progress_fn = partial(
        utils.print_percent_progress_multithreaded, 
        procedure_str='Fragment mass tolerance comparison progress: ', 
        one_percent_total_count=one_percent_number_seqs_per_cpu, 
        cores=config.globals['CPU Count'])
    mp_pool = multiprocessing.Pool(
        config.globals['CPU Count'], 
        initializer=child_initialize, 
        initargs=(spec_dfs, print_percent_progress_fn))
    tol_match_arrays = mp_pool.map(
        find_mass_tol_matches_for_spec, range(len(spec_dfs)))
    mp_pool.close()
    mp_pool.join()

    #Delete the global variable that was assigned in child_initialize.
    global spec_dfs
    del(spec_dfs)
    gc.collect()

    tol_match_cols = [frag_mass_tol + ' Match Value' for frag_mass_tol in frag_mass_tols]
    prediction_df[[frag_mass_tol + ' Match Value' for frag_mass_tol in frag_mass_tols]] = \
        pd.DataFrame(np.concatenate(tol_match_arrays), index=prediction_df.index)

    return prediction_df

def child_initialize(_spec_dfs, _print_percent_progress_fn):
    '''
    Set up the function, find_mass_tol_matches_for_spec.

    Parameters
    ----------
    _spec_dfs : list
        One DataFrame per spectrum.
    _print_percent_progress_fn : function

    Returns
    -------
    None
    '''

    global spec_dfs, print_percent_progress_fn

    spec_dfs = _spec_dfs
    print_percent_progress_fn = _print_percent_progress_fn
    
    del(_spec_dfs)
    gc.collect()

    return

def find_mass_tol_matches_for_spec(spec_index):
    '''
    Record information on matches between sequence predictions 
    at different fragment mass tolerances.

    Parameters
    ----------
    spec_index : int
        Recovers DataFrame of information on the spectrum.

    Returns
    -------
    tol_match_array : numpy array
    '''

    #This function has a global scope to permit tracking progress across multiple processes.
    print_percent_progress_fn()

    spec_df = spec_dfs[spec_index]
    spec_df.reset_index('Spectrum ID', drop=True, inplace=True)
    mass_tol_gb = spec_df.groupby(level=frag_mass_tols)
    #Return an array recording matches between predictions across fragment mass tolerances.
    #Example with low-resolution spectra and two de novo algorithms: 
    #There are no predictions at 0.2 Da, 
    #predictions that did not form consensus sequences at 0.3-0.6 Da, 
    #and predictions that formed consensus sequences at 0.7 Da.
    #Sequences are only compared to other sequences originating from the same algorithm.
    #The array is initialized to a value of -2, 
    #which is retained for self-comparisons (e.g., 0.3 Novor seq x 0.3 Novor seq).
    #A value of -1 is assigned if there is no sequence to compare to 
    #(e.g., 0.3 Novor seq x 0.2 Novor seq, which does not exist here).
    #A value of 0 is assigned if the first sequence is not a subsequence of the second 
    #(e.g., 0.3 PepNovo+ seq x 0.5 PepNovo+ seq).
    #A value of 1 is assigned if the first sequence is a subsequence of the second, 
    #and the second is not a consensus sequence 
    #(e.g., 0.3 Novor seq x 0.4 Novor seq).
    #A value of 2 is assigned if the first sequence is a subsequence of the second, 
    #and the second is a consensus sequence 
    #(e.g., 0.3 Novor seq x 0.7 Novor-PepNovo+ consensus seq).
    #Match                                    0.2   0.3   0.4   0.5   0.6   0.7
    #0.3 Novor: 'EAAAADK'                     -1    -2     1     1     1     2
    #0.3 PepNovo+: 'AEAAADK'                  -1    -2    -1     0     1     0
    #0.4 Novor: 'EAAAADK'                     -1     1    -2     1     1     2
    #0.5 Novor: 'EAAAADK'                     -1     1     1    -2     1     2
    #0.5 PepNovo+: 'AAAADK'                   -1     0    -1    -2     0     2
    #0.6 Novor: 'EAAAADK'                     -1     1     1     1    -2     2
    #0.6 PepNovo+: 'AEAAADK'                  -1     1    -1     0    -2     0
    #0.7 Novor: 'EAAAADK'                     -1     1     1     1     1    -2
    #0.7 PepNovo+: 'EAAAADK'                  -1     0    -1     0     0    -2
    #0.7 Novor-PepNovo+ consensus: 'EAAAADK'  -1     1     1     1     1    -2
    tol_match_array = np.full([len(spec_df), len(tol_group_keys)], -2)
    for first_tol_group_key_index, first_tol_group_key in enumerate(tol_group_keys):
        #The spectrum may not contain predictions for each mass tolerance.
        if first_tol_group_key not in spec_df.index:
            tol_match_array[:, first_tol_group_key_index] = -1
            continue

        #Select the sequences from the first fragment mass tolerance in the comparison.
        first_tol_group_df = mass_tol_gb.get_group(first_tol_group_key)
        try:
            first_tol_group_start_index = spec_df.index.get_loc(first_tol_group_key).start
        #Certain mass tolerances may have only one row (sequence prediction), 
        #triggering the exception.
        except AttributeError:
            first_tol_group_start_index = spec_df.index.get_loc(first_tol_group_key)

        for first_seq_index, first_seq_row in enumerate(first_tol_group_df.values):
            first_encoded_seq = first_seq_row[0]
            first_alg_origins = first_seq_row[1:]

            for second_tol_group_key_index, second_tol_group_key in enumerate(tol_group_keys):
                #Do not compare sequences from the same fragment mass tolerance.
                if first_tol_group_key_index == second_tol_group_key_index:
                    continue

                #If there are no sequences at the second fragment mass tolerance, 
                #record a value of -1 in the appropriate cell.
                if second_tol_group_key not in spec_df.index:
                    tol_match_array[
                        first_tol_group_start_index + first_seq_index, 
                        second_tol_group_key_index] = -1
                    continue

                #Select the sequences from the second fragment mass tolerance in the comparison.
                second_tol_group_df = mass_tol_gb.get_group(second_tol_group_key)

                #If none of the sequences from the second fragment mass tolerance 
                #share a source algorithm with the first sequence under consideration, 
                #record a value of -1 in the appropriate cell, 
                #and proceed to the next fragment mass tolerance for comparison.
                for i, first_alg_origin in enumerate(first_alg_origins):
                    if first_alg_origin == 1 and \
                        second_tol_group_df[alg_origin_headers[i]].sum() > 0:
                        break
                else:
                    tol_match_array[
                        first_tol_group_start_index + first_seq_index, 
                        second_tol_group_key_index] = -1
                    continue
                
                try:
                    second_tol_group_start_index = spec_df.index.get_loc(
                        second_tol_group_key).start
                #Certain mass tolerances may have only one row (sequence prediction), 
                #triggering the exception.
                except AttributeError:
                    second_tol_group_start_index = spec_df.index.get_loc(second_tol_group_key)

                for second_seq_index, second_seq_row in enumerate(second_tol_group_df.values):
                    second_encoded_seq = second_seq_row[0]
                    second_alg_origins = second_seq_row[1:]

                    #Do not compare the sequences 
                    #if they do not have at least one source algorithm in common.
                    for i, first_alg_origin in enumerate(first_alg_origins):
                        if first_alg_origin == 1 and second_alg_origins[i] == 1:
                            break
                    else:
                        continue

                    if find_subarray(first_encoded_seq, second_encoded_seq) == -1:
                        #The first sequence is not a subsequence of the second sequence, 
                        #so record a value of 0 in the appropriate cell, 
                        #unless there is already a value in the cell that exceeds 0 
                        #(the first sequence matches 
                        #a different second sequence from the fragment mass tolerance).
                        if tol_match_array[
                            first_tol_group_start_index + first_seq_index, 
                            second_tol_group_key_index] < 0:
                            tol_match_array[
                                first_tol_group_start_index + first_seq_index, 
                                second_tol_group_key_index] = 0
                    else:
                        #The first sequence is a subsequence of the second sequence.
                        #Record a value of 2 if the second sequence is consensus sequence, 
                        #otherwise record a value of 1.
                        #Only record this value if it is higher than a value already in the cell 
                        #(2 is the maximum value).
                        if sum(second_alg_origins) > 1:
                            tol_match_array[
                                first_tol_group_start_index + first_seq_index, 
                                second_tol_group_key_index] = 2
                        else:
                            if tol_match_array[
                                first_tol_group_start_index + first_seq_index, 
                                second_tol_group_key_index] < 1:
                                tol_match_array[
                                    first_tol_group_start_index + first_seq_index, 
                                    second_tol_group_key_index] = 1

    return tol_match_array
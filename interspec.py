''' Compare sequences predicted from different spectra but the same peptide '''

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

def do_interspec_procedure(prediction_df):
    '''
    Determine interspectrum agreement features 
    from clusters of spectra likely derived from the same peptide species.

    Parameters
    ----------
    prediction_df : pandas DataFrame
        Must contain spectrum ID, de novo algorithm origin, 
        fragment mass tolerance, and encoded sequence columns.

    Returns
    -------
    prediction_df : pandas DataFrame
        Six columns are added, with five being Postnovo model features.
    '''

    #Load the MaRaCluster output table.
    all_clusters_input_df = pd.read_csv(
        config.globals['Clusters Filepath'], sep='\t', header=None, usecols=[1, 2])
    all_clusters_input_df.columns = ['Spectrum ID', 'Spectrum Cluster ID']
    
    #Feature: The number of spectra in the cluster (with or without de novo sequence predictions).
    all_clusters_input_df['Clustered Spectra'] = \
        all_clusters_input_df.groupby('Spectrum Cluster ID').transform(len)

    #Merge the cluster information with sequence prediction information.
    prediction_df.reset_index(inplace=True)
    if 'index' in prediction_df.columns:
        prediction_df.drop('index', axis=1, inplace=True)
    global frag_mass_tols
    frag_mass_tols = config.globals['Fragment Mass Tolerances']
    global alg_origin_headers
    alg_origin_headers = config.globals['De Novo Algorithm Origin Headers']
    all_clusters_df = prediction_df[
        ['Spectrum ID'] + frag_mass_tols + ['Encoded Sequence'] + alg_origin_headers].merge(
            all_clusters_input_df, how='left', on='Spectrum ID')
    del(all_clusters_input_df)
    gc.collect()
    all_clusters_df.set_index(['Spectrum Cluster ID'] + frag_mass_tols, inplace=True)
    all_clusters_df.sort_index(ascending=[True] + [False] * len(frag_mass_tols), inplace=True)
    #Setting sort to "False" keeps the proper order of fragment mass tolerance.
    all_clusters_gb = all_clusters_df.groupby(all_clusters_df.index, sort=False)
    cluster_dfs = [cluster_df for _, cluster_df in all_clusters_gb]
    del(all_clusters_gb)
    gc.collect()
    one_percent_number_clusters_per_cpu = \
        len(cluster_dfs) / 100 / config.globals['CPU Count']

    ##Single process
    #spec_comparison_arrays = []
    #print_percent_progress_fn = partial(
    #    utils.print_percent_progress_singlethreaded, 
    #    procedure_str='Spectrum cluster comparison progress: ', 
    #    one_percent_total_count=one_percent_number_clusters_per_cpu)
    #child_initialize(cluster_dfs, print_percent_progress_fn)
    #for cluster_index in range(len(cluster_dfs)):
    #    spec_comparison_arrays.append(compare_clustered_spectra(cluster_index))

    #Multiprocessing
    print_percent_progress_fn = partial(
        utils.print_percent_progress_multithreaded, 
        procedure_str='Spectrum cluster comparison progress: ', 
        one_percent_total_count=one_percent_number_clusters_per_cpu, 
        cores=config.globals['CPU Count'])
    mp_pool = multiprocessing.Pool(
        config.globals['CPU Count'], 
        initializer=child_initialize, 
        initargs=(cluster_dfs, print_percent_progress_fn))
    #Since larger clusters are loaded toward the front of iterable, 
    #the default multiprocessing setting of one chunk per process 
    #can result in most work being done by one process. 
    #So instead pass a larger number of small chunks to the processes to spread the workload.
    spec_comparison_arrays = mp_pool.map(
        compare_clustered_spectra, range(len(cluster_dfs)), chunksize=5)
    mp_pool.close()
    mp_pool.join()

    #Delete the global variable that was assigned in child_initialize.
    global cluster_dfs
    del(cluster_dfs)
    gc.collect()

    spec_comparison_cols = [
        'Clustered Spectra with a Sequence Containing This Sequence', 
        'Clustered Spectra with a Consensus Sequence Containing This Sequence', 
        'Clustered Spectra with a Sequence Contained in This Sequence', 
        'Clustered Spectra with a Consensus Sequence Contained in This Sequence']
    all_clusters_df[spec_comparison_cols] = pd.DataFrame(
        np.concatenate(spec_comparison_arrays), index=all_clusters_df.index)
    all_clusters_df.reset_index(inplace=True)
    all_clusters_df.set_index(['Spectrum ID'] + frag_mass_tols, inplace=True)
    all_clusters_df.sort_index(ascending=[True] + [False] * len(frag_mass_tols), inplace=True)
    prediction_df.set_index(['Spectrum ID'] + frag_mass_tols, inplace=True)
    prediction_df.sort_index(ascending=[True] + [False] * len(frag_mass_tols), inplace=True)
    prediction_df[['Spectrum Cluster ID', 'Clustered Spectra'] + spec_comparison_cols] = \
        all_clusters_df[['Spectrum Cluster ID', 'Clustered Spectra'] + spec_comparison_cols]

    return prediction_df

def child_initialize(_cluster_dfs, _print_percent_progress_fn):
    '''
    Set up the function, compare_clustered_spectra.

    Parameters
    ----------
    _cluster_dfs : list
        List of pandas DataFrames
    _print_percent_progress_fn : fn

    Returns
    -------
    None
    '''

    global cluster_dfs, print_percent_progress_fn

    cluster_dfs = _cluster_dfs
    print_percent_progress_fn = _print_percent_progress_fn

    del(_cluster_dfs)
    gc.collect()

    return

def compare_clustered_spectra(cluster_index):
    '''
    Record information on matches between sequence predictions 
    from different spectra in the peptide cluster.

    Parameters
    ----------
    cluster_index : int

    Returns
    -------
    spec_comparison_array : numpy array
    '''

    print_percent_progress_fn()

    cluster_df = cluster_dfs[cluster_index]

    spec_comparison_array = np.zeros([len(cluster_df), 4])

    #Sequences are only compared to other sequences 
    #originating from the same algorithm at the same fragment mass tolerance, 
    #e.g., Novor sequences are compared to Novor sequences from other spectra  
    #and consensus sequences derived from Novor sequences.
    #Features recorded in spec_comparison_array columns: 
    #1. The number of other spectra in the cluster with at least one sequence
    #containing as a subsequence the sequence under consideration.
    #2. The number of other spectra in the cluster with at least one consensus sequence 
    #containing as a subsequence the sequence under consideration.
    #3. The number of other spectra in the cluster with at least one sequence 
    #being a subsequence of the sequence under consideration.
    #4. The number of other spectra in the cluster with at least one consensus sequence 
    #being a subsequence of the sequence under consideration.

    #Determine the first and second columns of the array.
    for first_seq_index, first_seq_row in enumerate(cluster_df.values):
        first_spec_id = first_seq_row[0]
        first_encoded_seq = first_seq_row[1]
        #The last column is the feature that was determined in do_interspec_procedure.
        first_alg_origins = first_seq_row[2: -1]
        #As only one match to single-algorithm and consensus sequences is recorded per spectrum, 
        #track whether such matches have been found to the second spectrum.
        previously_matched_single_alg_spec_id = None
        previously_matched_consensus_spec_id = None

        for second_seq_index, second_seq_row in enumerate(cluster_df.values):
            second_spec_id = second_seq_row[0]
            if first_spec_id == second_spec_id or \
                second_spec_id == previously_matched_consensus_spec_id:
                continue

            second_encoded_seq = second_seq_row[1]
            second_alg_origins = second_seq_row[2: -1]

            #The sequences from the first and second spectra must share a source algorithm 
            #if a sequence comparison is to be performed.
            for i, first_alg_origin in enumerate(first_alg_origins):
                if first_alg_origin == 1 and second_alg_origins[i] == 1:
                    break
            else:
                continue

            if find_subarray(first_encoded_seq, second_encoded_seq) != -1:
                if second_spec_id != previously_matched_single_alg_spec_id:
                    spec_comparison_array[first_seq_index, 0] += 1
                    previously_matched_single_alg_spec_id = second_spec_id
                if sum(second_alg_origins) > 1:
                    spec_comparison_array[first_seq_index, 1] += 1
                    previously_matched_consensus_spec_id = second_spec_id

    #Determine the third and fourth columns of the array.
    for first_seq_index, first_seq_row in enumerate(cluster_df.values):
        first_spec_id = first_seq_row[0]
        first_encoded_seq = first_seq_row[1]
        #The last column is the feature that was determined in do_interspec_procedure.
        first_alg_origins = first_seq_row[2: -1]

        #As only one match to single-algorithm and consensus sequences is recorded per spectrum, 
        #track whether such matches have been found to the second spectrum.
        previously_matched_single_alg_spec_id = None
        previously_matched_consensus_spec_id = None

        for second_seq_index, second_seq_row in enumerate(cluster_df.values):
            second_spec_id = second_seq_row[0]
            if first_spec_id == second_spec_id or \
                second_spec_id == previously_matched_consensus_spec_id:
                continue

            second_encoded_seq = second_seq_row[1]
            second_alg_origins = second_seq_row[2: -1]

            #The sequences from the first and second spectra must share a source algorithm 
            #if a sequence comparison is to be performed.
            for i, first_alg_origin in enumerate(first_alg_origins):
                if first_alg_origin == 1 and second_alg_origins[i] == 1:
                    break
            else:
                continue

            if find_subarray(second_encoded_seq, first_encoded_seq) != -1:
                if second_spec_id != previously_matched_single_alg_spec_id:
                    spec_comparison_array[first_seq_index, 2] += 1
                    previously_matched_single_alg_spec_id = second_spec_id
                if sum(second_alg_origins) > 1:
                    spec_comparison_array[first_seq_index, 3] += 1
                    previously_matched_consensus_spec_id = second_spec_id

    return spec_comparison_array
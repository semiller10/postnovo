''' Find consensus sequences from de novo sequences. '''

import config
import utils

from utils import count_low_scoring_peptides, get_potential_substitution_info

import gc
import multiprocessing
import numpy as np
import os
import pandas as pd
import sys

from collections import OrderedDict
from copy import deepcopy
from functools import partial
from itertools import groupby, product, combinations_with_replacement
from re import finditer

possible_algs = config.possible_algs

class Seq():
    '''
    A Seq object represents a sequence originating from de novo sequencing algorithms.

    Parameters
    ----------
    aas : numpy array
    algs : list
    rank_index : tuple
    source_aa_starts : tuple

    Attributes
    ----------
    aas : numpy array
    length : int
    algs : list
    alg_indices : dict
    rank_index : tuple
    source_aa_starts : tuple
    '''

    def __init__(self, aas, algs, rank_index=(0, ), source_aa_starts=(0, )):
        #The sequence's amino acids are encoded as integers in an array.
        self.aas = aas
        self.length = len(aas)
        self.algs = algs
        self.alg_indices = {(alg, i) for i, alg in enumerate(algs)}
        #The rank index links the sequence 
        #to the original de novo sequences from which it is derived.
        #Example: A rank index of (0, 3) and algs of ['Novor', PepNovo'] 
        #mean that the sequence comes from Novor candidate sequence #1 and PepNovo candidate #4.
        self.rank_index = rank_index
        #The following attribute records the starting position of the sequence 
        #in the original de novo sequences from which it is derived.
        #Example: A value of (0, 1) and algs of ['Novor', 'PepNovo'] 
        #mean that this is a subsequence starting at the first amino acid of the Novor candidate 
        #and the second amino acid of the PepNovo candidate.
        self.source_aa_starts = source_aa_starts

        return

    def find_lcs(seq1, seq2, min_len):
        '''
        Find the longest common subsequence (LCS) 
        of at least the mininum length from two parent sequences.

        Parameters
        ----------
        seq1 : Seq object
        seq2 : Seq object
        min_len : int

        Returns
        -------
        lcs : CommonSeq object
        or None
            if no LCS meeting the criteria is found
        '''

        #Make a matrix comparing each amino acid in the two sequences.
        #Make the seq1 and seq2 amino acid vectors orthogonal.
        seq1_aas = seq1.aas.reshape(seq1.length, 1)
        #Fill in the 2D matrix formed by the dimensions of the vectors with seq2's amino acids.
        tiled_seq2_aas = np.tile(seq2.aas, (seq1.length, 1))
        #Project the seq1 amino acids over the 2D matrix to find identical amino acids.
        match_arr = np.equal(seq1_aas, tiled_seq2_aas).astype(int)

        #Find any common substrings, which are diagonals of True values in match_arr.
        #Diagonal index 0 is the main diagonal.
        #Negatively indexed diagonals lie below the main diagonal.
        #Only consider diagonals long enough 
        #to contain common substrings meeting the minimum length.
        diags = [match_arr.diagonal(d) for d in range(
            -seq1.length + min_len, seq2.length - min_len + 1)]

        #Identify common substrings in the diagonals.
        lcs_len = min_len
        found_long_cs = False
        #Loop through the diagonals from bottom left to upper right.
        for diag_index, diag in enumerate(diags):
            #Create and loop through two groups of Trues (common substrings) and Falses
            #from the elements of the diagonal.
            for match_status, diag_group in groupby(diag):
                #If considering a common substring, retain it as the longest common substring 
                #if it is at least as long as any LCS found from the comparison.
                if match_status:
                    cs_len = sum(diag_group)
                    if cs_len >= lcs_len:
                        found_long_cs = True
                        lcs_len = cs_len
                        #Record the diagonal's index, 
                        #with the leftmost bottom corner of the matrix indexed as zero.
                        lcs_diag_index = diag_index
                        lcs_diag = diag

        if found_long_cs:
            #Find where the LCS is located in the diagonal.
            #Take the first LCS if, improbably, multiple LCS's of equal length are in the diagonal.
            for diag_aa_position in range(lcs_diag.size - lcs_len + 1):
                for lcs_aa_position in range(lcs_len):
                    if not lcs_diag[diag_aa_position + lcs_aa_position]:
                        break
                else:
                    diag_lcs_start_position = diag_aa_position
                    break

            #Determine the position of the first amino acid of the LCS in seq1 and seq2.
            #Reindex the LCS-containing diagonal to the main diagonal.
            #Negatively indexed diagonals lie below the main diagonal.
            upper_left_diag_index = seq1.length - min_len
            relative_lcs_diag_index = lcs_diag_index - upper_left_diag_index
            if relative_lcs_diag_index < 0:
                seq1_lcs_start_position = diag_lcs_start_position - relative_lcs_diag_index
                seq2_lcs_start_position = diag_lcs_start_position
            else:
                seq1_lcs_start_position = diag_lcs_start_position
                seq2_lcs_start_position = relative_lcs_diag_index + diag_lcs_start_position

            return CommonSeq(
                seq1.aas[seq1_lcs_start_position: seq1_lcs_start_position + lcs_len], 
                seq1.algs + seq2.algs, 
                seq1.rank_index + seq2.rank_index, 
                [source_aa_start + seq1_lcs_start_position 
                 for source_aa_start in seq1.source_aa_starts] + \
                     [source_aa_start + seq2_lcs_start_position 
                      for source_aa_start in seq2.source_aa_starts], 
                [seq1, seq2])

        else:
            return None

        return
    
class CommonSeq(Seq):
    '''
    A Seq that is a consensus subsequence of parent sequences.

    Parameters
    ----------
    aas : numpy array
    algs : list
    rank_index : tuple
    source_aa_starts : tuple
    parent_seqs : list of arrays

    Attributes
    ----------
    aas : numpy array
    length : int
    algs : list
    alg_indices : dict
    rank_index : tuple
    source_aa_starts : tuple
    parent_seqs : list of arrays
    rank_sum : int
    alg_info_dict : OrderedDict object
    '''

    def __init__(
        self, 
        aas, 
        algs, 
        rank_index, 
        source_aa_starts, 
        parent_seqs):

        super(CommonSeq, self).__init__(aas, algs, rank_index, source_aa_starts)
        #Link to the immediate "parent" Seqs of this CommonSeq.
        self.parent_seqs = parent_seqs
        self.rank_sum = sum(rank_index)
        self.alg_info_dict = OrderedDict()

        return

def do_consensus_procedure():
    '''
    Find longest and top-ranked consensus sequences 
    from de novo algorithm predictions at each fragment mass tolerance parameterization.

    Parameters
    ----------
    None

    Returns
    -------
    prediction_df : DataFrame object
    '''

    global min_len, frag_mass_tols
    min_len = config.globals['Minimum Postnovo Sequence Length']
    frag_mass_tols = config.globals['Fragment Mass Tolerances']

    #Find the combinations of de novo algorithms considered in the consensus procedure.
    #Example: 
    ##combo_level_alg_combos_dict = OrderedDict(
    ##    2: [('Novor', 'PepNovo'), ('Novor', 'DeepNovo'), ('PepNovo', 'DeepNovo')], 
    ##    3: [('Novor', 'PepNovo', 'DeepNovo')])
    combo_level_alg_combos_dict = OrderedDict()
    for combo_level in range(2, len(config.globals['De Novo Algorithms']) + 1):
        combo_level_alg_combos_dict[combo_level] = []
        combo_level_alg_combos_dict[combo_level] += [
            alg_combo for alg_combo in config.globals['De Novo Algorithm Comparisons']
            if len(alg_combo) == combo_level]
    #Example:
    ##highest_level_alg_combo = ('Novor', 'PepNovo', 'DeepNovo')
    highest_level_alg_combo = config.globals['De Novo Algorithm Comparisons'][-1]

    for frag_mass_tol in frag_mass_tols:
        utils.verbose_print('Finding', frag_mass_tol, 'Da consensus sequences')

        #Load the DataFrames for the fragment mass tolerance.
        alg_source_df_dict = OrderedDict()
        for alg in config.globals['De Novo Algorithms']:
            alg_source_df_dict[alg] = utils.load_pkl_objects(
                config.globals['Output Directory'], 
                alg + '.' + config.globals['MGF Filename'] + '.' + frag_mass_tol + '.pkl')

        #Make a list of the spectrum IDs with de novo sequences for the fragment mass tolerance.
        all_spec_ids = []
        for alg, source_df in alg_source_df_dict.items():
            all_spec_ids += source_df.index.get_level_values('Spectrum ID').tolist()
        all_spec_ids = sorted(list(set(all_spec_ids)))
        one_percent_number_seqs_per_cpu = len(all_spec_ids) / 100 / config.globals['CPU Count']

        ##Single process
        #print_percent_progress_fn = partial(
        #    utils.print_percent_progress_singlethreaded, 
        #    procedure_str=frag_mass_tol + ' Da progress: ', 
        #    one_percent_total_count=one_percent_number_seqs_per_cpu)
        ##Define global variables, done for the purpose of multiprocessing.
        #child_initialize(
        #    frag_mass_tol, 
        #    combo_level_alg_combos_dict, 
        #    alg_source_df_dict, 
        #    print_percent_progress_fn)
        #result_dfs = []
        #for spec_id in all_spec_ids:
        #    result_dfs.append(analyze_spectrum(spec_id))

        #Multiprocessing        
        print_percent_progress_fn = partial(
            utils.print_percent_progress_multithreaded, 
            procedure_str=frag_mass_tol + ' Da progress: ', 
            one_percent_total_count=one_percent_number_seqs_per_cpu, 
            cores=config.globals['CPU Count'])
        mp_pool = multiprocessing.Pool(
            config.globals['CPU Count'],
            initializer=child_initialize,
            initargs=(
                frag_mass_tol, 
                combo_level_alg_combos_dict, 
                alg_source_df_dict, 
                print_percent_progress_fn))
        result_dfs = mp_pool.map(analyze_spectrum, all_spec_ids)
        mp_pool.close()
        mp_pool.join()
        del(alg_source_df_dict)

        #Concatenate DataFrames from each spectrum, 
        #with each row representing a top-ranked or consensus sequence.
        frag_mass_tol_df = pd.concat(result_dfs, ignore_index=True)
        del(result_dfs)
        for possible_frag_mass_tol in config.globals['Fragment Mass Tolerances']:
            if frag_mass_tol == possible_frag_mass_tol:
                frag_mass_tol_df[possible_frag_mass_tol] = 1
            else:
                frag_mass_tol_df[possible_frag_mass_tol] = 0
        #Instead of keeping the DataFrames for each fragment mass tolerance in memory, 
        #which can exceed available memory for large datasets, temporarily save them to file.
        utils.save_pkl_objects(
            config.globals['Output Directory'], 
            **{'consensus_prediction_df.' + frag_mass_tol + '.pkl': frag_mass_tol_df})
        del(frag_mass_tol_df)
        gc.collect()

    consensus_prediction_df = pd.DataFrame()
    for frag_mass_tol in config.globals['Fragment Mass Tolerances']:
        frag_mass_tol_df = utils.load_pkl_objects(
            config.globals['Output Directory'], 
            'consensus_prediction_df.' + frag_mass_tol + '.pkl')
        consensus_prediction_df = pd.concat(
            [consensus_prediction_df, frag_mass_tol_df], ignore_index=True)
        os.remove(os.path.join(
            config.globals['Output Directory'], 
            'consensus_prediction_df.' + frag_mass_tol + '.pkl'))

    return consensus_prediction_df

def child_initialize(
    _frag_mass_tol, 
    _combo_level_alg_combos_dict, 
    _alg_source_df_dict, 
    _print_percent_progress_fn):
    '''
    Initialize global variables for the function, analyze_spectrum.

    Parameters
    ----------
    _frag_mass_tol : str
    _combo_level_alg_combos_dict : dict
    _alg_source_df_dict : dict
    _print_percent_progress_fn : function

    Returns
    -------
    None
    '''

    global frag_mass_tol, combo_level_alg_combos_dict, alg_source_df_dict, print_percent_progress_fn

    frag_mass_tol = _frag_mass_tol
    combo_level_alg_combos_dict = _combo_level_alg_combos_dict
    alg_source_df_dict = _alg_source_df_dict
    print_percent_progress_fn = _print_percent_progress_fn

    return

def analyze_spectrum(spec_id):
    '''
    Find longest and top-ranked consensus sequences for a spectrum.

    Parameters
    ----------
    spec_id : int

    Returns
    -------
    result_df : DataFrame object
    '''
    print_percent_progress_fn()

    #Determine whether consensus sequences are capable of being found for the spectrum.
    #Record de novo sequence candidate information from each algorithm for the spectrum.
    alg_source_df_for_spec_dict = OrderedDict()
    #Record the number of de novo sequence candidates predicted by each algorithm.
    #DeepNovo can have a variable number of candidates per spectrum
    #due to the consolidation of candidates differing only by Ile/Leu.
    #PepNovo+ can have fewer than the expected 20 candidates for low-quality spectra.
    alg_seq_count_dict = OrderedDict()
    #Record the maximum length of qualifying de novo sequence candidates for each algorithm.
    alg_max_seq_len_dict = OrderedDict()

    algs_with_long_seqs_count = 0
    for alg, source_df in alg_source_df_dict.items():
        #The algorithm may not have predicted sequence candidates for the spectrum.
        if spec_id in source_df.index:
            source_df_for_spec = source_df.loc[spec_id]
            alg_source_df_for_spec_dict[alg] = source_df_for_spec
            reported_seq_count = len(source_df_for_spec)
            alg_max_seq_len_dict[alg] = source_df_for_spec['Sequence Length'].max()

            #Determine whether any sequence candidates meet Postnovo's minimum length.
            if alg_max_seq_len_dict[alg] >= min_len:
                algs_with_long_seqs_count += 1
                alg_seq_count_dict[alg] = reported_seq_count
            else:
                alg_seq_count_dict[alg] = 0
                alg_max_seq_len_dict[alg] = 0
        else:
            alg_seq_count_dict[alg] = 0
            alg_max_seq_len_dict[alg] = 0
    #Predictions from multiple de novo algorithms are necessary for consensus sequences.
    if algs_with_long_seqs_count <= 1:
        return None

    #Find consensus sequences between increasing numbers of algorithms (2, 3, etc.).
    #Map out the candidate sequence comparisons.
    #Certain algorithms may not have candidates meeting the length threshold at each rank, 
    #but these are processed for contiguity of the rank index:
    #for example, if PepNovo+ candidates #2 and 20 are shorter than the minimum length, 
    #but candidates #1 and #3-19 are longer, 
    #then #2 but not #20 will still be processed by the consensus procedure.
    #Here is an example of the data structure recording the candidate comparisons, 
    #with 3 algorithms and the maximum number of candidates per algorithm considered:
    ##rank_comparison_dict = OrderedDict([
    ##    (2, OrderedDict([
    ##        (('Novor', 'PepNovo'), [(0, 0), (0, 1), ..., (0, 18), (0, 19)]), 
    ##        (('Novor', 'DeepNovo'), [(0, 0), (0, 1), ..., (0, 18), (0, 19)]), 
    ##        (('PepNovo', 'DeepNovo'), [(0, 0), (0, 1), ..., (19, 18), (19, 19)])])), 
    ##    (3, OrderedDict([
    ##      (('Novor', 'PepNovo', 'DeepNovo'), 
    ##       [(0, 0, 0), (0, 0, 1), ..., (0, 19, 18), (0, 19, 19)])]))])
    rank_comparison_dict = OrderedDict()
    for combo_level in combo_level_alg_combos_dict:
        rank_comparison_dict[combo_level] = combo_level_dict = OrderedDict()
        for alg_combo in combo_level_alg_combos_dict[combo_level]:
            alg_rank_ranges = []
            for alg in alg_combo:
                alg_rank_ranges.append(range(alg_seq_count_dict[alg]))
            #Example with 0 predictions from one of the algorithms: 
            ##1 Novor, 0 PepNovo+, 20 DeepNovo candidates results in 
            ##list(product(*[range(1), range(0), range(20)])) == []
            combo_level_dict[alg_combo] = list(product(*alg_rank_ranges))

    #Store generator functions 
    #used in longest common subsequence (LCS) comparisons of lists of sequences.
    #The states of these generators must be maintained through the spectrum consensus procedure.
    #Example with 3 algorithms total:
    ##spec_generator_fns_dict = OrderedDict([
    ##    (2, OrderedDict([
    ##        (('Novor', 'PepNovo'), <generator function>), 
    ##        (('Novor', 'DeepNovo'): <generator function>), 
    ##        (('PepNovo', 'DeepNovo'): <generator function>)])), 
    ##    (3: OrderedDict([
    ##        (('Novor', 'PepNovo', 'DeepNovo'), <generator function>)]))])
    spec_generator_fns_dict = OrderedDict()
    for combo_level, alg_combos in combo_level_alg_combos_dict.items():
        spec_generator_fns_dict[combo_level] = combo_level_dict = OrderedDict()
        for alg_combo in alg_combos:
            combo_level_dict[alg_combo] = OrderedDict()

    #Store information on whether sequence comparisons have been performed (0 = no, 1 = yes).
    #Example with 3 algorithms total: 
    ##did_comparison_dict = OrderedDict([
    ##    (2, OrderedDict([
    ##        (('Novor', 'PepNovo'), OrderedDict([
    ##            ((0, 0), <0 or 1>), 
    ##            ((0, 1), <0 or 1>), 
    ##            ...
    ##            ((0, 19), <0 or 1>)])), 
    ##        (('Novor', 'DeepNovo'), OrderedDict([...])), 
    ##        (('PepNovo', 'DeepNovo'), OrderedDict([...]))])), 
    ##    (3, OrderedDict([
    ##        (('Novor', 'PepNovo', 'DeepNovo'), OrderedDict([
    ##            ((0, 0, 0), <0 or 1>), 
    ##            ...
    ##            ((0, 19, 19), <0 or 1>)]))]))])
    did_comparison_dict = OrderedDict()
    for combo_level, alg_combos in combo_level_alg_combos_dict.items():
        did_comparison_dict[combo_level] = combo_level_did_comparison_dict = OrderedDict()
        combo_level_rank_comparison_dict = rank_comparison_dict[combo_level]
        for alg_combo, rank_comparisons in combo_level_rank_comparison_dict.items():
            combo_level_did_comparison_dict[alg_combo] = alg_combo_did_comparison_dict = \
                OrderedDict()
            for rank_comparison in rank_comparisons:
                #None of the comparisons have been performed yet, so initialize to 0.
                alg_combo_did_comparison_dict[rank_comparison] = 0

    #Store the LCS CommonSeq object from each comparison.
    #A value of None is stored if the comparison has not been performed, 
    #or if no LCS was found in the comparison.
    #A record of all identified LCS's is necessary for the recursive procedure.
    #Example with 3 algorithms total, after every possible candidate has been compared:
    ##spec_lcs_info_dict = OrderedDict([
    ##    (2, OrderedDict([
    ##        (('Novor', 'PepNovo'), OrderedDict([
    ##            ((0, 0), <CommonSeq object>), 
    ##            ((0, 1), None), 
    ##            ...
    ##            ((0, 19), <CommonSeq object)])), 
    ##        (('Novor', 'DeepNovo'), OrderedDict([...])),
    ##        (('PepNovo', 'DeepNovo'), OrderedDict([...]))])), 
    ##    (3, OrderedDict([
    ##        (('Novor', 'PepNovo', 'DeepNovo'), OrderedDict([
    ##            ((0, 0, 0), <CommonSeq object>), 
    ##            ...
    ##            ((0, 19, 19), None)]))]))])
    spec_lcs_info_dict = OrderedDict()
    for combo_level, alg_combos in combo_level_alg_combos_dict.items():
        spec_lcs_info_dict[combo_level] = combo_level_dict = OrderedDict()
        for alg_combo in alg_combos:
            combo_level_dict[alg_combo] = OrderedDict()

    #Store consensus sequence results for the spectrum.
    #Example with 3 algorithms total:
    ##spec_consensus_info_dict = OrderedDict([
    ##    (2, OrderedDict([
    ##        (('Novor', 'PepNovo'), dict([
    ##            ('Top-Ranked Sequence', <CommonSeq object>), 
    ##            ('Longest Sequence', <CommonSeq object>)])), 
    ##        (('Novor', 'DeepNovo'), dict([
    ##            ('Top-Ranked Sequence', <CommonSeq object>), 
    ##            ('Longest Sequence', <CommonSeq object>)])), 
    ##        (('PepNovo', 'DeepNovo'), dict([
    ##            ('Top-Ranked Sequence', <CommonSeq object>), 
    ##            ('Longest Sequence', <CommonSeq object>)]))])), 
    ##    (3, OrderedDict([
    ##        (('Novor', 'PepNovo', 'DeepNovo'), dict([
    ##            ('Top-Ranked Sequence', <CommonSeq object>), 
    ##            ('Longest Sequence', <CommonSeq object>)]))]))])
    spec_consensus_info_dict = OrderedDict()
    for combo_level, alg_combos in combo_level_alg_combos_dict.items():
        spec_consensus_info_dict[combo_level] = combo_level_dict = OrderedDict()
        for alg_combo in alg_combos:
            combo_level_dict[alg_combo] = dict()

    for combo_level in rank_comparison_dict:
        rank_comparison_for_combo_level_dict = rank_comparison_dict[combo_level]
        spec_generator_fns_for_combo_level_dict = spec_generator_fns_dict[combo_level]
        spec_consensus_info_for_combo_level_dict = spec_consensus_info_dict[combo_level]

        for alg_combo, rank_comparisons in rank_comparison_for_combo_level_dict.items():
            longest_lcs = None
            top_ranked_lcs = None
            spec_consensus_info_for_alg_combo_dict = spec_consensus_info_for_combo_level_dict[
                alg_combo]

            #If at least one algorithm does not have any sequences for the spectrum, 
            #then comparisons involving the algorithm cannot be performed.
            if len(rank_comparisons) == 0:
                continue

            spec_lcs_info_for_alg_combo_dict = spec_lcs_info_dict[combo_level][alg_combo] = \
                OrderedDict([(rank_comparison, None) for rank_comparison in rank_comparisons])
            did_comparison_for_alg_combo_dict = did_comparison_dict[combo_level][alg_combo]

            #Comparisons of two algorithms are considered separately from >2 algorithm comparisons: 
            #1. With >2 algorithms, not all needed parent consensus sequences may have been found.
            #2. All sequence comparisons need to be performed for >2 algorithms 
            #if the >2 algorithm LCS and TRCS fast-track procedures (see below) were unsuccessful.
            if combo_level == 2:
                seq1_alg = alg_combo[0]
                seq2_alg = alg_combo[1]

                seq1_dict = OrderedDict([
                    ((rank, ), Seq(encoded_seq, (seq1_alg, ), (rank, ))) 
                    if encoded_seq.size > 0 else ((rank, ), None) 
                    for rank, encoded_seq 
                    in enumerate(alg_source_df_for_spec_dict[seq1_alg]['Encoded Sequence'])])
                seq2_dict = OrderedDict([
                    ((rank, ), Seq(encoded_seq, (seq2_alg, ), (rank, ))) 
                    if encoded_seq.size > 0 else ((rank, ), None) 
                    for rank, encoded_seq 
                    in enumerate(alg_source_df_for_spec_dict[seq2_alg]['Encoded Sequence'])])

                seq_comparison_generator = compare_alg_seqs(seq1_dict, seq2_dict)
                spec_generator_fns_for_combo_level_dict[alg_combo] = seq_comparison_generator

                max_possible_lcs_len = min(
                    alg_max_seq_len_dict[seq1_alg], alg_max_seq_len_dict[seq2_alg])
                max_lcs_len = 0
                #Set a consensus rank sum larger than any that is possible.
                min_lcs_rank_sum = 1000
                for lcs, seq1_rank_index, seq2_rank_index in seq_comparison_generator:
                    did_comparison_for_alg_combo_dict[seq1_rank_index + seq2_rank_index] = True

                    if lcs != None:
                        spec_lcs_info_for_alg_combo_dict[lcs.rank_index] = lcs

                        #Find the "top-ranked" LCS for the spectrum.
                        if lcs.rank_sum < min_lcs_rank_sum:
                            min_lcs_rank_sum = lcs.rank_sum
                            spec_consensus_info_for_alg_combo_dict['Top-Ranked Sequence'] = \
                                top_ranked_lcs = lcs

                        #Find the "longest" LCS for the spectrum.
                        if lcs.length > max_lcs_len:
                            max_lcs_len = lcs.length
                            spec_consensus_info_for_alg_combo_dict['Longest Sequence'] = \
                                longest_lcs = lcs

                        #Stop generating LCSs from the algorithm combination 
                        #upon finding the longest possible LCS, 
                        #equal in length to the shortest source algorithm sequence, 
                        #and if it can be shown that this LCS must also be the top-ranked LCS.
                        if max_lcs_len == max_possible_lcs_len and longest_lcs is top_ranked_lcs:
                            if check_if_top_ranked(lcs, alg_seq_count_dict):
                                break

            #Considering consensus sequences of >2 algorithms.
            else:
                seq1_algs = alg_combo[: -1]
                seq2_alg = alg_combo[-1]

                #First check if any comparisons can be performed: 
                #a consensus sequence must have been found 
                #from the first N-1 algorithms under consideration.
                if 'Longest Sequence' not in spec_consensus_info_dict[combo_level - 1][seq1_algs]:
                    #The current algorithm combination can't produce an LCS.
                    break

                #Get the consensus sequences from the first N-1 algorithms 
                #to do a heuristic "fast-tracked" search for consensus sequences 
                #of the algorithm combination under consideration.
                parent_longest_lcs = \
                    spec_consensus_info_dict[combo_level - 1][seq1_algs]['Longest Sequence']
                parent_top_ranked_lcs = \
                    spec_consensus_info_dict[combo_level - 1][seq1_algs]['Top-Ranked Sequence']

                seq2_dict = OrderedDict([
                    ((rank, ), Seq(encoded_seq, (seq2_alg, ), (rank, ))) 
                    if encoded_seq.size > 0 else ((rank, ), None) 
                    for rank, encoded_seq 
                    in enumerate(alg_source_df_for_spec_dict[seq2_alg]['Encoded Sequence'])])

                #Determine whether the longest LCS for the current algorithm combination 
                #can be found from the longest LCS for N-1 algorithms.
                seq1_dict = OrderedDict([(parent_longest_lcs.rank_index, parent_longest_lcs)])
                seq_comparison_generator = compare_alg_seqs(seq1_dict, seq2_dict, combo_level)
                for lcs, _, _ in seq_comparison_generator:
                    if lcs != None:
                        if lcs.length == parent_longest_lcs.length:
                            spec_consensus_info_for_alg_combo_dict['Longest Sequence'] = \
                                longest_lcs = lcs
                            break

                #Determine whether the top-ranked LCS for the current algorithm combination 
                #can be found from the top-ranked LCS for N-1 algorithms.
                seq1_dict = OrderedDict([
                    (parent_top_ranked_lcs.rank_index, parent_top_ranked_lcs)])
                seq_comparison_generator = compare_alg_seqs(seq1_dict, seq2_dict)
                for lcs, _, _ in seq_comparison_generator:
                    if lcs != None:
                        if check_if_top_ranked(lcs, alg_seq_count_dict):
                            spec_consensus_info_for_alg_combo_dict['Top-Ranked Sequence'] = \
                                top_ranked_lcs = lcs
                            break

                #The fast-tracked comparisons are not recorded 
                #in spec_lcs_info_dict or did_comparison_dict.
                #If the target consensus sequences were not found, proceed with all comparisons.
                if longest_lcs == None or top_ranked_lcs == None:
                    seq1_dict = spec_lcs_info_dict[combo_level - 1][seq1_algs]
                    seq_comparison_generator = compare_alg_seqs(seq1_dict, seq2_dict)
                    spec_generator_fns_for_combo_level_dict[alg_combo] = seq_comparison_generator

                    max_possible_lcs_len = min(
                        spec_consensus_info_dict[
                            combo_level - 1][seq1_algs]['Longest Sequence'].length, 
                        alg_max_seq_len_dict[seq2_alg])
                    max_lcs_len = 0
                    #Set a consensus rank sum larger than any that is possible.
                    min_lcs_rank_sum = 1000
                    for lcs, seq1_rank_index, seq2_rank_index in seq_comparison_generator:
                        did_comparison_for_alg_combo_dict[seq1_rank_index + seq2_rank_index] = True
                        if lcs != None:
                            spec_lcs_info_for_alg_combo_dict[lcs.rank_index] = lcs

                            #Find the top-ranked LCS for the spectrum.
                            if lcs.rank_sum < min_lcs_rank_sum:
                                min_lcs_rank_sum = lcs.rank_sum
                                spec_consensus_info_for_alg_combo_dict['Top-Ranked Sequence'] = \
                                    top_ranked_lcs = lcs

                            #Find the longest LCS for the spectrum.
                            if lcs.length > max_lcs_len:
                                max_lcs_len = lcs.length
                                spec_consensus_info_for_alg_combo_dict['Longest Sequence'] = \
                                    longest_lcs = lcs

                            #Stop generating LCSs from the algorithm combination 
                            #upon finding the longest possible LCS, 
                            #equal in length to the shortest source algorithm sequence, 
                            #and if it can be shown that this LCS must also be the top-ranked LCS.
                            if max_lcs_len == max_possible_lcs_len and longest_lcs is top_ranked_lcs:
                                if check_if_top_ranked(lcs, alg_seq_count_dict):
                                    break

            #Recover consensus sequence information.
            if longest_lcs != None:
                if longest_lcs is top_ranked_lcs:
                    recover_lcs_info(
                        longest_lcs, alg_source_df_for_spec_dict=alg_source_df_for_spec_dict)
                    top_ranked_lcs.alg_info_dict = longest_lcs.alg_info_dict
                else:
                    recover_lcs_info(
                        longest_lcs, alg_source_df_for_spec_dict=alg_source_df_for_spec_dict)
                    recover_lcs_info(
                        top_ranked_lcs, alg_source_df_for_spec_dict=alg_source_df_for_spec_dict)

    #Make a DataFrame of results for the spectrum.
    #Make a DataFrame for each consensus sequence and concatenate these as rows.
    result_df = pd.DataFrame()
    result_row_series = []
    spec_info_dict = config.mgf_info_dict[spec_id]
    mz = spec_info_dict['M/Z']
    charge = spec_info_dict['Charge']
    rt = spec_info_dict['Retention Time']
    for combo_level_consensus_info_dict in spec_consensus_info_dict.values():
        for alg_combo, alg_combo_consensus_info_dict in combo_level_consensus_info_dict.items():
            for consensus_type, seq in alg_combo_consensus_info_dict.items():
                seq_info_dict = OrderedDict()
                #Record the source algorithms of the LCS.
                for alg in config.globals['De Novo Algorithms']:
                    if alg in alg_combo:
                        seq_info_dict['Is ' + alg + ' Sequence'] = 1
                    else:
                        seq_info_dict['Is ' + alg + ' Sequence'] = 0
                #Record general information regarding the spectrum.
                seq_info_dict['Spectrum ID'] = spec_id
                seq_info_dict['M/Z'] = mz
                seq_info_dict['Charge'] = charge
                seq_info_dict['Retention Time'] = rt
                seq_info_dict['Encoded Sequence'] = seq.aas
                #Record the type of LCS.
                if consensus_type == 'Top-Ranked Sequence':
                    seq_info_dict['Is Consensus Top-Ranked Sequence'] = 1
                    seq_info_dict['Is Consensus Longest Sequence'] = 0
                elif consensus_type == 'Longest Sequence':
                    seq_info_dict['Is Consensus Top-Ranked Sequence'] = 0
                    seq_info_dict['Is Consensus Longest Sequence'] = 1

                for alg, alg_consensus_info_dict in seq.alg_info_dict.items():
                    seq_info_dict.update(alg_consensus_info_dict)
                result_row_series.append(pd.Series(seq_info_dict))

    if len(result_row_series) == 0:
        return None

    result_df = pd.DataFrame(result_row_series)

    return result_df

def compare_alg_seqs(
    seq1_dict, 
    seq2_dict, 
    combo_level=2, 
    alg_combo=None, 
    did_comparison_dict=None, 
    generator_dict=None):
    '''
    Generator that compares lists of sequences to find LCSs.

    Parameters
    ----------
    seq1_dict : OrderedDict
    seq2_dict : OrderedDict
    combo_level : int
    alg_combo : tuple
    did_comparison_dict : OrderedDict
    generator_dict : OrderedDict

    Yields
    ------
    CommonSeq
    seq1_rank_index : tuple
    seq2_rank_index : tuple

    Returns
    -------
    None
    '''

    for seq1_rank_index, seq1 in seq1_dict.items():

        if combo_level > 2:
            #If the first sequence (N-1 algs) must itself be an LCS, 
            #it may not have been generated due to truncation of the N-1 comparison procedure.
            #Therefore, go back to N-1, N-2, ... generators to search for a first LCS.
            if seq1 == None:
                if not did_comparison_dict[combo_level - 1][alg_combo[: -1]][
                    seq1_rank_index[: -1]]:
                    parent_combo_level = combo_level - 1
                    parent_alg_combo = alg_combo[: -1]
                    seq1.parent_seqs[0] = next(
                        generator_dict[parent_combo_level][parent_alg_combo])
                    did_comparison_dict[parent_alg_combo][parent_alg_combo][
                        seq1_rank_index[: -1]] = True

        for seq2_rank_index, seq2 in seq2_dict.items():
            if seq1 == None or seq2 == None:
                yield None, seq1_rank_index, seq2_rank_index
            else:
                yield seq1.find_lcs(seq2, min_len), seq1_rank_index, seq2_rank_index

    return

def check_if_top_ranked(lcs, alg_seq_count_dict):
    '''
    Determine whether an LCS has the minimum possible rank sum for the algorithm comparison.

    Parameters
    ----------
    lcs : CommonSeq
    alg_seq_count_dict : OrderedDict

    Returns
    -------
    found_top_rank_lcs : bool
    '''

    found_top_rank_lcs = False

    #If all of the LCS's ancestral seqs are rank 0 or only one is rank 1,
    #then it will be impossible to find an LCS with a lower rank sum from subsequent comparisons.
    if lcs.rank_sum <= 1:
        found_top_rank_lcs = True

    #In addition, the following rule governs whether a seq is a top-ranked LCS:
    #The potential rank reduction of the second parent seq in subsequent comparisons is <= 1
    #AND 
    #the total rank increment of the first parent seq in subsequent comparisons is <= 1.
    else:
        seq1_rank_index = lcs.parent_seqs[1].rank_index
        if sum(seq1_rank_index) <= 1:
            #Example, with a 4-alg LCS: 
            #lcs.parent_seqs[0].rank_index: (0, 18, 17)
            potential_rank_reduction = 0
            seq1_algs = lcs.parent_seqs[0].algs
            for i, seq1_rank1 in enumerate(seq1_rank_index[: -1]):
                #First iteration:
                #seq1_rank = 0 == 1 (total number of seqs considered for alg) - 1 = 0
                #CONDITION NOT FULFILLED
                #Second iteration:
                #seq1_rank = 18 < 20 (total number of seqs considered for alg) - 1 = 19
                #CONDITION FULFILLED
                if seq1_rank1 < alg_seq_count_dict[seq1_algs[i]] - 1:
                    #First iteration:
                    #seq1_rank2 = 18
                    #potential_rank_reduction += seq1_rank2
                    #potential_rank_reduction = 18
                    #Second iteration:
                    #seq1_rank2 = 17
                    #potential_rank_reduction += seq1_rank2
                    #potential_rank_reduction = 35
                    for seq1_rank2 in seq1_rank_index[i + 1: ]:
                        potential_rank_reduction += seq1_rank2
                #There is a potential rank reduction from searching for more LCSs.
                if potential_rank_reduction > 1:
                    break
            else:
                found_top_rank_lcs = True

    return found_top_rank_lcs

def recover_lcs_info(lcs, seq_source_series=None, alg_source_df_for_spec_dict=None):
    '''
    For the longest common subsequence (LCS), recover information from source de novo candidates.

    Parameters
    ----------
    lcs : CommonSeq object
    seq_source_series : pandas Series object
    alg_source_df_for_spec_dict : dict mapping strings to pandas DataFrame objects

    Returns
    -------
    None
        The mutable alg_info_dict OrderedDict attribute of lcs is updated.
    '''

    #Loop through each source algorithm from which the LCS is derived.
    for i, alg in enumerate(lcs.algs):
        seq_rank = lcs.rank_index[i]
        seq_source_series = alg_source_df_for_spec_dict[alg].iloc[seq_rank]
        encoded_source_seq = seq_source_series.at['Encoded Sequence']
        source_aa_start = lcs.source_aa_starts[i]
        source_slice_end = source_aa_start + lcs.length
        lcs.alg_info_dict[alg] = alg_info_dict = dict()

        if alg == 'Novor':
            alg_info_dict['Novor Source Sequence'] = seq_source_series.at['Sequence']
            alg_info_dict['Novor Fraction Parent Sequence Length'] = \
                lcs.length / encoded_source_seq.size
            alg_info_dict['De Novo Peptide Ion Mass'] = \
                seq_source_series.at['De Novo Peptide Ion Mass']
            alg_info_dict['De Novo Peptide Ion Mass Error (ppm)'] = \
                seq_source_series.at['De Novo Peptide Ion Mass Error (ppm)']
            alg_info_dict['Novor Peptide Score'] = seq_source_series.at['Novor Peptide Score']
            alg_info_dict['Novor Consensus Amino Acid Scores'] = consensus_aa_scores = \
                seq_source_series.at['Novor Amino Acid Scores'][
                    source_aa_start: source_slice_end]
            alg_info_dict['Novor Average Amino Acid Score'] = np.mean(consensus_aa_scores)
            alg_info_dict['Novor Low-Scoring Dipeptide Count'] = \
                count_low_scoring_peptides(consensus_aa_scores, 2)
            alg_info_dict['Novor Low-Scoring Tripeptide Count'] = \
                count_low_scoring_peptides(consensus_aa_scores, 3)
            isobaric_subseqs_dict, near_isobaric_subseqs_dict = \
                get_potential_substitution_info(lcs.aas, consensus_aa_scores, alg)
            alg_info_dict['Novor Isobaric Mono-Dipeptide Substitution Score'] = \
                isobaric_subseqs_dict[(1, 2)][1]
            alg_info_dict['Novor Isobaric Dipeptide Substitution Score'] = \
                isobaric_subseqs_dict[(2, 2)][1]
            alg_info_dict['Novor Near-Isobaric Mono-Dipeptide Substitution Score'] = \
                near_isobaric_subseqs_dict[(1, 2)][1]
            alg_info_dict['Novor Near-Isobaric Dipeptide Substitution Score'] = \
                near_isobaric_subseqs_dict[(2, 2)][1]
            alg_info_dict['Novor Isobaric Mono-Dipeptide Substitution Average Position'] = \
                isobaric_subseqs_dict[(1, 2)][0]
            alg_info_dict['Novor Isobaric Dipeptide Substitution Average Position'] = \
                isobaric_subseqs_dict[(2, 2)][0]
            alg_info_dict['Novor Near-Isobaric Mono-Dipeptide Substitution Average Position'] = \
                near_isobaric_subseqs_dict[(1, 2)][0]
            alg_info_dict['Novor Near-Isobaric Dipeptide Substitution Average Position'] = \
                near_isobaric_subseqs_dict[(2, 2)][0]

        elif alg == 'PepNovo':
            alg_info_dict['PepNovo Source Sequence Rank'] = seq_rank
            alg_info_dict['PepNovo Source Sequence'] = seq_source_series.at['Sequence']
            alg_info_dict['PepNovo N-terminal Mass Gap'] = \
                seq_source_series.at['PepNovo N-terminal Mass Gap']
            alg_info_dict['PepNovo C-terminal Mass Gap'] = \
                seq_source_series.at['PepNovo C-terminal Mass Gap']
            alg_info_dict['PepNovo Fraction Parent Sequence Length'] = \
                lcs.length / encoded_source_seq.size
            alg_info_dict['PepNovo Rank Score'] = seq_source_series.at['PepNovo Rank Score']
            alg_info_dict['PepNovo Score'] = seq_source_series.at['PepNovo Score']
            alg_info_dict['PepNovo Spectrum Quality Score (SQS)'] = \
                seq_source_series.at['PepNovo Spectrum Quality Score (SQS)']

        elif alg == 'DeepNovo':
            alg_info_dict['DeepNovo Source Sequence Rank'] = seq_rank
            alg_info_dict['DeepNovo Source Sequence'] = seq_source_series.at['Sequence']
            alg_info_dict['DeepNovo Fraction Parent Sequence Length'] = \
                lcs.length / encoded_source_seq.size
            alg_info_dict['DeepNovo Source Average Amino Acid Score'] = \
                seq_source_series.at['DeepNovo Average Amino Acid Score']
            alg_info_dict['DeepNovo Consensus Amino Acid Scores'] = consensus_aa_scores = \
                seq_source_series.at['DeepNovo Amino Acid Scores'][
                    source_aa_start: source_slice_end]
            alg_info_dict['DeepNovo Average Amino Acid Score'] = np.mean(consensus_aa_scores)
            alg_info_dict['DeepNovo Low-Scoring Dipeptide Count'] = \
                count_low_scoring_peptides(consensus_aa_scores, 2)
            alg_info_dict['DeepNovo Low-Scoring Tripeptide Count'] = \
                count_low_scoring_peptides(consensus_aa_scores, 3)
            isobaric_subseqs_dict, near_isobaric_subseqs_dict = \
                get_potential_substitution_info(lcs.aas, consensus_aa_scores, alg)
            alg_info_dict['DeepNovo Isobaric Mono-Dipeptide Substitution Score'] = \
                isobaric_subseqs_dict[(1, 2)][1]
            alg_info_dict['DeepNovo Isobaric Dipeptide Substitution Score'] = \
                isobaric_subseqs_dict[(2, 2)][1]
            alg_info_dict['DeepNovo Near-Isobaric Mono-Dipeptide Substitution Score'] = \
                near_isobaric_subseqs_dict[(1, 2)][1]
            alg_info_dict['DeepNovo Near-Isobaric Dipeptide Substitution Score'] = \
                near_isobaric_subseqs_dict[(2, 2)][1]
            alg_info_dict['DeepNovo Isobaric Mono-Dipeptide Substitution Average Position'] = \
                isobaric_subseqs_dict[(1, 2)][0]
            alg_info_dict['DeepNovo Isobaric Dipeptide Substitution Average Position'] = \
                isobaric_subseqs_dict[(2, 2)][0]
            alg_info_dict[
                'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Average Position'] = \
                near_isobaric_subseqs_dict[(1, 2)][0]
            alg_info_dict['DeepNovo Near-Isobaric Dipeptide Substitution Average Position'] = \
                near_isobaric_subseqs_dict[(2, 2)][0]

    return
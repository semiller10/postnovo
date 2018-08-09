''' Find consensus sequences from de novo sequences. '''

import multiprocessing
import numpy as np
import pandas as pd
import sys

from collections import OrderedDict
from copy import deepcopy
from functools import partial
from itertools import groupby, product
from re import finditer

if 'postnovo' in sys.modules:
    import postnovo.config as config
    import postnovo.utils as utils
else:
    import config
    import utils

progress_count = 0

def make_prediction_df(input_df_dict):
    utils.verbose_print()

    if config.globals['mode'] in ['train', 'optimize']:
        consensus_min_len = config.train_consensus_len
    elif config.globals['mode'] in ['predict', 'test']:
        consensus_min_len = config.globals['min_len']

    tol_prediction_df_list = []
    for tol in config.globals['frag_mass_tols']:
        utils.verbose_print('Setting up', tol, 'Da consensus comparison')
        alg_df_dict = OrderedDict([(alg, input_df_dict[alg][tol]) for alg in config.globals['algs']])
        tol_prediction_df = make_prediction_df_for_tol(consensus_min_len, alg_df_dict, tol)
        tol_prediction_df_list.append(tol_prediction_df)

    prediction_df = pd.concat(tol_prediction_df_list)
    prediction_df['retention time'] = prediction_df.groupby(['spec_id'])[
        'retention time'
    ].transform(max)
    prediction_df = prediction_df[~prediction_df['retention time'].isnull()]

    for tol in config.globals['frag_mass_tols']:
        prediction_df[tol].fillna(0, inplace = True)

    for is_alg_name in config.globals['is_alg_names']:
        prediction_df[is_alg_name].fillna(0, inplace=True)
        prediction_df[is_alg_name] = prediction_df[is_alg_name].astype(int)
    prediction_df.set_index(config.globals['is_alg_names'] + ['spec_id'], inplace = True)
    prediction_df.sort_index(level = ['spec_id'] + config.globals['is_alg_names'], inplace = True)

    return prediction_df

def make_prediction_df_for_tol(consensus_min_len, alg_df_dict, tol):

    for alg, df in alg_df_dict.items():
        encode_seqs(df, consensus_min_len)

    combo_level_alg_dict = make_combo_level_alg_dict(alg_df_dict)
    highest_level_alg_combo = config.globals['alg_combos'][-1]
    ##Example: 
    ##combo_level_alg_dict = odict(
    ##    2: [('novor', 'pn'), ('novor', 'deepnovo'), ('pn', 'deepnovo')], 
    ##    3: [('novor', 'pn', 'deepnovo')]
    ##)
    ##highest_level_alg_combo = ('novor', 'pn', 'deepnovo')

    alg_df_dict = add_measured_mass_col(alg_df_dict)
    alg_df_dict = add_retention_time_cols(alg_df_dict)
    alg_consensus_source_df_dict = make_alg_consensus_source_df_dict(
        highest_level_alg_combo, alg_df_dict
    )
    consensus_spec_list = alg_consensus_source_df_dict[
        highest_level_alg_combo[0]
    ].index.get_level_values('spec_id').tolist()
    one_percent_number_consensus_specs = len(consensus_spec_list) / 100 / config.globals['cpus']

    spec_consensus_info_dict, \
        spec_generator_fns_dict, \
        did_comparison_dict, \
        spec_common_substrings_info_dict = setup_spec_info_dicts(combo_level_alg_dict)
    ##Examples: 
    ##spec_consensus_info_dict = OrderedDict(
    ##    2: OrderedDict(
    ##        ('novor', 'pn'): 
    ##            {
    ##                'longest_cs': {
    ##                    'comparison_seq_starts': None, 
    ##                    'rank_sum': None, 
    ##                    'consensus_len': None, 
    ##                    'alg_ranks': None, 
    ##                    'source_seq_starts': None
    ##                }, 
    ##                'top_rank_cs': {
    ##                    'comparison_seq_starts': None, 
    ##                    'rank_sum': None, 
    ##                    'consensus_len': None, 
    ##                    'alg_ranks': None, 
    ##                    'source_seq_starts': None
    ##                }
    ##            },
    ##        ('novor', 'deepnovo'):
    ##            ...
    ##        ...,
    ##    3: OrderedDict(
    ##        ('novor', 'pn', 'deepnovo'):
    ##            ...
    ##)
    ##spec_generator_fns_dict = OrderedDict(
    ##    2: OrderedDict(
    ##        ('novor', 'pn'): generator_fn,
    ##        ('novor', 'deepnovo'): generator_fn, 
    ##        ('pn', 'deepnovo'): generator_fn
    ##    ),
    ##    3: OrderedDict(
    ##        ('novor', 'pn', 'deepnovo'): generator_fn
    ##    )
    ##)
    ##did_comparison_dict = OrderedDict(
    ##    2: OrderedDict(
    ##        ('novor', 'pn'): OrderedDict(ranks: 0 or 1 for tracking comparison),
    ##        ('novor', 'deepnovo'): OrderedDict(ranks: 0 or 1), 
    ##        ('pn', 'deepnovo'): OrderedDict(ranks: 0 or 1)
    ##    ),
    ##    3: OrderedDict(
    ##        ('novor', 'pn', 'deepnovo'): OrderedDict(ranks: 0 or 1)
    ##    )
    ##)
    ##spec_common_substrings_info_dict = OrderedDict(
    ##2: OrderedDict(
    ##    ('novor', 'pn'): OrderedDict(
    ##        (0, 0): dict(
    ##           'encoded_seq': seq, 
    ##            'parent_substring_starts': [novor seq start index, pn seq start index]
    ##        ), 
    ##        (0, 1): ...,
    ##        ...
    ##        ),
    ##    ('novor', 'deepnovo'): dict(...),
    ##    ('pn', 'deepnovo'): dict(...)
    ##),
    ##3: OrderedDict(
    ##    ('novor', 'pn', 'deepnovo'): dict(...)
    ##)

    rank_comparison_dict = make_rank_comparison_dict(spec_consensus_info_dict)
    ##rank_comparison_dict = OrderedDict(
    ##    2: OrderedDict(
    ##        ('novor', 'pn'): [(0, 0), (0, 1), ..., (0, 18), (0, 19)],
    ##        ('novor', 'deepnovo'): [(0, 0), (0, 1), ..., (0, 18), (0, 19)],
    ##        ('pn', 'deepnovo'): [(0, 0), (0, 1), ..., (19, 18), (19, 19)]
    ##    ),
    ##    3: OrderedDict(
    ##        ('novor', 'pn', 'deepnovo'): [(0, 0, 0), (0, 0, 1), ..., (0, 19, 18), (0, 19, 19)]
    ##    )
    ##)

    alg_max_rank_dict = OrderedDict()
    for alg in config.globals['algs']:
        alg_max_rank_dict[alg] = config.seqs_reported_per_alg_dict[alg]

    progress_count = 0

    ##Single processor method
    #one_percent_number_consensus_specs * config.globals['cpus']
    #print_percent_progress_fn = partial(
    #    utils.print_percent_progress_singlethreaded,
    #    procedure_str = tol + ' Da progress: ',
    #    one_percent_total_count = one_percent_number_consensus_specs
    #)
    #child_initialize(
    #    alg_consensus_source_df_dict,
    #    spec_consensus_info_dict,
    #    spec_generator_fns_dict,
    #    did_comparison_dict,
    #    spec_common_substrings_info_dict,
    #    consensus_min_len,
    #    rank_comparison_dict,
    #    alg_max_rank_dict,
    #    print_percent_progress_fn
    #)
    #grand_spec_prediction_dict_list = []
    #utils.verbose_print('Finding', tol, 'Da consensus sequences')
    #for consensus_spec in consensus_spec_list:
    #    grand_spec_prediction_dict_list.append(
    #        make_spec_prediction_dicts(consensus_spec)
        #)

    #Multiprocessing method
    print_percent_progress_fn = partial(
        utils.print_percent_progress_multithreaded, 
        procedure_str = tol + ' Da progress: ', 
        one_percent_total_count = one_percent_number_consensus_specs, 
        cores = config.globals['cpus']
    )
    mp_pool = multiprocessing.Pool(
        config.globals['cpus'],
        initializer = child_initialize,
        initargs = (
            alg_consensus_source_df_dict,
            spec_consensus_info_dict,
            spec_generator_fns_dict,
            did_comparison_dict,
            spec_common_substrings_info_dict,
            consensus_min_len,
            rank_comparison_dict,
            alg_max_rank_dict,
            print_percent_progress_fn
        )
    )
    utils.verbose_print('Finding', tol, 'Da consensus sequences')
    grand_spec_prediction_dict_list = mp_pool.map(make_spec_prediction_dicts, consensus_spec_list)
    mp_pool.close()
    mp_pool.join()

    spec_prediction_dict_list = [
        seq_prediction_dict for spec_prediction_dict_list in grand_spec_prediction_dict_list
        for seq_prediction_dict in spec_prediction_dict_list
    ]
    tol_prediction_df = pd.DataFrame().from_dict(spec_prediction_dict_list)
    tol_prediction_df[tol] = 1

    return tol_prediction_df

def child_initialize(_alg_consensus_source_df_dict,
                     _spec_consensus_info_dict,
                     _spec_generator_fns_dict,
                     _did_comparison_dict,
                     _spec_common_substrings_info_dict,
                     _consensus_min_len,
                     _rank_comparison_dict,
                     _alg_max_rank_dict,
                     _print_percent_progress_fn):

     global alg_consensus_source_df_dict
     global spec_consensus_info_dict
     global spec_generator_fns_dict
     global did_comparison_dict
     global spec_common_substrings_info_dict
     global consensus_min_len
     global rank_comparison_dict
     global alg_max_rank_dict
     global print_percent_progress_fn

     alg_consensus_source_df_dict = _alg_consensus_source_df_dict
     spec_consensus_info_dict = _spec_consensus_info_dict
     spec_generator_fns_dict = _spec_generator_fns_dict
     did_comparison_dict = _did_comparison_dict
     spec_common_substrings_info_dict = _spec_common_substrings_info_dict
     consensus_min_len = _consensus_min_len
     rank_comparison_dict = _rank_comparison_dict
     alg_max_rank_dict = _alg_max_rank_dict
     print_percent_progress_fn = _print_percent_progress_fn

def make_spec_prediction_dicts(consensus_spec):

    print_percent_progress_fn()

    spec_prediction_dict_list = []

    rank_comparison_dict_copy = deepcopy(rank_comparison_dict)
    alg_consensus_source_df_for_spec_dict = OrderedDict()
    max_seq_len_dict = OrderedDict()
    #To make a consensus seq, at least two algs need seqs of at least min length.
    short_seq_alg_count = 0
    for alg in alg_consensus_source_df_dict:
        alg_consensus_source_df_for_spec = alg_consensus_source_df_dict[alg].loc[consensus_spec]
        #DeepNovo can have a variable number of seqs per spectrum 
        #due to the consolidation of Ile/Leu.
        num_seqs = len(alg_consensus_source_df_for_spec)
        if alg == 'deepnovo':
            #Initialize with the maximum possible number of seqs.
            alg_max_rank_dict[alg] = config.seqs_reported_per_alg_dict['deepnovo']
            if num_seqs < alg_max_rank_dict[alg]:
                alg_max_rank_dict['deepnovo'] = len(alg_consensus_source_df_for_spec)

                #Changing the number of DeepNovo seqs also changes the number of seq comparisons.
                #Retabulate the rank comparisons that can be performed.
                for combo_level, rank_comparison_dict_for_combo_level in \
                    rank_comparison_dict_copy.items():
                    for alg_combo in rank_comparison_dict_for_combo_level:
                        if 'deepnovo' in alg_combo:
                            alg_ranks_ranges = []
                            last_ranks_list = []

                            for alg in alg_combo:
                                last_ranks_list.append(alg_max_rank_dict[alg])
                                alg_ranks_ranges.append(range(last_ranks_list[-1]))
                            rank_comparison_dict_for_combo_level[alg_combo] = list(
                                product(*alg_ranks_ranges)
                            )

        #Expect a set number of seqs per spectrum from Novor and PepNovo.
        #Occasionally, fewer are reported for poor-quality spectra, 
        #in which case, ignore predictions from ALL algs, returning an empty list.
        else:
            if num_seqs < alg_max_rank_dict[alg]:
                return []

        #I am currently retaining seqs that do not meet the minimum length
        #in the consensus source table.
        #This allows the rank index ranges to be contiguous,
        #unlike if the short seqs were removed from the tables.
        max_seq_len_dict[alg] = alg_consensus_source_df_for_spec['encoded seq'].map(len).max()

        if max_seq_len_dict[alg] == 0:
            short_seq_alg_count += 1
        else:
            #Information on the top single-alg seq prediction is reported.
            #Seqs shorter than the min length have an empty list for the encoded seq.
            if alg_consensus_source_df_for_spec.at[0, 'encoded seq'] != []:
                spec_prediction_dict_list.append(
                    make_seq_prediction_dict(
                        consensus_spec,
                        alg_consensus_source_df_for_spec=alg_consensus_source_df_for_spec,
                        alg=alg
                    )
                )
        alg_consensus_source_df_for_spec_dict[alg] = alg_consensus_source_df_for_spec

    if short_seq_alg_count >= len(alg_consensus_source_df_dict) - 1:
        return spec_prediction_dict_list

    #Find consensus seqs between increasing numbers of algs (2, 3, etc.).
    spec_generator_fns_dict_copy = deepcopy(spec_generator_fns_dict)
    spec_consensus_info_dict_copy = deepcopy(spec_consensus_info_dict)
    for combo_level in spec_consensus_info_dict:
        rank_comparison_for_combo_level_dict = rank_comparison_dict_copy[combo_level]
        spec_generator_fns_for_combo_level_dict = spec_generator_fns_dict_copy[combo_level]
        spec_consensus_info_for_combo_level_dict = spec_consensus_info_dict_copy[combo_level]

        for alg_combo in spec_consensus_info_for_combo_level_dict:
            rank_comparisons = rank_comparison_for_combo_level_dict[alg_combo]
            spec_consensus_info_for_alg_combo_dict = spec_consensus_info_for_combo_level_dict[
                alg_combo
            ]

            spec_common_substrings_info_for_alg_combo_dict = \
                spec_common_substrings_info_dict[combo_level][alg_combo] = \
                OrderedDict([
                    (rank_index, {'encoded_seq': None, 'parent_substring_starts': []}) 
                    for rank_index in rank_comparisons
                ])

            did_comparison_for_alg_combo_dict = did_comparison_dict[combo_level][alg_combo] =\
                OrderedDict([(rank_index, False) for rank_index in rank_comparisons])

            longest_cs_dict = spec_consensus_info_for_alg_combo_dict['longest_cs']
            top_rank_cs_dict = spec_consensus_info_for_alg_combo_dict['top_rank_cs']
            #longest_cs_dict = deepcopy(spec_consensus_info_for_alg_combo_dict['longest_cs'])
            #top_rank_cs_dict = deepcopy(spec_consensus_info_for_alg_combo_dict['top_rank_cs'])

            #Consensus seqs of length 2 are considered separately from seqs involving more algs.
            #This is because 
            #1. Not all necessary parent consensus seqs may have been generated for the latter.
            #2. All seq comparisons need to be performed for the latter
            #if the >2-alg comparison LCS and TRCS fast-track procedures were unsuccessful.
            if combo_level == 2:
                first_seq_alg = alg_combo[0]
                second_seq_alg = alg_combo[1]

                #Additional info for the first seq is kept 
                #on the start indices of common substrings.
                #This is only important for consensus sequences,
                #but the format must be the same for single-alg first seqs 
                #in do_seq_comparisons fn.
                first_seq_info_dict = OrderedDict([
                    ((rank,), {'encoded_seq': encoded_seq, 'parent_substring_starts': [0]})
                    for rank, encoded_seq
                    in enumerate(
                        alg_consensus_source_df_for_spec_dict[first_seq_alg]['encoded seq']
                    )
                ])
                second_encoded_seq_dict = OrderedDict([
                    ((rank,), encoded_seq) for rank, encoded_seq
                    in enumerate(
                        alg_consensus_source_df_for_spec_dict[second_seq_alg]['encoded seq']
                    )
                ])

                max_possible_cs_len = min(
                    max_seq_len_dict[first_seq_alg], max_seq_len_dict[second_seq_alg]
                )

                seq_comparison_generator = do_seq_comparisons(
                    first_seq_info_dict, second_encoded_seq_dict, consensus_min_len
                )
                spec_generator_fns_for_combo_level_dict[alg_combo] = seq_comparison_generator
                longest_cs_len = 0
                min_cs_rank_sum = 1000

                for first_seq_cs_start_position, \
                    new_first_seq_source_starts, \
                    second_seq_cs_start_position, \
                    cs_len, first_seq_rank_index, \
                    second_seq_rank_index in seq_comparison_generator:
                    did_comparison_for_alg_combo_dict[
                        first_seq_rank_index + second_seq_rank_index
                    ] = True

                    if cs_len is not None:
                        longest_cs_len, min_cs_rank_sum = parse_generator_output(
                            longest_cs_len,
                            min_cs_rank_sum,
                            first_seq_info_dict,
                            spec_common_substrings_info_for_alg_combo_dict,
                            first_seq_cs_start_position,
                            new_first_seq_source_starts, 
                            second_seq_cs_start_position,
                            cs_len,
                            first_seq_rank_index,
                            second_seq_rank_index,
                            longest_cs_dict,
                            top_rank_cs_dict
                        )

                        #Stop searching for CS's when the longest possible CS is found 
                        #(equal to len of shortest parent seq)
                        #and when it can be shown that the LCS must also be the T-R CS.
                        if longest_cs_len == max_possible_cs_len and \
                            longest_cs_dict['alg_ranks'] == top_rank_cs_dict['alg_ranks']:
                            top_rank_cs_found = False

                            #If all of the parent seqs are rank 0
                            #or only 1 of the parent seqs is rank 1,
                            #then it will be impossible to find a CS with a lower sum rank,
                            #so stop searching for additional CS's.
                            if sum(first_seq_rank_index + second_seq_rank_index) <= 1:
                                top_rank_cs_found = True

                            #In addition, the following rule shows whether a seq is a T-R CS:
                            #The potential rank reduction of the second seq <= 1
                            #AND 
                            #there is no total rank increment of the first seq's parent seqs > 1.
                            else:
                                if sum(second_seq_rank_index) <= 1:
                                    #Example: 
                                    #first seq ranks: (0, 18, 17)
                                    potential_rank_reduction = 0
                                    for first_seq_parent_index1, \
                                        first_seq_parent_rank1 in enumerate(
                                            first_seq_rank_index[: -1]
                                    ):
                                        #first seq parent rank 1 = 0 == max rank = 1 - 1 = 0
                                        #CONDITION NOT FULFILLED
                                        #first seq parent rank 1 = 18 < max rank = 20 - 1 = 19
                                        #CONDITION FULFILLED
                                        if first_seq_parent_rank1 < alg_max_rank_dict[
                                            first_seq_algs[first_seq_parent_index1]
                                            ] - 1:
                                            #potential rank reduction = 
                                            #first seq parent rank 2 = 18
                                            #potential rank reduction + 17 = 35
                                            for first_seq_parent_rank2 in first_seq_rank_index[
                                                first_seq_parent_index1 + 1: 
                                            ]:
                                                potential_rank_reduction += first_seq_parent_rank2
                                        #There is a potential rank reduction 
                                        #from continuing the search for consensus seqs.
                                        if potential_rank_reduction > 1:
                                            break
                                    else:
                                        top_rank_cs_found = True


                            if top_rank_cs_found:
                                break

            #Considering consensus seqs of >2 algs
            else:
                first_seq_algs = alg_combo[: -1]
                second_seq_alg = alg_combo[-1]

                #Get the first seq LCS.
                first_seq_lcs_dict = \
                    spec_consensus_info_dict_copy[combo_level - 1][first_seq_algs]['longest_cs']
                if 'encoded_consensus_seq' in first_seq_lcs_dict:
                    first_seq_lcs_rank_index = first_seq_lcs_dict['alg_ranks']
                    first_seq_encoded_lcs = first_seq_lcs_dict['encoded_consensus_seq']
                    first_seq_lcs_source_starts = first_seq_lcs_dict['source_seq_starts']
                else:
                    first_seq_lcs_rank_index = None
                    first_seq_encoded_lcs = None
                    first_seq_lcs_source_starts = None

                #If any constituent alg combo doesn't have an LCS,
                #then there are no consensus sequences for this combo level.
                if first_seq_encoded_lcs is None:
                    break

                #Get the T-R CS for the first seq algs.
                first_seq_trcs_dict = \
                    spec_consensus_info_dict_copy[combo_level - 1][first_seq_algs]['top_rank_cs']
                if 'encoded_consensus_seq' in first_seq_trcs_dict:
                    first_seq_trcs_rank_index = first_seq_trcs_dict['alg_ranks']
                    first_seq_encoded_trcs = first_seq_trcs_dict['encoded_consensus_seq']
                    first_seq_trcs_source_starts = first_seq_trcs_dict['source_seq_starts']
                else:
                    first_seq_trcs_rank_index = None
                    first_seq_encoded_trcs = None
                    first_seq_trcs_source_starts = None

                #Get all of the CS's and source seq starts for the first seq algs.
                first_seq_info_dict = deepcopy(
                    spec_common_substrings_info_dict[combo_level - 1][first_seq_algs]
                )
                #Get seqs for the second seq alg.
                alg_rank_encoded_seq_list = \
                    alg_consensus_source_df_for_spec_dict[second_seq_alg]['encoded seq']
                second_encoded_seq_dict = OrderedDict([
                    ((rank,), encoded_seq) for rank, encoded_seq in 
                    enumerate(alg_rank_encoded_seq_list)
                ])

                #Initialize the seq comparison generator.
                seq_comparison_generator = do_seq_comparisons(
                    first_seq_info_dict, 
                    second_encoded_seq_dict, 
                    consensus_min_len, 
                    first_seq_lcs_rank_index=first_seq_lcs_rank_index,
                    first_seq_encoded_lcs=first_seq_encoded_lcs,
                    first_seq_lcs_source_starts=first_seq_lcs_source_starts, 
                    first_seq_trcs_rank_index=first_seq_trcs_rank_index,
                    first_seq_encoded_trcs=first_seq_encoded_trcs, 
                    first_seq_trcs_source_starts=first_seq_trcs_source_starts, 
                    alg_max_rank_dict=alg_max_rank_dict, 
                    first_seq_algs=first_seq_algs
                )
                spec_generator_fns_for_combo_level_dict[alg_combo] = seq_comparison_generator
                
                #Fast-track LCS and TRCS comparisons
                #If these seqs do not yield a new LCS and TRCS, 
                #the comparisons are later repeated so the results can be recorded.
                #This is an overall efficiency that compensates
                #for that later inefficiency of repeating these two comparisons.

                #LCS
                first_seq_cs_start_position, \
                    new_first_seq_source_starts, \
                    second_seq_cs_start_position, \
                    cs_len, \
                    first_seq_rank_index, \
                    second_seq_rank_index = next(seq_comparison_generator)
                if first_seq_cs_start_position != None:
                    consensus_seq_rank_index = first_seq_rank_index + second_seq_rank_index
                    longest_cs_dict['alg_ranks'] = consensus_seq_rank_index
                    longest_cs_dict['rank_sum'] = sum(consensus_seq_rank_index)
                    longest_cs_dict['comparison_seq_starts'] = (
                        first_seq_cs_start_position, second_seq_cs_start_position
                    )
                    longest_cs_dict['source_seq_starts'] = new_first_seq_source_starts
                    longest_cs_dict['consensus_len'] = cs_len

                #TRCS
                #A separate comparison is performed for the TRCS even if the LCS is also the TRCS
                #due to the excessive complexity of determining that.
                first_seq_cs_start_position, \
                    new_first_seq_source_starts, \
                    second_seq_cs_start_position, \
                    cs_len, \
                    first_seq_rank_index, \
                    second_seq_rank_index = next(seq_comparison_generator)
                if first_seq_cs_start_position != None:
                    consensus_seq_rank_index = first_seq_rank_index + second_seq_rank_index
                    top_rank_cs_dict['alg_ranks'] = consensus_seq_rank_index
                    top_rank_cs_dict['rank_sum'] = sum(consensus_seq_rank_index)
                    top_rank_cs_dict['comparison_seq_starts'] = (
                        first_seq_cs_start_position, second_seq_cs_start_position
                    )
                    top_rank_cs_dict['source_seq_starts'] = new_first_seq_source_starts
                    top_rank_cs_dict['consensus_len'] = cs_len
                    
                #Get the generators and comparison records 
                #for the first seqs and their parent first seqs.
                first_seq_spec_generator_fns_dict = OrderedDict()
                first_seq_did_comparison_dict = OrderedDict()
                first_seq_spec_common_substrings_info_dict = OrderedDict()
                for parent_seq_combo_level in range(combo_level - 1, 1, -1):
                    first_seq_spec_generator_fns_dict[parent_seq_combo_level] = \
                        spec_generator_fns_dict_copy[parent_seq_combo_level][
                            first_seq_algs[: parent_seq_combo_level]
                        ]
                    first_seq_did_comparison_dict[parent_seq_combo_level] = \
                        did_comparison_dict[parent_seq_combo_level][
                            first_seq_algs[: parent_seq_combo_level]
                        ]
                    first_seq_spec_common_substrings_info_dict[parent_seq_combo_level] = \
                        spec_common_substrings_info_dict[parent_seq_combo_level][
                            first_seq_algs[: parent_seq_combo_level]
                        ]

                first_seq_rank_indices = list(first_seq_info_dict.keys())
                last_comparison_index = len(first_seq_rank_indices) - 1

                longest_cs_len = 0
                min_cs_rank_sum = 1000

                #Even if the LCS and TRCS were found earlier,
                #perform the first comparison for the purposes of
                #the operation of the consensus first seq comparison check.
                for comparison_index, generator_output in enumerate(seq_comparison_generator):

                    first_seq_cs_start_position, \
                        new_first_seq_source_starts, \
                        second_seq_cs_start_position, \
                        cs_len, \
                        first_seq_rank_index, \
                        second_seq_rank_index = generator_output
                    did_comparison_for_alg_combo_dict[
                        first_seq_rank_index + second_seq_rank_index
                    ] = True

                    if cs_len is not None:
                        parse_generator_output(
                            longest_cs_len,
                            min_cs_rank_sum,
                            first_seq_info_dict,
                            spec_common_substrings_info_for_alg_combo_dict,
                            first_seq_cs_start_position, 
                            new_first_seq_source_starts, 
                            second_seq_cs_start_position,
                            cs_len,
                            first_seq_rank_index,
                            second_seq_rank_index,
                            longest_cs_dict,
                            top_rank_cs_dict
                        )
                                            
                    #Check whether the first seq in the next comparison was ever considered.
                    #The next first seq cannot exist if its comparison was not performed.
                    if comparison_index < last_comparison_index:
                        first_seq_next_rank_index = first_seq_rank_indices[comparison_index + 1]
                        first_seq_combo_level = combo_level - 1
                        if not first_seq_did_comparison_dict[first_seq_combo_level][
                            first_seq_next_rank_index
                        ]:
                            #Perform the first seq comparison 
                            #and all further necessary parent seq comparisons.
                            do_parent_comparisons(
                                first_seq_combo_level, 
                                alg_combo, 
                                first_seq_next_rank_index, 
                                first_seq_spec_generator_fns_dict, 
                                first_seq_did_comparison_dict, 
                                first_seq_spec_common_substrings_info_dict, 
                                alg_consensus_source_df_for_spec_dict
                            )

                    #Break if LCS/TRCS was found
                    if longest_cs_dict['alg_ranks'] != None and \
                    top_rank_cs_dict['alg_ranks'] != None:
                        break

            #Back to consideration of both 2- and >2-alg consensus seqs:
            #If an LCS meeting the min length threshold was found
            if longest_cs_dict['alg_ranks'] is not None:

                #If considering a CS that is both LCS and T-R CS
                if longest_cs_dict['alg_ranks'] == top_rank_cs_dict['alg_ranks']:
                    cs_prediction_dict = make_seq_prediction_dict(
                        consensus_spec, 
                        alg_consensus_source_df_for_spec_dict=\
                            alg_consensus_source_df_for_spec_dict, 
                        cs_info_dict=longest_cs_dict, 
                        cs_type_list=['longest', 'top rank'], 
                        alg_combo=alg_combo, 
                    )
                    top_rank_cs_dict['encoded_consensus_seq'] = \
                        longest_cs_dict['encoded_consensus_seq']
                    spec_prediction_dict_list.append(cs_prediction_dict)
                #Else the LCS and T-R CS are different seqs
                else:
                    longest_cs_prediction_dict = make_seq_prediction_dict(
                        consensus_spec,
                        alg_consensus_source_df_for_spec_dict=\
                            alg_consensus_source_df_for_spec_dict,
                        cs_info_dict=longest_cs_dict,
                        cs_type_list=['longest'],
                        alg_combo=alg_combo,
                    )
                    top_rank_cs_prediction_dict = make_seq_prediction_dict(
                        consensus_spec,
                        alg_consensus_source_df_for_spec_dict=\
                            alg_consensus_source_df_for_spec_dict,
                        cs_info_dict=top_rank_cs_dict,
                        cs_type_list=['top rank'],
                        alg_combo=alg_combo,
                    )
                    spec_prediction_dict_list.append(longest_cs_prediction_dict)
                    spec_prediction_dict_list.append(top_rank_cs_prediction_dict)
                    
    ##Reset the global spec_consensus_info_dict.
    #for spec_consensus_info_for_combo_level_dict in spec_consensus_info_dict.values():
    #    for spec_consensus_info_for_alg_combo_dict in 
    #    spec_consensus_info_for_combo_level_dict.values():
    #        for cs_dict in spec_consensus_info_for_alg_combo_dict.values():
    #            for info in cs_dict:
    #                cs_dict[info] = None
    
    return spec_prediction_dict_list

def parse_generator_output(
    longest_cs_len,
    min_cs_rank_sum,
    first_seq_info_dict,
    spec_common_substrings_info_for_alg_combo_dict,
    first_seq_cs_start_position,
    new_first_seq_source_starts, 
    second_seq_cs_start_position,
    cs_len,
    first_seq_rank_index,
    second_seq_rank_index,
    longest_cs_dict,
    top_rank_cs_dict
):

    #Record the common substrings and source seq aa start indices from each comparison.
    #Example: 
    #Common substring: 
    #spec_common_substrings_info_for_alg_combo_dict[3][((1, 19), (11, ))]['seq'] = 
    #    first_seq_info_dict[(1, 19)][3: 3 + 10] = np.array([4, 7, 10, 14, 19, 3, 14, 5, 6, 2])
    #First seq parent starts:
    #spec_common_substrings_info_for_alg_combo_dict[3][((1, 19), (11, ))][
    #    'parent_substring_starts'
    #] = [5, 2] + [4] # cs starts at aa #5 in novor, aa #2 in pn, and aa #4 in deepnovo
    cs_rank_index = first_seq_rank_index + second_seq_rank_index
    spec_common_substrings_info_for_rank_index_dict = \
        spec_common_substrings_info_for_alg_combo_dict[cs_rank_index]
    spec_common_substrings_info_for_rank_index_dict['encoded_seq'] = \
        first_seq_info_dict[first_seq_rank_index]['encoded_seq'][
            first_seq_cs_start_position: first_seq_cs_start_position + cs_len
        ]
    cs_source_starts = new_first_seq_source_starts + [second_seq_cs_start_position]
    spec_common_substrings_info_for_rank_index_dict['parent_substring_starts'] = cs_source_starts

    #Each rank index is a tuple.
    first_rank_sum = sum(first_seq_rank_index)
    second_rank_sum = sum(second_seq_rank_index)
    cs_rank_sum = first_rank_sum + second_rank_sum

    #LCS found
    if cs_len > longest_cs_len:
        longest_cs_len = cs_len
        longest_cs_dict['alg_ranks'] = first_seq_rank_index + second_seq_rank_index
        longest_cs_dict['rank_sum'] = cs_rank_sum
        longest_cs_dict['comparison_seq_starts'] = (
            first_seq_cs_start_position, second_seq_cs_start_position
        )
        longest_cs_dict['source_seq_starts'] = cs_source_starts
        longest_cs_dict['consensus_len'] = cs_len

    #Top-ranking (T-R) seq found
    if cs_rank_sum < min_cs_rank_sum:
        min_cs_rank_sum = cs_rank_sum
        top_rank_cs_dict['alg_ranks'] = first_seq_rank_index + second_seq_rank_index
        top_rank_cs_dict['rank_sum'] = cs_rank_sum
        top_rank_cs_dict['comparison_seq_starts'] = (
            first_seq_cs_start_position, second_seq_cs_start_position
        )
        top_rank_cs_dict['source_seq_starts'] = cs_source_starts
        top_rank_cs_dict['consensus_len'] = cs_len

    return longest_cs_len, min_cs_rank_sum

def do_parent_comparisons(
    combo_level, 
    alg_combo, 
    rank_index, 
    first_seq_spec_generator_fns_dict, 
    first_seq_did_comparison_dict,
    first_seq_spec_common_substrings_info_dict,
    alg_consensus_source_df_for_spec_dict
):

    #If the comparison involves a first seq that is a consensus seq,
    #then check whether the comparison to generate that first seq was itself performed.
    if combo_level > 2:
        first_seq_combo_level = combo_level - 1
        first_seq_alg_combo = alg_combo[: -1]
        first_seq_rank_index = rank_index[: -1]
        if not first_seq_did_comparison_dict[combo_level][first_seq_rank_index]:
            # Perform the comparison
            do_parent_comparisons(
                first_seq_combo_level, 
                first_seq_alg_combo, 
                first_seq_rank_index, 
                first_seq_spec_generator_fns_dict[first_seq_combo_level],
                spec_common_substrings_info_dict,
                alg_consensus_source_df_for_spec_dict
                )
            first_seq_cs_start_position, \
                new_first_seq_source_starts, \
                second_seq_cs_start_position, \
                cs_len, \
                first_seq_rank_index, \
                second_seq_rank_index = \
                next(first_seq_spec_generator_fns_dict[combo_level])
            first_seq_did_comparison_dict[combo_level][rank_index] = True
            if cs_len is not None:
                # Record the consensus seq found from the comparison
                first_seq_spec_common_substrings_info_dict[combo_level][rank_index]['encoded_seq']\
                   = first_seq_spec_common_substrings_info_dict[combo_level - 1][rank_index[: -1]]\
                   ['encoded_seq'][first_seq_cs_start_position: first_seq_cs_start_position + cs_len]
                # Record the consensus seq source seq start indices
                parent_substring_starts = first_seq_spec_common_substrings_info_dict\
                    [combo_level - 1][rank_index[: -1]]['parent_substring_starts']
                first_seq_spec_common_substrings_info_dict[combo_level][rank_index]\
                    ['parent_substring_starts'] = new_first_seq_source_starts

    #Else the first seq is an irreducible single-alg seq
    else:
        first_seq_cs_start_position, \
            new_first_seq_source_starts, \
            second_seq_cs_start_position, \
            cs_len, \
            first_seq_rank_index, \
            second_seq_rank_index =\
            next(first_seq_spec_generator_fns_dict[combo_level])
        first_seq_did_comparison_dict[combo_level][rank_index] = True
        if cs_len is not None:
            first_seq_spec_common_substrings_info_dict[combo_level][rank_index]['encoded_seq'] = \
                alg_consensus_source_df_for_spec_dict[alg_combo[0]]['encoded seq'][rank_index[0]]\
                [first_seq_cs_start_position: first_seq_cs_start_position + cs_len]
            first_seq_spec_common_substrings_info_dict[combo_level][rank_index]\
                ['parent_substring_starts'] = new_first_seq_source_starts
    return

def do_seq_comparisons(
    first_seq_info_dict,
    second_encoded_seq_dict,
    consensus_min_len,
    first_seq_lcs_rank_index=None,
    first_seq_encoded_lcs=None, 
    first_seq_lcs_source_starts=None, 
    first_seq_trcs_rank_index=None,
    first_seq_encoded_trcs=None, 
    first_seq_trcs_source_starts=None,  
    alg_max_rank_dict=None,
    first_seq_algs=None
):
    #yield: first_seq_lcs_start_position, second_seq_lcs_start_position, lcs_len, 
    #first_seq_rank_index, second_seq_rank_index

    def do_seq_comparison(
        first_seq_rank_index, 
        first_encoded_seq, 
        first_seq_parent_starts,
        second_seq_rank_index, 
        second_encoded_seq
    ):

        if first_encoded_seq is None or second_encoded_seq is None:
            return None, None, None, None, first_seq_rank_index, second_seq_rank_index

        if len(first_encoded_seq) == 0 or len(second_encoded_seq) == 0:
            return None, None, None, None, first_seq_rank_index, second_seq_rank_index

        else:
            #Make the seq vectors orthogonal.
            first_encoded_seq = first_encoded_seq.reshape(first_encoded_seq.size, 1)
            #Fill in the 2D matrix formed by the dimensions of the seq vectors 
            #with the aa's of the second seq.
            tiled_second_encoded_seq = np.tile(second_encoded_seq, (first_encoded_seq.size, 1))
            #Project the first seq over the 2D matrix to find any identical aa's.
            match_arr = np.equal(first_encoded_seq, tiled_second_encoded_seq).astype(int)

            #Find any common substrings, which are diagonals of True values in match_arr.
            #Diagonal index 0 is the main diagonal.
            #Negatively indexed diagonals lie below the main diagonal.
            #Consideration of diagonals can be restricted to those
            #that can contain common substrings longer than the minimum length.
            diags = [
                match_arr.diagonal(d)
                for d in range(
                    -len(first_encoded_seq) + consensus_min_len, 
                    len(second_encoded_seq) - consensus_min_len
                )
            ]

            #Identify common substrings in the diagonals.
            lcs_len = consensus_min_len
            found_long_consensus = False
            #Loop through the bottom left min length diagonal to 
            #the upper right min length diagonal.
            for diag_index, diag in enumerate(diags):
                #Create and loop through two groups of Trues (common substrings) and Falses
                #from the elements of the diagonal.
                for match_status, diag_group in groupby(diag):
                    #If considering a common substring
                    if match_status:
                        consensus_len = sum(diag_group)
                        #Retain the longest common substring, preferring the upper-rightmost LCS
                        #if multiple LCS's of equal length are present.
                        if consensus_len >= lcs_len:
                            found_long_consensus = True
                            lcs_len = consensus_len
                            #Record the diagonal's index, 
                            #starting from the zero of the lower leftmost corner.
                            lcs_diag_index = diag_index
                            lcs_diag = diag

            if found_long_consensus:
                #Find where the LCS resides in the selected diagonal.
                #Take the first LCS if multiple LCS's of equal length are present in the diagonal.
                for diag_aa_position in range(lcs_diag.size - lcs_len + 1):
                    for lcs_aa_position in range(lcs_len):
                        if not lcs_diag[diag_aa_position + lcs_aa_position]:
                            break
                    else:
                        diag_lcs_start_position = diag_aa_position
                        break

                #Determine the position of the first LCS aa in the first and second seqs.
                #Reindex the LCS-containing diagonal to the main diagonal.
                upper_left_diag_index = first_encoded_seq.size - consensus_min_len
                relative_lcs_diag_index = lcs_diag_index - upper_left_diag_index
                #Negatively indexed diagonals lie below the main diagonal.
                if relative_lcs_diag_index < 0:
                    first_seq_lcs_start_position = \
                        diag_lcs_start_position - relative_lcs_diag_index
                    second_seq_lcs_start_position = diag_lcs_start_position
                else:
                    first_seq_lcs_start_position = diag_lcs_start_position
                    second_seq_lcs_start_position = \
                        relative_lcs_diag_index + diag_lcs_start_position

                #Pause the loop,
                #returning the position of the first aa in the first and second seqs,
                #the length of the LCS,
                #and the ranks of the first and second seqs.
                return first_seq_lcs_start_position, \
                    [
                        first_seq_parent_start + first_seq_lcs_start_position for 
                        first_seq_parent_start in first_seq_parent_starts
                    ], \
                    second_seq_lcs_start_position, \
                    lcs_len, \
                    first_seq_rank_index, \
                    second_seq_rank_index
            else:
                return None, None, None, None, first_seq_rank_index, second_seq_rank_index

    if first_seq_lcs_rank_index:
        max_possible_cs_len = first_seq_encoded_lcs.size
        for second_seq_rank_index, second_encoded_seq in second_encoded_seq_dict.items():
            first_seq_cs_start_position, \
                new_first_seq_source_starts, \
                second_seq_cs_start_position, \
                cs_len, \
                first_seq_rank_index, \
                second_seq_rank_index = \
                do_seq_comparison(
                    first_seq_lcs_rank_index, 
                    first_seq_encoded_lcs, 
                    first_seq_lcs_source_starts, 
                    second_seq_rank_index, 
                    second_encoded_seq
                )

            if first_seq_cs_start_position is not None:
                if cs_len == max_possible_cs_len:
                    yield first_seq_cs_start_position, \
                        new_first_seq_source_starts, \
                        second_seq_cs_start_position, \
                        cs_len, \
                        first_seq_rank_index, \
                        second_seq_rank_index
                    break
        else:
            yield None, None, None, None, first_seq_rank_index, second_seq_rank_index

    if first_seq_trcs_rank_index:
        first_rank_sum = sum(first_seq_trcs_rank_index)
        for second_seq_rank_index, second_encoded_seq in second_encoded_seq_dict.items():
            first_seq_cs_start_position, \
                new_first_seq_source_starts, \
                second_seq_cs_start_position, \
                cs_len, \
                first_seq_rank_index, \
                second_seq_rank_index = \
                do_seq_comparison(
                    first_seq_trcs_rank_index, 
                    first_seq_encoded_trcs, 
                    first_seq_trcs_source_starts, 
                    second_seq_rank_index, 
                    second_encoded_seq
                )

            #If a consensus seq is found, determine whether it must be a top-ranked consensus seq.
            if first_seq_cs_start_position is not None:
                #If all of the parent seqs are rank 0, 
                #or only 1 of the parent seqs is rank 1, 
                #then it will be impossible to find a CS with a lower sum rank, 
                #so stop searching for additional CS's.
                top_rank_cs_found = False
                second_rank_sum = sum(second_seq_rank_index)
                if first_rank_sum + second_rank_sum <= 1:
                    top_rank_cs_found = True

                #In addition, the following rule shows whether a seq is a T-R CS: 
                #The potential rank reduction of the second seq <= 1
                #AND 
                #there is no total rank increment of the first seq's parent seqs > 1.
                else:
                    if sum(second_seq_rank_index) <= 1:
                        potential_rank_reduction = 0
                        for first_seq_parent_index1, first_seq_parent_rank1 in \
                        enumerate(first_seq_rank_index[: -1]):
                            if first_seq_parent_rank1 < alg_max_rank_dict[
                                first_seq_algs[first_seq_parent_index1]
                            ] - 1:
                                for first_seq_parent_rank2 in first_seq_rank_index[
                                    first_seq_parent_index1 + 1: 
                                ]:
                                    potential_rank_reduction += first_seq_parent_rank2
                            if potential_rank_reduction > 1:
                                break
                        else:
                            top_rank_cs_found = True

                if top_rank_cs_found:
                    yield first_seq_cs_start_position, \
                        new_first_seq_source_starts, \
                        second_seq_cs_start_position, \
                        cs_len, \
                        first_seq_rank_index, \
                        second_seq_rank_index
                    break

        else:
            yield None, None, None, None, first_seq_rank_index, second_seq_rank_index

    #All comparisons
    for first_seq_rank_index, first_seq_info_for_rank_index_dict in first_seq_info_dict.items():
        first_encoded_seq = first_seq_info_for_rank_index_dict['encoded_seq']
        first_seq_parent_starts = first_seq_info_for_rank_index_dict['parent_substring_starts']
        
        for second_seq_rank_index, second_encoded_seq in second_encoded_seq_dict.items():
            yield do_seq_comparison(
                first_seq_rank_index, 
                first_encoded_seq, 
                first_seq_parent_starts, 
                second_seq_rank_index, 
                second_encoded_seq
            )

def make_seq_prediction_dict(
    consensus_spec,
    alg_consensus_source_df_for_spec_dict=None,
    alg_consensus_source_df_for_spec=None, 
    alg=None, 
    cs_info_dict=None,
    cs_type_list=None,
    alg_combo=None
):

    def calc_sub_score(seq, aa_scores, sub_list, alg):

        if alg == 'novor':
            max_score = 100
        elif alg == 'deepnovo':
            max_score = 1

        sub_score = 0
        for pep in sub_list:
            if pep in seq:
                for match_group in finditer(pep, seq):
                    sub_score += max_score - np.average(
                        aa_scores[match_group.start(): match_group.end()]
                    )

        #REMOVE
        if np.isnan(sub_score):
            print(alg_combo, flush=True)
            print(str(consensus_spec), flush=True)
            print(consensus_seq, flush=True)
            print(last_alg_consensus_source_df.at[last_seq_rank_index, 'seq'], flush=True)
            print(aa_scores, flush=True)
            for match_group in finditer(pep, seq):
                print(match_group, flush=True)
                print(str(max_score - np.average(
                    aa_scores[match_group.start(): match_group.end()]
                )), flush=True)

        return sub_score

    def count_low_scoring_peptides(aa_scores, pep_len):

        #Count the number of isolated, relatively low-scoring aa subseqs.
        #These often are incorrect due to inversion or isobaric substitution errors.

        score_stdev = np.std(aa_scores)
        low_scoring_pep_count = 0
        last_pep_start = len(aa_scores) - pep_len
        for i in range(last_pep_start):
            is_low_scoring = False
            pep_score = np.average(aa_scores[i: i + pep_len])
            if i == 0:
                bounding_score = aa_scores[pep_len]
                bounding_stdev = 0
            #When considering an interior subseq,
            #ensure that the bounding aa's have similarly high scores, 
            #e.g., bounding Novor aa scores of 0 and 100 are almost certainly unacceptable,
            #but scores of 80 and 90 are likely acceptable.
            elif i < last_pep_start:
                first_bounding_score = aa_scores[i - 1]
                second_bounding_score = aa_scores[i + pep_len]
                bounding_score = np.average([first_bounding_score, second_bounding_score])
                bounding_stdev = np.std([first_bounding_score, second_bounding_score])
            else:
                bounding_score = aa_scores[i - 1]
                bounding_stdev = 0
            if bounding_stdev <= score_stdev:
                if pep_score < bounding_score - score_stdev:
                    low_scoring_pep_count += 1

        return low_scoring_pep_count

    #If no common substring searches have yet been performed.
    if cs_info_dict is None:
        prediction_dict = OrderedDict().fromkeys(config.single_alg_prediction_dict_cols['general'])
        for k in prediction_dict:
            if k == 'spec_id':
                prediction_dict['spec_id'] = consensus_spec
            elif k == 'measured mass':
                prediction_dict['measured mass'] = alg_consensus_source_df_for_spec.at[
                    0, 'measured mass'
                ]
            elif k == 'is top rank single alg':
                prediction_dict['is top rank single alg'] = 1
            elif k == 'seq':
                prediction_dict['seq'] = seq = alg_consensus_source_df_for_spec.at[0, 'seq']
            elif k == 'len':
                prediction_dict['len'] = alg_consensus_source_df_for_spec.at[0, 'encoded seq'].size
        
        alg_prediction_dict = OrderedDict().fromkeys(config.single_alg_prediction_dict_cols[alg])
        if alg == 'novor':
            aa_scores = np.array(alg_consensus_source_df_for_spec.at[0, 'aa score'])
            for k in alg_prediction_dict:
                if k == 'retention time':
                    alg_prediction_dict['retention time'] = alg_consensus_source_df_for_spec.at[
                        0, 'retention time'
                    ]
                elif k == 'is novor seq':
                    alg_prediction_dict['is novor seq'] = 1
                elif k == 'avg novor aa score':
                    alg_prediction_dict['avg novor aa score'] = \
                        alg_consensus_source_df_for_spec.at[0, 'avg aa score']
                elif k == 'novor low-scoring dipeptide count':
                    alg_prediction_dict['novor low-scoring dipeptide count'] = \
                        count_low_scoring_peptides(aa_scores, 2)
                elif k == 'novor low-scoring tripeptide count':
                    alg_prediction_dict['novor low-scoring tripeptide count'] = \
                        count_low_scoring_peptides(aa_scores, 3)
                elif k == 'novor mono-di isobaric sub score':
                    alg_prediction_dict['novor mono-di isobaric sub score'] = \
                        calc_sub_score(seq, aa_scores, config.mono_di_isobaric_subs, alg)
                elif k == 'novor di isobaric sub score':
                    alg_prediction_dict['novor di isobaric sub score'] = \
                        calc_sub_score(seq, aa_scores, config.di_isobaric_subs, alg)
                elif k == 'novor mono-di near-isobaric sub score':
                    alg_prediction_dict['novor mono-di near-isobaric sub score'] = \
                        calc_sub_score(seq, aa_scores, config.mono_di_near_isobaric_subs, alg)
                elif k == 'novor di near-isobaric sub score':
                    alg_prediction_dict['novor di near-isobaric sub score'] = \
                        calc_sub_score(seq, aa_scores, config.di_near_isobaric_subs, alg)
        elif alg == 'pn':
            for k in alg_prediction_dict:
                if k == 'is pn seq':
                    alg_prediction_dict['is pn seq'] = 1
                elif k == 'rank score':
                    alg_prediction_dict['rank score'] = alg_consensus_source_df_for_spec.at[
                        0, 'rank score'
                    ]
                elif k == 'pn score':
                    alg_prediction_dict['pn score'] = alg_consensus_source_df_for_spec.at[
                        0, 'pn score'
                    ]
                elif k == 'sqs':
                    alg_prediction_dict['sqs'] = alg_consensus_source_df_for_spec.at[0, 'sqs']
        elif alg == 'deepnovo':
            aa_scores = np.array(alg_consensus_source_df_for_spec.at[0, 'aa score'])
            for k in alg_prediction_dict:
                if k == 'is deepnovo seq':
                    alg_prediction_dict['is deepnovo seq'] = 1
                elif k == 'avg deepnovo aa score':
                    alg_prediction_dict['avg deepnovo aa score'] = \
                        alg_consensus_source_df_for_spec.at[0, 'avg aa score']
                elif k == 'deepnovo low-scoring dipeptide count':
                    alg_prediction_dict['deepnovo low-scoring dipeptide count'] = \
                        count_low_scoring_peptides(aa_scores, 2)
                elif k == 'deepnovo low-scoring tripeptide count':
                    alg_prediction_dict['deepnovo low-scoring tripeptide count'] = \
                        count_low_scoring_peptides(aa_scores, 3)
                elif k == 'deepnovo mono-di isobaric sub score':
                    alg_prediction_dict['deepnovo mono-di isobaric sub score'] = \
                        calc_sub_score(seq, aa_scores, config.mono_di_isobaric_subs, alg)
                elif k == 'deepnovo di isobaric sub score':
                    alg_prediction_dict['deepnovo di isobaric sub score'] = \
                        calc_sub_score(seq, aa_scores, config.di_isobaric_subs, alg)
                elif k == 'deepnovo mono-di near-isobaric sub score':
                    alg_prediction_dict['deepnovo mono-di near-isobaric sub score'] = \
                        calc_sub_score(seq, aa_scores, config.mono_di_near_isobaric_subs, alg)
                elif k == 'deepnovo di near-isobaric sub score':
                    alg_prediction_dict['deepnovo di near-isobaric sub score'] = \
                        calc_sub_score(seq, aa_scores, config.di_near_isobaric_subs, alg)

        prediction_dict.update(alg_prediction_dict)

        return prediction_dict

    else:
        last_alg_consensus_source_df = alg_consensus_source_df_for_spec_dict[alg_combo[-1]]
        prediction_dict = OrderedDict().fromkeys(config.consensus_prediction_dict_cols['general'])
        last_seq_consensus_start_index = cs_info_dict['comparison_seq_starts'][1]
        last_seq_consensus_end_index = last_seq_consensus_start_index + cs_info_dict[
            'consensus_len'
        ]
        last_seq_rank_index = cs_info_dict['alg_ranks'][-1]
        for k in prediction_dict:
            if k == 'spec_id':
                prediction_dict['spec_id'] = consensus_spec
            elif k == 'measured mass':
                prediction_dict['measured mass'] = last_alg_consensus_source_df.at[
                    last_seq_rank_index, 'measured mass'
                ]
            elif k == 'seq':
                cs_info_dict['encoded_consensus_seq'] = \
                    last_alg_consensus_source_df.at[last_seq_rank_index, 'encoded seq'][
                        last_seq_consensus_start_index: last_seq_consensus_end_index
                    ]
                prediction_dict['seq'] = cs_info_dict['consensus_seq'] = consensus_seq = \
                    last_alg_consensus_source_df.at[last_seq_rank_index, 'seq'][
                        last_seq_consensus_start_index: last_seq_consensus_end_index
                    ]
            elif k == 'len':
                prediction_dict['len'] = cs_info_dict['consensus_len']
            elif k == 'avg rank':
                prediction_dict['avg rank'] = cs_info_dict['rank_sum'] / len(alg_combo)
            elif k == 'is longest consensus':
                if 'longest' in cs_type_list:
                    prediction_dict['is longest consensus'] = 1
                else:
                    prediction_dict['is longest consensus'] = 0
            elif k == 'is top rank consensus':
                if 'top rank' in cs_type_list:
                    prediction_dict['is top rank consensus'] = 1
                else:
                    prediction_dict['is top rank consensus'] = 0

        last_alg_combo_index = len(alg_combo) - 1
        for i, alg in enumerate(alg_combo):
            alg_rank = cs_info_dict['alg_ranks'][i]
            alg_prediction_dict = OrderedDict().fromkeys(
                config.consensus_prediction_dict_cols[alg]
            )
            if alg == 'novor':
                consensus_source_df = alg_consensus_source_df_for_spec_dict['novor']
                if i == last_alg_combo_index:
                    consensus_aa_scores = consensus_source_df.at[alg_rank, 'aa score'][
                        last_seq_consensus_start_index: last_seq_consensus_end_index
                    ]
                else:
                    source_seq_cs_start = cs_info_dict['source_seq_starts'][i]
                    consensus_aa_scores = consensus_source_df.at[alg_rank, 'aa score'][
                        source_seq_cs_start: source_seq_cs_start + cs_info_dict['consensus_len']
                    ]
                for k in alg_prediction_dict:
                    if k == 'retention time':
                        alg_prediction_dict['retention time'] = consensus_source_df.at[
                            alg_rank, 'retention time'
                        ]
                    elif k == 'is novor seq':
                        alg_prediction_dict['is novor seq'] = 1
                    elif k == 'fraction novor parent len':
                        alg_prediction_dict['fraction novor parent len'] = \
                            cs_info_dict['consensus_len'] / \
                            consensus_source_df.at[alg_rank, 'encoded seq'].size
                    elif k == 'avg novor aa score':
                        alg_prediction_dict['avg novor aa score'] = np.average(consensus_aa_scores)
                    elif k == 'novor low-scoring dipeptide count':
                        alg_prediction_dict['novor low-scoring dipeptide count'] = \
                            count_low_scoring_peptides(consensus_aa_scores, 2)
                    elif k == 'novor low-scoring tripeptide count':
                        alg_prediction_dict['novor low-scoring tripeptide count'] = \
                            count_low_scoring_peptides(consensus_aa_scores, 3)
                    elif k == 'novor mono-di isobaric sub score':
                        alg_prediction_dict['novor mono-di isobaric sub score'] = calc_sub_score(
                            consensus_seq, consensus_aa_scores, config.mono_di_isobaric_subs, alg
                        )
                    elif k == 'novor di isobaric sub score':
                        alg_prediction_dict['novor di isobaric sub score'] = calc_sub_score(
                            consensus_seq, consensus_aa_scores, config.di_isobaric_subs, alg
                        )
                    elif k == 'novor mono-di near-isobaric sub score':
                        alg_prediction_dict['novor mono-di near-isobaric sub score'] = \
                            calc_sub_score(
                                consensus_seq, 
                                consensus_aa_scores, 
                                config.mono_di_near_isobaric_subs, 
                                alg
                            )
                    elif k == 'novor di near-isobaric sub score':
                        alg_prediction_dict['novor di near-isobaric sub score'] = \
                            calc_sub_score(
                                consensus_seq, 
                                consensus_aa_scores, 
                                config.di_near_isobaric_subs, 
                                alg
                            )
            elif alg == 'pn':
                consensus_source_df = alg_consensus_source_df_for_spec_dict['pn']
                for k in alg_prediction_dict:
                    if k == 'is pn seq':
                        alg_prediction_dict['is pn seq'] = 1
                    elif k == 'fraction pn parent len':
                        alg_prediction_dict['fraction pn parent len'] = \
                            cs_info_dict['consensus_len'] / \
                            consensus_source_df.at[alg_rank, 'encoded seq'].size
                    elif k == 'rank score':
                        alg_prediction_dict['rank score'] = consensus_source_df.at[
                            alg_rank, 'rank score'
                        ]
                    elif k == 'pn score':
                        alg_prediction_dict['pn score'] = consensus_source_df.at[
                            alg_rank, 'pn score'
                        ]
                    elif k == 'pn rank':
                        alg_prediction_dict['pn rank'] = alg_rank
                    elif k == 'sqs':
                        alg_prediction_dict['sqs'] = consensus_source_df.at[alg_rank, 'sqs']
            elif alg == 'deepnovo':
                consensus_source_df = alg_consensus_source_df_for_spec_dict['deepnovo']
                if i == last_alg_combo_index:
                    consensus_aa_scores = consensus_source_df.at[alg_rank, 'aa score'][
                        last_seq_consensus_start_index: last_seq_consensus_end_index
                    ]
                else:
                    source_seq_cs_start = cs_info_dict['source_seq_starts'][i]
                    consensus_aa_scores = consensus_source_df.at[alg_rank, 'aa score'][
                        source_seq_cs_start: source_seq_cs_start + cs_info_dict['consensus_len']
                    ]
                for k in alg_prediction_dict:
                    if k == 'is deepnovo seq':
                        alg_prediction_dict['is deepnovo seq'] = 1
                    elif k == 'fraction deepnovo parent len':
                        alg_prediction_dict['fraction deepnovo parent len'] = \
                            cs_info_dict['consensus_len'] / \
                            consensus_source_df.at[alg_rank, 'encoded seq'].size
                    elif k == 'deepnovo rank':
                        alg_prediction_dict['deepnovo rank'] = alg_rank
                    elif k == 'avg deepnovo aa score':
                        alg_prediction_dict['avg deepnovo aa score'] = np.average(
                            consensus_aa_scores
                        )
                    elif k == 'deepnovo low-scoring dipeptide count':
                        alg_prediction_dict['deepnovo low-scoring dipeptide count'] = \
                            count_low_scoring_peptides(consensus_aa_scores, 2)
                    elif k == 'deepnovo low-scoring tripeptide count':
                        alg_prediction_dict['deepnovo low-scoring tripeptide count'] = \
                            count_low_scoring_peptides(consensus_aa_scores, 3)
                    elif k == 'deepnovo mono-di isobaric sub score':
                        alg_prediction_dict['deepnovo mono-di isobaric sub score'] = \
                            calc_sub_score(
                                consensus_seq, 
                                consensus_aa_scores, 
                                config.mono_di_isobaric_subs, 
                                alg
                            )
                    elif k == 'deepnovo di isobaric sub score':
                        alg_prediction_dict['deepnovo di isobaric sub score'] = \
                            calc_sub_score(
                                consensus_seq, consensus_aa_scores, config.di_isobaric_subs, alg
                            )
                    elif k == 'deepnovo mono-di near-isobaric sub score':
                        alg_prediction_dict['deepnovo mono-di near-isobaric sub score'] = \
                            calc_sub_score(
                                consensus_seq, 
                                consensus_aa_scores, 
                                config.mono_di_near_isobaric_subs, 
                                alg
                            )
                    elif k == 'deepnovo di near-isobaric sub score':
                        alg_prediction_dict['deepnovo di near-isobaric sub score'] = \
                            calc_sub_score(
                                consensus_seq, 
                                consensus_aa_scores, 
                                config.di_near_isobaric_subs, 
                                alg
                            )
            prediction_dict.update(alg_prediction_dict)

        return prediction_dict

def encode_seqs(df, consensus_min_len):

    df['encoded seq'] = df['seq'].where(df['seq'].str.len() >= consensus_min_len)
    df['encoded seq'].fillna('', inplace = True)
    df['encoded seq'] = df['encoded seq'].apply(list)
    map_ord = partial(map, ord)
    df['encoded seq'] = df['encoded seq'].apply(map_ord).apply(list)
    df['encoded seq'] = df['encoded seq'].apply(np.array).apply(
        lambda x: x - config.unicode_decimal_A
    )
    return

def make_combo_level_alg_dict(alg_df_dict):
    combo_level_alg_dict = OrderedDict()
    for combo_level in range(2, len(alg_df_dict) + 1):
        combo_level_alg_dict[combo_level] = []
        combo_level_alg_dict[combo_level] += [
            alg_combo for alg_combo in config.globals['alg_combos']
            if len(alg_combo) == combo_level
        ]
    return combo_level_alg_dict

def add_measured_mass_col(alg_df_dict):

    #Mass is reported in Novor and PepNovo output.
    for alg in ['novor', 'pn']:
        alg_df = alg_df_dict[alg]
        alg_df['measured mass'] = alg_df['m/z'] * alg_df['charge']
    if 'deepnovo' in config.globals['algs']:
        #Mass is not reported in DeepNovo output,
        #and is therefore transferred into the DeepNovo table from Novor.
        novor_df = alg_df_dict['novor']
        deepnovo_df = alg_df_dict['deepnovo']
        indices = [
            (spec_id, 0) for spec_id in deepnovo_df.index.get_level_values('spec_id').tolist()
        ]
        deepnovo_df['measured mass'] = novor_df.ix[indices]['measured mass'].tolist()
        
    return alg_df_dict

def add_retention_time_cols(alg_df_dict):
    #Retention time is reported in Novor, PepNovo but not DeepNovo output.

    novor_df = alg_df_dict['novor']
    pn_df = alg_df_dict['pn']
    indices = [(spec_id, 0) for spec_id in pn_df.index.get_level_values('spec_id').tolist()]
    pn_df['retention time'] = novor_df.ix[indices]['retention time'].tolist()

    if 'deepnovo' in config.globals['algs']:
        deepnovo_df = alg_df_dict['deepnovo']
        indices = [
            (spec_id, 0) for spec_id in deepnovo_df.index.get_level_values('spec_id').tolist()
        ]
        deepnovo_df['retention time'] = novor_df.ix[indices]['retention time'].tolist()

    return alg_df_dict

def make_alg_consensus_source_df_dict(highest_level_alg_combo, alg_df_dict):
    alg_consensus_source_df_dict = OrderedDict().fromkeys(highest_level_alg_combo)
    for alg in alg_consensus_source_df_dict:
        alg_consensus_source_df_dict[alg] = alg_df_dict[alg][
            config.prediction_dict_source_cols[alg]
        ]
    return alg_consensus_source_df_dict

def setup_spec_info_dicts(combo_level_alg_dict):

    spec_consensus_info_dict = OrderedDict().fromkeys(combo_level_alg_dict)
    spec_generator_fns_dict = OrderedDict().fromkeys(combo_level_alg_dict)
    did_comparison_dict = OrderedDict().fromkeys(combo_level_alg_dict)
    spec_common_substrings_info_dict = OrderedDict().fromkeys(combo_level_alg_dict)

    consensus_info_keys = [
        'alg_ranks', 'rank_sum', 'comparison_seq_starts', 'source_seq_starts', 'consensus_len'
    ]

    for i, combo_level in enumerate(combo_level_alg_dict):
        combo_level_alg_dict_for_combo_level = combo_level_alg_dict[combo_level]
        spec_consensus_info_dict[combo_level] = OrderedDict().fromkeys(
            combo_level_alg_dict_for_combo_level
        )
        for alg_combo in combo_level_alg_dict_for_combo_level:
            spec_consensus_info_dict[combo_level][alg_combo] = {}
            spec_consensus_info_dict[combo_level][alg_combo]['longest_cs'] = {}.fromkeys(
                consensus_info_keys
            )
            spec_consensus_info_dict[combo_level][alg_combo]['top_rank_cs'] = {}.fromkeys(
                consensus_info_keys
            )

        spec_generator_fns_dict[combo_level] = OrderedDict().fromkeys(
            combo_level_alg_dict_for_combo_level
        )
        did_comparison_dict[combo_level] = OrderedDict().fromkeys(
            combo_level_alg_dict_for_combo_level
        )
        spec_common_substrings_info_dict[combo_level] = OrderedDict().fromkeys(
            combo_level_alg_dict_for_combo_level
        )

    return spec_consensus_info_dict, \
        spec_generator_fns_dict, \
        did_comparison_dict, \
        spec_common_substrings_info_dict

def make_rank_comparison_dict(spec_consensus_info_dict):

    rank_comparison_dict = OrderedDict()

    for combo_level in spec_consensus_info_dict:
        rank_comparison_dict[combo_level] = OrderedDict()
        for alg_combo in spec_consensus_info_dict[combo_level]:
            last_ranks_list = []
            alg_ranks_ranges = []

            for alg in alg_combo:
                last_ranks_list.append(config.seqs_reported_per_alg_dict[alg])
                alg_ranks_ranges.append(range(last_ranks_list[-1]))
            rank_comparison_dict[combo_level][alg_combo] = list(product(*alg_ranks_ranges))

    return rank_comparison_dict
''' Find consensus sequences from de novo sequences '''

import numpy as np
import pandas as pd
import sys

from collections import OrderedDict
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

    if config.mode[0] in ['train', 'optimize']:
        consensus_min_len = config.train_consensus_len
    elif config.mode[0] in ['predict', 'test']:
        consensus_min_len = config.min_len[0]

    tol_prediction_df_list = []
    for tol in config.frag_mass_tols:
        utils.verbose_print('setting up', tol, 'Da consensus comparison')
        alg_df_dict = OrderedDict([(alg, input_df_dict[alg][tol]) for alg in config.alg_list])
        tol_prediction_df = make_prediction_df_for_tol(consensus_min_len, alg_df_dict, tol)
        tol_prediction_df_list.append(tol_prediction_df)

    prediction_df = pd.concat(tol_prediction_df_list)
    grouped_by_scan = prediction_df.groupby(['scan'])
    prediction_df['retention time'] = grouped_by_scan['retention time'].transform(max)
    prediction_df = prediction_df[~prediction_df['retention time'].isnull()]

    for tol in config.frag_mass_tols:
        prediction_df[tol].fillna(0, inplace = True)

    for is_alg_col_name in config.is_alg_col_names:
        prediction_df[is_alg_col_name].fillna(0, inplace = True)
        prediction_df[is_alg_col_name] = prediction_df[is_alg_col_name].astype(int)
    prediction_df.set_index(config.is_alg_col_names + ['scan'], inplace = True)
    prediction_df.sort_index(level = ['scan'] + config.is_alg_col_names, inplace = True)

    return prediction_df

def make_prediction_df_for_tol(consensus_min_len, alg_df_dict, tol):

    for alg, df in alg_df_dict.items():
        encode_seqs(df, consensus_min_len)

    combo_level_alg_dict = make_combo_level_alg_dict(alg_df_dict)
    highest_level_alg_combo = config.alg_combo_list[-1]
    ## example
    ## combo_level_alg_dict = odict(2: [('novor', 'pn'), ('novor', 'deepnovo'), ('pn', 'deepnovo')], 3: [('novor', 'pn', 'deepnovo')])
    ## highest_level_alg_combo = ('novor', 'pn', 'deepnovo')

    alg_df_dict = add_measured_mass_col(alg_df_dict)
    alg_consensus_source_df_dict = make_alg_consensus_source_df_dict(highest_level_alg_combo, alg_df_dict)
    consensus_scan_list = alg_consensus_source_df_dict[highest_level_alg_combo[0]].index.get_level_values('scan').tolist()
    one_percent_number_consensus_scans = len(consensus_scan_list) / 100 / config.cores[0]

    scan_consensus_info_dict, scan_generator_fns_dict, did_comparison_dict, scan_common_substrings_info_dict = setup_scan_info_dicts(combo_level_alg_dict)
    ##examples
    ##scan_consensus_info_dict = OrderedDict(
    ##    2: OrderedDict(
    ##        ('novor', 'pn'): 
    ##            {
    ##                'longest_cs': {'seq_starts': None, 'rank_sum': None, 'consensus_len': None, 'alg_ranks': None}, 
    ##                'top_rank_cs': {'seq_starts': None, 'rank_sum': None, 'consensus_len': None, 'alg_ranks': None}
    ##                },
    ##        ('novor', 'deepnovo'):
    ##            ...
    ##        ...,
    ##    3: OrderedDict(
    ##        ('novor', 'pn', 'deepnovo'):
    ##            ...
    ##    )
    ##scan_generator_fns_dict = OrderedDict(
    ##    2: OrderedDict(
    ##        ('novor', 'pn'): generator_fn,
    ##        ('novor', 'deepnovo'): generator_fn, 
    ##        ('pn', 'deepnovo'): generator_fn
    ##        ),
    ##    3: OrderedDict(
    ##        ('novor', 'pn', 'deepnovo'): generator_fn
    ##        )
    ##    )
    ##did_comparison_dict = OrderedDict(
    ##    2: OrderedDict(
    ##        ('novor', 'pn'): OrderedDict(ranks: 0 or 1 for tracking comparison),
    ##        ('novor', 'deepnovo'): OrderedDict(ranks: 0 or 1), 
    ##        ('pn', 'deepnovo'): OrderedDict(ranks: 0 or 1)
    ##        ),
    ##    3: OrderedDict(
    ##        ('novor', 'pn', 'deepnovo'): OrderedDict(ranks: 0 or 1)
    ##        )
    ##    )
    ##scan_common_substrings_info_dict = OrderedDict(
    ##    2: OrderedDict(
    ##        ('novor', 'pn'): OrderedDict(ranks: common substrings),
    ##        ('novor', 'deepnovo'): OrderedDict(ranks: common substrings),
    ##        ('pn', 'deepnovo'): OrderedDict(ranks: common substrings)
    ##        ),
    ##    3: OrderedDict(
    ##        ('novor', 'pn', 'deepnovo'): OrderedDict(ranks: common substrings)
    ##        )
    ##    )

    rank_comparison_dict = make_rank_comparison_dict(scan_consensus_info_dict)
    ##rank_comparison_dict = OrderedDict(
    ##    2: OrderedDict(
    ##        ('novor', 'pn'): [(0, 0), (0, 1), ..., (0, 18), (0, 19)],
    ##        ('novor', 'deepnovo'): [(0, 0), (0, 1), ..., (0, 18), (0, 19)],
    ##        ('pn', 'deepnovo'): [(0, 0), (0, 1), ..., (19, 18), (19, 19)]
    ##        ),
    ##    3: OrderedDict(
    ##        ('novor', 'pn', 'deepnovo'): [(0, 0, 0), (0, 0, 1), ..., (0, 19, 18), (0, 19, 19)]
    ##        )
    ##    )

    alg_max_rank_dict = OrderedDict()
    for alg in config.alg_list:
        alg_max_rank_dict[alg] = config.seqs_reported_per_alg_dict[alg]

    ## single processor method
    print_percent_progress_fn = partial(
        utils.print_percent_progress_singlethreaded,
        procedure_str = tol + ' Da progress: ',
        one_percent_total_count = one_percent_number_consensus_scans
        )
    child_initialize(
        alg_consensus_source_df_dict,
        scan_consensus_info_dict,
        scan_generator_fns_dict,
        did_comparison_dict,
        scan_common_substrings_info_dict,
        consensus_min_len,
        rank_comparison_dict,
        alg_max_rank_dict,
        print_percent_progress_fn
        )
    grand_scan_prediction_dict_list = []
    utils.verbose_print('finding', tol, 'Da consensus sequences')
    for consensus_scan in consensus_scan_list:
        grand_scan_prediction_dict_list.append(
            make_scan_prediction_dicts(consensus_scan)
            )

    ## multiprocessing method
    #print_percent_progress_fn = partial(utils.print_percent_progress_multithreaded,
    #                                    procedure_str = tol + ' Da progress: ',
    #                                    one_percent_total_count = one_percent_number_consensus_scans,
    #                                    cores = config.cores[0])
    #multiprocessing_pool = Pool(config.cores[0],
    #                            initializer = child_initialize,
    #                            initargs = (alg_consensus_source_df_dict,
    #                                        scan_consensus_info_dict,
    #                                        scan_generator_fns_dict,
    #                                        did_comparison_dict,
    #                                        scan_common_substrings_info_dict,
    #                                        consensus_min_len,
    #                                        rank_comparison_dict,
    #                                        alg_max_rank_dict,
    #                                        print_percent_progress_fn)
    #                            )
    #utils.verbose_print('finding', tol, 'Da consensus sequences')
    #grand_scan_prediction_dict_list = multiprocessing_pool.map(make_scan_prediction_dicts, consensus_scan_list)
    #multiprocessing_pool.close()
    #multiprocessing_pool.join()

    scan_prediction_dict_list = [seq_prediction_dict
                                 for scan_prediction_dict_list in grand_scan_prediction_dict_list
                                 for seq_prediction_dict in scan_prediction_dict_list]
    tol_prediction_df = pd.DataFrame().from_dict(scan_prediction_dict_list)
    tol_prediction_df[tol] = 1
    
    return tol_prediction_df

def child_initialize(_alg_consensus_source_df_dict,
                     _scan_consensus_info_dict,
                     _scan_generator_fns_dict,
                     _did_comparison_dict,
                     _scan_common_substrings_info_dict,
                     _consensus_min_len,
                     _rank_comparison_dict,
                     _alg_max_rank_dict,
                     _print_percent_progress_fn):

     global alg_consensus_source_df_dict
     global scan_consensus_info_dict
     global scan_generator_fns_dict
     global did_comparison_dict
     global scan_common_substrings_info_dict
     global consensus_min_len
     global rank_comparison_dict
     global alg_max_rank_dict
     global print_percent_progress_fn

     alg_consensus_source_df_dict = _alg_consensus_source_df_dict
     scan_consensus_info_dict = _scan_consensus_info_dict
     scan_generator_fns_dict = _scan_generator_fns_dict
     did_comparison_dict = _did_comparison_dict
     scan_common_substrings_info_dict = _scan_common_substrings_info_dict
     consensus_min_len = _consensus_min_len
     rank_comparison_dict = _rank_comparison_dict
     alg_max_rank_dict = _alg_max_rank_dict
     print_percent_progress_fn = _print_percent_progress_fn

def make_scan_prediction_dicts(consensus_scan):

    print_percent_progress_fn()

    scan_prediction_dict_list = []

    alg_consensus_source_df_for_scan_dict = OrderedDict()
    max_seq_len_dict = OrderedDict()
    # See below for explanation of is_short_seq_alg
    is_short_seq_alg = False
    for alg in alg_consensus_source_df_dict:
        alg_consensus_source_df_for_scan = alg_consensus_source_df_dict[alg].loc[consensus_scan]
        # deepnovo can have a variable number of seqs per scan due to the consolidation of Ile/Leu
        num_seqs = len(alg_consensus_source_df_for_scan)
        if alg == 'deepnovo':
            if num_seqs < alg_max_rank_dict[alg]:
                alg_max_rank_dict['deepnovo'] = len(alg_consensus_source_df_for_scan)

                # Changing the number of deepnovo seqs also changes the number of seq comparisons
                # Retabulate the rank comparisons that can be performed
                for combo_level, rank_comparison_dict_for_combo_level in rank_comparison_dict.items():
                    for alg_combo in rank_comparison_dict_for_combo_level:
                        if 'deepnovo' in alg_combo:
                            alg_ranks_ranges = []
                            last_ranks_list = []

                            for alg in alg_combo:
                                last_ranks_list.append(alg_max_rank_dict[alg])
                                alg_ranks_ranges.append(range(last_ranks_list[-1]))
                            rank_comparison_dict_for_combo_level[alg_combo] = list(product(*alg_ranks_ranges))

        # Expect a set number of seqs per scan from novor and pn
        # Occasionally fewer are reported for poor-quality spectra,
        # in which case, ignore predictions from ALL algs, returning an empty list
        else:
            if num_seqs < alg_max_rank_dict[alg]:
                return []

        # I am currently retaining seqs that do not meet the minimum length
        # in the consensus source table
        # This allows the rank index ranges to be contiguous,
        # unlike if the short seqs were removed from the tables
        max_seq_len_dict[alg] = alg_consensus_source_df_for_scan['encoded seq'].map(len).max()
        # Do not proceed to the consensus routine
        # if any one of the algs lacks seqs above the minimum length
        if max_seq_len_dict[alg] == 0:
            is_short_seq_alg = True
        else:
            if alg_consensus_source_df_for_scan.at[0, 'encoded seq'] != []:
                # The single-alg seqs are placed in the results
                scan_prediction_dict_list.append(
                    make_seq_prediction_dict(
                        consensus_scan,
                        alg_consensus_source_df_for_scan=alg_consensus_source_df_for_scan,
                        alg=alg
                        )
                    )
            alg_consensus_source_df_for_scan_dict[alg] = alg_consensus_source_df_for_scan

    if is_short_seq_alg:
        return scan_prediction_dict_list

    # Find consensus seqs between increasing numbers of algs (2, 3, etc.)
    for combo_level in scan_consensus_info_dict:
        rank_comparison_for_combo_level_dict = rank_comparison_dict[combo_level]
        scan_generator_fns_for_combo_level_dict = scan_generator_fns_dict[combo_level]
        scan_consensus_info_for_combo_level_dict = scan_consensus_info_dict[combo_level]

        for alg_combo in scan_consensus_info_for_combo_level_dict:
            rank_comparisons = rank_comparison_for_combo_level_dict[alg_combo]
            scan_consensus_info_for_alg_combo_dict = scan_consensus_info_for_combo_level_dict[alg_combo]

            scan_common_substrings_info_for_alg_combo_dict = scan_common_substrings_info_dict[combo_level][alg_combo] =\
                OrderedDict().fromkeys(rank_comparisons)
            did_comparison_for_alg_combo_dict = did_comparison_dict[combo_level][alg_combo] =\
                OrderedDict([(rank_index, False) for rank_index in rank_comparisons])

            longest_cs_dict = scan_consensus_info_for_alg_combo_dict['longest_cs']
            top_rank_cs_dict = scan_consensus_info_for_alg_combo_dict['top_rank_cs']

            # Consensus seqs of length 2 are considered separately from seqs involving more algs
            # This is because 
            # 1. Not all necessary parent consensus seqs may have been generated for the latter
            # 2. All seq comparisons need to be performed for the latter
            # if the >2-alg comparison LCS and TRCS fast-track procedures were unsuccessful
            if combo_level == 2:
                first_seq_alg = alg_combo[0]
                second_seq_alg = alg_combo[1]

                first_encoded_seq_dict = OrderedDict([
                    ((rank,), encoded_seq) for rank, encoded_seq
                    in enumerate(alg_consensus_source_df_for_scan_dict[first_seq_alg]['encoded seq'])
                    ])
                second_encoded_seq_dict = OrderedDict([
                    ((rank,), encoded_seq) for rank, encoded_seq
                    in enumerate(alg_consensus_source_df_for_scan_dict[second_seq_alg]['encoded seq'])
                    ])

                max_possible_cs_len = min(max_seq_len_dict[first_seq_alg], max_seq_len_dict[second_seq_alg])

                seq_comparison_generator = do_seq_comparisons(first_encoded_seq_dict, second_encoded_seq_dict, consensus_min_len)
                scan_generator_fns_for_combo_level_dict[alg_combo] = seq_comparison_generator
                longest_cs_len = 0
                min_cs_rank_sum = 1000

                for first_seq_cs_start_position, second_seq_cs_start_position, cs_len, first_seq_rank_index, second_seq_rank_index in seq_comparison_generator:
                    did_comparison_for_alg_combo_dict[first_seq_rank_index + second_seq_rank_index] = True

                    if cs_len is not None:
                        longest_cs_len, min_cs_rank_sum = parse_generator_output(
                            longest_cs_len,
                            min_cs_rank_sum,
                            first_encoded_seq_dict,
                            scan_common_substrings_info_for_alg_combo_dict,
                            first_seq_cs_start_position,
                            second_seq_cs_start_position,
                            cs_len,
                            first_seq_rank_index,
                            second_seq_rank_index,
                            longest_cs_dict,
                            top_rank_cs_dict
                            )

                        # Stop searching for CS's when the longest possible CS is found (equal to len of shortest parent seq)
                        # and when it can be shown that the LCS must also be the T-R CS
                        if longest_cs_len == max_possible_cs_len and longest_cs_dict['alg_ranks'] == top_rank_cs_dict['alg_ranks']:
                            top_rank_cs_found = False

                            # If all of the parent seqs are rank 0
                            # or only 1 of the parent seqs is rank 1,
                            # then it will be impossible to find a CS with a lower sum rank,
                            # so stop searching for additional CS's
                            if sum(first_seq_rank_index + second_seq_rank_index) <= 1:
                                top_rank_cs_found = True

                            # In addition, the following rule can demonstrate that the seq is a T-R CS:
                            # The potential rank reduction of the second seq <= 1
                            # AND 
                            # there is no total rank increment of the first seq's parent seqs > 1
                            else:
                                if sum(second_seq_rank_index) <= 1:
                                    potential_rank_reduction = 0
                                    for first_seq_parent_index, first_seq_parent_rank in enumerate(first_seq_rank_index):
                                        if alg_max_rank_dict[first_seq_algs[first_seq_parent_index]] > 0:
                                            potential_rank_reduction += 1
                                    if potential_rank_reduction > 1:
                                        top_rank_cs_found = True

                            if top_rank_cs_found:
                                break

            # Considering consensus seqs of >2 algs
            else:
                first_seq_algs = alg_combo[: -1]
                second_seq_alg = alg_combo[-1]

                # Get the first seq LCS
                first_seq_lcs_dict = scan_consensus_info_dict[combo_level - 1][first_seq_algs]['longest_cs']
                if 'encoded_consensus_seq' in first_seq_lcs_dict:
                    parent_seq_alg_ranks = first_seq_lcs_dict['alg_ranks']
                    first_seq_lcs_rank_index = parent_seq_alg_ranks[0] + parent_seq_alg_ranks[1]
                    first_seq_encoded_lcs = first_seq_lcs_dict['encoded_consensus_seq']
                else:
                    first_seq_lcs_rank_index = None
                    first_seq_encoded_lcs = None

                # If any constituent alg combo doesn't have an LCS,
                # then there are no consensus sequences for this combo level
                if first_seq_encoded_lcs is None:
                    break

                # Get the T-R CS for the first seq algs
                first_seq_trcs_dict = scan_consensus_info_dict[combo_level - 1][first_seq_algs]['top_rank_cs']
                if 'encoded_consensus_seq' in first_seq_trcs_dict:
                    parent_seq_alg_ranks = first_seq_trcs_dict['alg_ranks']
                    first_seq_trcs_rank_index = parent_seq_alg_ranks[0] + parent_seq_alg_ranks[1]
                    first_seq_encoded_trcs = first_seq_trcs_dict['encoded_consensus_seq']
                else:
                    first_seq_trcs_rank_index = None
                    first_seq_encoded_trcs = None

                # Get all of the CS's for the first seq algs
                # Example:
                # rank_index = ((1, 19), (11, ))
                # encoded_cs = np.array([4, 7, 10, 14, 19, 3, 14, 5, 6, 2])
                # combo_level = 4 - 1
                # first_seq_algs = ('a', 'b', 'c')
                # first_encoded_seq_dict[(1, 19, 11)] = encoded_cs
                first_encoded_seq_dict = OrderedDict()
                for rank_index, encoded_cs in scan_common_substrings_info_dict[combo_level - 1][first_seq_algs].items():
                    first_encoded_seq_dict[rank_index[0] + rank_index[1]] = encoded_cs

                # Get seqs for the second seq alg
                alg_rank_encoded_seq_list = alg_consensus_source_df_for_scan_dict[second_seq_alg]['encoded seq']
                second_encoded_seq_dict = OrderedDict(
                    [((rank,), encoded_seq) for rank, encoded_seq in enumerate(alg_rank_encoded_seq_list)])

                max_possible_cs_len = len(first_seq_encoded_lcs)

                # Initialize the seq comparison generator
                seq_comparison_generator = do_seq_comparisons(
                    first_encoded_seq_dict, 
                    second_encoded_seq_dict, 
                    consensus_min_len, 
                    first_seq_lcs_rank_index, 
                    first_seq_encoded_lcs, 
                    first_seq_trcs_rank_index, 
                    first_seq_encoded_trcs, 
                    alg_max_rank_dict
                    )
                scan_generator_fns_for_combo_level_dict[alg_combo] = seq_comparison_generator
                
                # Fast-track LCS and TRCS comparisons.
                # If these seqs do not yield a new LCS and TRCS, 
                # the comparisons are later repeated so the results can be recorded.
                # This is an overall efficiency that compensates
                # for that later inefficiency of repeating these two comparisons

                longest_cs_dict = scan_consensus_info_for_alg_combo_dict['longest_cs']
                longest_cs_dict = longest_cs_dict.fromkeys(longest_cs_dict, None)
                top_rank_cs_dict = scan_consensus_info_for_alg_combo_dict['top_rank_cs']
                top_rank_cs_dict = top_rank_cs_dict.fromkeys(top_rank_cs_dict, None)

                # LCS
                first_seq_cs_start_position, second_seq_cs_start_position, cs_len, first_seq_rank_index, second_seq_rank_index = next(seq_comparison_generator)
                if first_seq_cs_start_position:
                    longest_cs_dict['alg_ranks'] = (first_seq_rank_index, second_seq_rank_index)
                    longest_cs_dict['rank_sum'] = sum(first_seq_rank_index) + sum(second_seq_rank_index) 
                    longest_cs_dict['seq_starts'] = (first_seq_cs_start_position, second_seq_cs_start_position)
                    longest_cs_dict['consensus_len'] = cs_len

                # TRCS
                # A separate comparison is performed for the TRCS even if the LCS is also the TRCS
                # due to the excessive complexity of determining that
                first_seq_cs_start_position, second_seq_cs_start_position, cs_len, first_seq_rank_index, second_seq_rank_index = next(seq_comparison_generator)
                if first_seq_cs_start_position:
                    top_rank_cs_dict['alg_ranks'] = (first_seq_rank_index, second_seq_rank_index)
                    top_rank_cs_dict['rank_sum'] = sum(first_seq_rank_index) + sum(second_seq_rank_index) 
                    top_rank_cs_dict['seq_starts'] = (first_seq_cs_start_position, second_seq_cs_start_position)
                    top_rank_cs_dict['consensus_len'] = cs_len
                    
                # Get the generators and comparison records for the first seqs and their parent first seqs
                first_seq_scan_generator_fns_dict = OrderedDict()
                first_seq_did_comparison_dict = OrderedDict()
                for parent_seq_combo_level in range(combo_level - 1, 1, -1):
                    first_seq_scan_generator_fns_dict[parent_seq_combo_level] = scan_generator_fns_dict[parent_seq_combo_level][first_seq_algs[: parent_seq_combo_level]]
                    first_seq_did_comparison_dict[parent_seq_combo_level] = did_comparison_dict[parent_seq_combo_level][first_seq_algs[: parent_seq_combo_level]]

                first_seq_rank_indices = list(first_encoded_seq_dict.keys())
                last_comparison_index = len(first_seq_rank_indices) - 1

                longest_cs_len = 0
                min_cs_rank_sum = 1000

                # Even if the LCS and TRCS were found earlier,
                # perform the first comparison for the purposes of
                # the operation of the consensus first seq comparison check
                for comparison_index, generator_output in enumerate(seq_comparison_generator):

                    first_seq_cs_start_position, second_seq_cs_start_position, cs_len, first_seq_rank_index, second_seq_rank_index = generator_output
                    did_comparison_for_alg_combo_dict[first_seq_rank_index + second_seq_rank_index] = True

                    if cs_len is not None:
                        parse_generator_output(
                            longest_cs_len,
                            min_cs_rank_sum,
                            first_encoded_seq_dict,
                            scan_common_substrings_info_for_alg_combo_dict,
                            first_seq_cs_start_position,
                            second_seq_cs_start_position,
                            cs_len,
                            first_seq_rank_index,
                            second_seq_rank_index,
                            max_possible_cs_len,
                            longest_cs_dict,
                            top_rank_cs_dict
                            )
                                            
                    # Check whether the next consensus first seq was ever considered
                    # The next first seq cannot exist if its comparison was not performed
                    if comparison_index < last_comparison_index:
                        first_seq_next_rank_index = first_seq_rank_indices[comparison_index + 1]
                        if not first_seq_did_comparison_dict[first_seq_next_rank_index]:
                            # Perform the first seq comparison and all further necessary parent seq comparisons
                            do_parent_comparisons(
                                combo_level, 
                                alg_combo, 
                                first_seq_next_rank_index, 
                                first_seq_scan_generator_fns_dict, 
                                first_seq_did_comparison_dict
                                )

            # Back to consideration of both 2- and >2-alg consensus seqs:
            # If an LCS meeting the min length threshold was found
            if longest_cs_dict['alg_ranks'] is not None:

                # If considering a CS that is both LCS and T-R CS
                if longest_cs_dict['alg_ranks'] == top_rank_cs_dict['alg_ranks']:
                    cs_prediction_dict = make_seq_prediction_dict(
                        consensus_scan,
                        alg_consensus_source_df_for_scan_dict=alg_consensus_source_df_for_scan_dict,
                        cs_info_dict=longest_cs_dict,
                        cs_type_list=['longest', 'top rank'],
                        alg_combo=alg_combo,
                        )
                    top_rank_cs_dict['encoded_consensus_seq'] = longest_cs_dict['encoded_consensus_seq']
                    scan_prediction_dict_list.append(cs_prediction_dict)
                # Else the LCS and T-R CS are different seqs
                else:
                    longest_cs_prediction_dict = make_seq_prediction_dict(
                        consensus_scan,
                        alg_consensus_source_df_for_scan_dict=alg_consensus_source_df_for_scan_dict,
                        cs_info_dict=longest_cs_dict,
                        cs_type_list=['longest'],
                        alg_combo=alg_combo,
                        )
                    top_rank_cs_prediction_dict = make_seq_prediction_dict(
                        consensus_scan,
                        alg_consensus_source_df_for_scan_dict=alg_consensus_source_df_for_scan_dict,
                        cs_info_dict=top_rank_cs_dict,
                        cs_type_list=['top rank'],
                        alg_combo=alg_combo,
                        )
                    scan_prediction_dict_list.append(longest_cs_prediction_dict)
                    scan_prediction_dict_list.append(top_rank_cs_prediction_dict)

    return scan_prediction_dict_list

def parse_generator_output(
    longest_cs_len,
    min_cs_rank_sum,
    first_encoded_seq_dict,
    scan_common_substrings_info_for_alg_combo_dict,
    first_seq_cs_start_position,
    second_seq_cs_start_position,
    cs_len,
    first_seq_rank_index,
    second_seq_rank_index,
    longest_cs_dict,
    top_rank_cs_dict
    ):

    # Record the common substrings found for each comparison
    # Example:
    # scan_common_substrings_info_for_alg_combo_dict[3][((1, 19), (11, ))] =\
    #   first_encoded_seq_dict[(1, 19)][3: 3 + 10] = np.array([4, 7, 10, 14, 19, 3, 14, 5, 6, 2])
    scan_common_substrings_info_for_alg_combo_dict[(first_seq_rank_index, second_seq_rank_index)] =\
        first_encoded_seq_dict[first_seq_rank_index][first_seq_cs_start_position: first_seq_cs_start_position + cs_len]

    # Each rank index is a tuple
    first_rank_sum = sum(first_seq_rank_index)
    second_rank_sum = sum(second_seq_rank_index)
    cs_rank_sum = first_rank_sum + second_rank_sum

    # LCS found
    if cs_len > longest_cs_len:
        longest_cs_len = cs_len
        longest_cs_dict['alg_ranks'] = first_seq_rank_index + second_seq_rank_index
        longest_cs_dict['rank_sum'] = cs_rank_sum
        longest_cs_dict['seq_starts'] = (first_seq_cs_start_position, second_seq_cs_start_position)
        longest_cs_dict['consensus_len'] = cs_len

    # Top-ranking (T-R) seq found
    if cs_rank_sum < min_cs_rank_sum:
        min_cs_rank_sum = cs_rank_sum
        top_rank_cs_dict['alg_ranks'] = first_seq_rank_index + second_seq_rank_index
        top_rank_cs_dict['rank_sum'] = cs_rank_sum
        top_rank_cs_dict['seq_starts'] = (first_seq_cs_start_position, second_seq_cs_start_position)
        top_rank_cs_dict['consensus_len'] = cs_len

    return longest_cs_len, min_cs_rank_sum

def do_parent_comparisons(
    combo_level, 
    alg_combo, 
    rank_index, 
    first_seq_scan_generator_fns_dict, 
    first_seq_did_comparison_dict
    ):

    # If the comparison involves a first seq that is a consensus seq,
    # then check whether the comparison to generate that first seq was itself performed
    if combo_level > 2:
        first_seq_combo_level = combo_level - 1
        first_seq_alg_combo = alg_combo[: -1]
        first_seq_rank_index = rank_index[: -1]
        if not first_seq_did_comparison_dict[combo_level][first_seq_rank_index]:
            do_parent_comparisons(first_seq_combo_level, first_seq_alg_combo, first_seq_rank_index, first_seq_scan_generator_fns_dict[first_seq_combo_level])
    # Else the first seq is an irreducible single-alg seq
    else:
        # Perform the comparison
        first_seq_cs_start_position, second_seq_cs_start_position, cs_len, first_seq_rank_index, second_seq_rank_index =\
            next(first_seq_scan_generator_fns_dict[combo_level][rank_index])
        first_seq_did_comparison_dict[combo_level][rank_index] = True
        if cs_len is not None:
            # Record the consensus seq found from the comparison
            scan_common_substrings_info_for_alg_combo_dict[(first_seq_rank_index, second_seq_rank_index)] =\
                first_encoded_seq_dict[first_seq_rank_index][first_seq_cs_start_position: first_seq_cs_start_position + cs_len]
    return

def do_seq_comparisons(
    first_encoded_seq_dict,
    second_encoded_seq_dict,
    consensus_min_len,
    first_seq_lcs_rank_index=None,
    first_seq_encoded_lcs=None,
    first_seq_trcs_rank_index=None,
    first_seq_encoded_trcs=None, 
    alg_max_rank_dict=None
    ):
    # yield: first_seq_lcs_start_position, second_seq_lcs_start_position, lcs_len, first_seq_rank_index, second_seq_rank_index

    def do_seq_comparison(
        first_seq_rank_index, 
        first_encoded_seq, 
        second_seq_rank_index, 
        second_encoded_seq
        ):

        if len(second_encoded_seq) == 0:
            return None, None, None, first_seq_rank_index, second_seq_rank_index

        else:
            # Make the seq vectors orthogonal
            first_encoded_seq = first_encoded_seq.reshape(first_encoded_seq.size, 1)
            # Fill in the 2D matrix formed by the dimensions of the seq vectors with the AA's of the second seq
            tiled_second_encoded_seq = np.tile(second_encoded_seq, (first_encoded_seq.size, 1))
            # Project the first seq over the 2D matrix to find any identical AA's
            match_arr = np.equal(first_encoded_seq, tiled_second_encoded_seq).astype(int)

            # Find any common substrings, which are diagonals of True values in match_arr
            # Diagonal index 0 is the main diagonal
            # Negatively indexed diagonals lie below the main diagonal
            # Consideration of diagonals can be restricted to those
            # that can contain common substrings longer than the minimum length
            diags = [match_arr.diagonal(d)
                        for d in range(-len(first_encoded_seq) + consensus_min_len,
                                    len(second_encoded_seq) - consensus_min_len)]

            # Identify common substrings in the diagonals
            lcs_len = consensus_min_len
            found_long_consensus = False
            # Loop through bottom left min length diagonal to upper right min length diagonal
            for diag_index, diag in enumerate(diags):
                # Create and loop through two groups of Trues (common substrings) and Falses
                # from the elements of the diagonal
                for match_status, diag_group in groupby(diag):
                    # If considering a common substring
                    if match_status:
                        consensus_len = sum(diag_group)
                        # Retain the longest common substring, preferring the upper-rightmost LCS
                        # if multiple LCS's of equal length are present
                        if consensus_len >= lcs_len:
                            found_long_consensus = True
                            lcs_len = consensus_len
                            # Record the diagonal's index starting from the zero of the lower leftmost corner
                            lcs_diag_index = diag_index
                            lcs_diag = diag

            if found_long_consensus:
                # Find where the LCS resides in the selected diagonal
                # Take the first LCS if multiple LCS's of equal length are present in the diagonal
                for diag_aa_position in range(lcs_diag.size - lcs_len + 1):
                    for lcs_aa_position in range(lcs_len):
                        if not lcs_diag[diag_aa_position + lcs_aa_position]:
                            break
                    else:
                        diag_lcs_start_position = diag_aa_position
                        break

                # Determine the position of the first LCS AA in the first and second seqs
                # Reindex the LCS-containing diagonal to the main diagonal
                upper_left_diag_index = first_encoded_seq.size - consensus_min_len
                relative_lcs_diag_index = lcs_diag_index - upper_left_diag_index
                # Negatively indexed diagonals lie below the main diagonal
                if relative_lcs_diag_index < 0:
                    first_seq_lcs_start_position = diag_lcs_start_position - relative_lcs_diag_index
                    second_seq_lcs_start_position = diag_lcs_start_position
                else:
                    first_seq_lcs_start_position = diag_lcs_start_position
                    second_seq_lcs_start_position = relative_lcs_diag_index + diag_lcs_start_position

                # Pause the loop,
                # returning the position of the first AA in the first and second seqs,
                # the length of the LCS,
                # the ranks of the first and second seqs
                return first_seq_lcs_start_position, second_seq_lcs_start_position, lcs_len, first_seq_rank_index, second_seq_rank_index
            else:
                return None, None, None, first_seq_rank_index, second_seq_rank_index

    if first_seq_lcs_rank_index:
        max_possible_cs_len = first_seq_encoded_lcs.size
        for second_seq_rank_index, second_encoded_seq in second_encoded_seq_dict.items():
            first_seq_cs_start_position, second_seq_cs_start_position, cs_len, first_seq_rank_index, second_seq_rank_index = do_seq_comparison(
                first_seq_lcs_rank_index, 
                first_seq_encoded_lcs, 
                second_seq_rank_index, 
                second_encoded_seq
                )

            if cs_len == max_possible_cs_len:
                yield first_seq_cs_start_position, second_seq_cs_start_position, cs_len, first_seq_rank_index, second_seq_rank_index
                break
        else:
            yield None, None, None, first_seq_rank_index, second_seq_rank_index

    if first_seq_trcs_rank_index:
        first_rank_sum = sum(first_seq_trcs_rank_index)
        for second_seq_rank_index, second_encoded_seq in second_encoded_seq_dict.items():
            first_seq_cs_start_position, second_seq_cs_start_position, cs_len, first_seq_rank_index, second_seq_rank_index = do_seq_comparison(
                first_seq_trcs_rank_index, 
                first_seq_encoded_trcs, 
                second_seq_rank_index, 
                second_encoded_seq
                )

            # If all of the parent seqs are rank 0
            # or only 1 of the parent seqs is rank 1,
            # then it will be impossible to find a CS with a lower sum rank,
            # so stop searching for additional CS's
            top_rank_cs_found = False
            second_rank_sum = sum(second_seq_rank_index)
            if first_rank_sum + second_rank_sum <= 1:
                top_rank_cs_found = True

            # In addition, the following rule can demonstrate that the seq is a T-R CS:
            # The potential rank reduction of the second seq <= 1
            # AND 
            # there is no total rank increment of the first seq's parent seqs > 1
            else:
                if sum(second_seq_rank_index) <= 1:
                    potential_rank_reduction = 0
                    for first_seq_parent_index, first_seq_parent_rank in enumerate(first_seq_rank_index):
                        if alg_max_rank_dict[first_seq_algs[first_seq_parent_index]] > 0:
                            potential_rank_reduction += 1
                    if potential_rank_reduction > 1:
                        top_rank_cs_found = True

            if top_rank_cs_found:
                yield first_seq_cs_start_position, second_seq_cs_start_position, cs_len, first_seq_rank_index, second_seq_rank_index

        else:
            yield None, None, None, first_seq_rank_index, second_seq_rank_index

    for first_seq_rank_index, first_encoded_seq in first_encoded_seq_dict.items():
        for second_seq_rank_index, second_encoded_seq in second_encoded_seq_dict.items():
            yield do_seq_comparison(
                first_seq_rank_index, 
                first_encoded_seq, 
                second_seq_rank_index, 
                second_encoded_seq
                )

def make_seq_prediction_dict(
    consensus_scan,
    alg_consensus_source_df_for_scan_dict=None,
    alg_consensus_source_df_for_scan=None, 
    alg=None, 
    cs_info_dict=None,
    cs_type_list=None,
    alg_combo=None
    ):

    # If no common substring searches have yet been performed
    if cs_info_dict is None:
        prediction_dict = {}.fromkeys(config.single_alg_prediction_dict_cols['general'])
        for k in prediction_dict:
            if k == 'scan':
                prediction_dict['scan'] = consensus_scan
            elif k == 'measured mass':
                prediction_dict['measured mass'] = alg_consensus_source_df_for_scan.at[0, 'measured mass']
            elif k == 'is top rank single alg':
                prediction_dict['is top rank single alg'] = 1
            elif k == 'seq':
                prediction_dict['seq'] = seq = alg_consensus_source_df_for_scan.at[0, 'seq']
            elif k == 'len':
                prediction_dict['len'] = alg_consensus_source_df_for_scan.at[0, 'encoded seq'].size
        
        alg_prediction_dict = {}.fromkeys(config.single_alg_prediction_dict_cols[alg])
        if alg == 'novor':
            aa_score = alg_consensus_source_df_for_scan.at[0, 'aa score']
            for k in alg_prediction_dict:
                if k == 'retention time':
                    alg_prediction_dict['retention time'] = alg_consensus_source_df_for_scan.at[0, 'retention time']
                elif k == 'is novor seq':
                    alg_prediction_dict['is novor seq'] = 1
                elif k == 'avg novor aa score':
                    alg_prediction_dict['avg novor aa score'] = alg_consensus_source_df_for_scan.at[0, 'avg aa score']
                elif k == 'mono-di isobaric sub score':
                    mono_di_isobaric_sub_score = 0
                    for peptide in config.mono_di_isobaric_subs:
                        if peptide in seq:
                            for match_group in finditer(peptide, seq):
                                mono_di_isobaric_sub_score += 100 - sum(aa_score[match_group.start(): match_group.end()]) / len(peptide)
                    alg_prediction_dict['mono-di isobaric sub score'] = mono_di_isobaric_sub_score
                elif k == 'di isobaric sub score':
                    di_isobaric_sub_score = 0
                    for peptide in config.di_isobaric_subs:
                        if peptide in seq:
                            for match_group in finditer(peptide, seq):
                                di_isobaric_sub_score += 100 - sum(aa_score[match_group.start(): match_group.end()]) / 2
                    alg_prediction_dict['di isobaric sub score'] = di_isobaric_sub_score
                elif k == 'mono-di near-isobaric sub score':
                    mono_di_near_isobaric_sub_score = 0
                    for peptide in config.mono_di_near_isobaric_subs:
                        if peptide in seq:
                            for match_group in finditer(peptide, seq):
                                mono_di_near_isobaric_sub_score += 100 - sum(aa_score[match_group.start(): match_group.end()]) / len(peptide)
                    alg_prediction_dict['mono-di near-isobaric sub score'] = mono_di_near_isobaric_sub_score
                elif k == 'di near-isobaric sub score':
                    di_near_isobaric_sub_score = 0
                    for peptide in config.di_near_isobaric_subs:
                        if peptide in seq:
                            for match_group in finditer(peptide, seq):
                                di_near_isobaric_sub_score += 100 - sum(aa_score[match_group.start(): match_group.end()]) / 2
                    alg_prediction_dict['di near-isobaric sub score'] = di_near_isobaric_sub_score
        elif alg == 'pn':
            for k in alg_prediction_dict:
                if k == 'is pn seq':
                    alg_prediction_dict['is pn seq'] = 1
                elif k == 'rank score':
                    alg_prediction_dict['rank score'] = alg_consensus_source_df_for_scan.at[0, 'rank score']
                elif k == 'pn score':
                    alg_prediction_dict['pn score'] = alg_consensus_source_df_for_scan.at[0, 'pn score']
                elif k == 'sqs':
                    alg_prediction_dict['sqs'] = alg_consensus_source_df_for_scan.at[0, 'sqs']
        elif alg == 'deepnovo':
            aa_score = alg_consensus_source_df_for_scan.at[0, 'aa score']
            for k in alg_prediction_dict:
                if k == 'is deepnovo seq':
                    alg_prediction_dict['is deepnovo seq'] = 1
                elif k == 'avg deepnovo aa score':
                    alg_prediction_dict['avg deepnovo aa score'] = alg_consensus_source_df_for_scan.at[0, 'avg aa score']
                elif k == 'mono-di isobaric sub score':
                    mono_di_isobaric_sub_score = 0
                    for peptide in config.mono_di_isobaric_subs:
                        if peptide in seq:
                            for match_group in finditer(peptide, seq):
                                mono_di_isobaric_sub_score += 100 - sum(aa_score[match_group.start(): match_group.end()]) / len(peptide)
                    alg_prediction_dict['mono-di isobaric sub score'] = mono_di_isobaric_sub_score
                elif k == 'di isobaric sub score':
                    di_isobaric_sub_score = 0
                    for peptide in config.di_isobaric_subs:
                        if peptide in seq:
                            for match_group in finditer(peptide, seq):
                                di_isobaric_sub_score += 100 - sum(aa_score[match_group.start(): match_group.end()]) / 2
                    alg_prediction_dict['di isobaric sub score'] = di_isobaric_sub_score
                elif k == 'mono-di near-isobaric sub score':
                    mono_di_near_isobaric_sub_score = 0
                    for peptide in config.mono_di_near_isobaric_subs:
                        if peptide in seq:
                            for match_group in finditer(peptide, seq):
                                mono_di_near_isobaric_sub_score += 100 - sum(aa_score[match_group.start(): match_group.end()]) / len(peptide)
                    alg_prediction_dict['mono-di near-isobaric sub score'] = mono_di_near_isobaric_sub_score
                elif k == 'di near-isobaric sub score':
                    di_near_isobaric_sub_score = 0
                    for peptide in config.di_near_isobaric_subs:
                        if peptide in seq:
                            for match_group in finditer(peptide, seq):
                                di_near_isobaric_sub_score += 100 - sum(aa_score[match_group.start(): match_group.end()]) / 2
                    alg_prediction_dict['di near-isobaric sub score'] = di_near_isobaric_sub_score

        prediction_dict.update(alg_prediction_dict)

        return prediction_dict

    else:
        last_alg_consensus_source_df = alg_consensus_source_df_for_scan_dict[alg_combo[-1]]
        prediction_dict = {}.fromkeys(config.consensus_prediction_dict_cols['general'])
        selection_seq_start = cs_info_dict['seq_starts'][1]
        selection_seq_end = selection_seq_start + cs_info_dict['consensus_len']
        last_seq_rank_index = cs_info_dict['alg_ranks'][-1]
        for k in prediction_dict:
            if k == 'scan':
                prediction_dict['scan'] = consensus_scan
            elif k == 'measured mass':
                prediction_dict['measured mass'] = last_alg_consensus_source_df.at[last_seq_rank_index, 'measured mass']
            elif k == 'seq':
                cs_info_dict['encoded_consensus_seq'] = last_alg_consensus_source_df.at[last_seq_rank_index, 'encoded seq'][selection_seq_start: selection_seq_end]
                prediction_dict['seq'] = cs_info_dict['consensus_seq'] = consensus_seq = last_alg_consensus_source_df.at[last_seq_rank_index, 'seq'][selection_seq_start: selection_seq_end]
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

        for i, alg in enumerate(alg_combo):
            alg_rank = cs_info_dict['alg_ranks'][i]
            alg_prediction_dict = {}.fromkeys(config.consensus_prediction_dict_cols[alg])
            if alg == 'novor':
                novor_consensus_source_df = alg_consensus_source_df_for_scan_dict['novor']
                consensus_aa_score = novor_consensus_source_df.at[alg_rank, 'aa score'][selection_seq_start: selection_seq_end]
                for k in alg_prediction_dict:
                    if k == 'retention time':
                        alg_prediction_dict['retention time'] = novor_consensus_source_df.at[alg_rank, 'retention time']
                    elif k == 'is novor seq':
                        alg_prediction_dict['is novor seq'] = 1
                    elif k == 'fraction novor parent len':
                        alg_prediction_dict['fraction novor parent len'] = cs_info_dict['consensus_len'] / novor_consensus_source_df.at[alg_rank, 'encoded seq'].size
                    elif k == 'avg novor aa score':
                        alg_prediction_dict['avg novor aa score'] = sum(consensus_aa_score) / cs_info_dict['consensus_len']
                    elif k == 'mono-di isobaric sub score':
                        mono_di_isobaric_sub_score = 0
                        for peptide in config.mono_di_isobaric_subs:
                            if peptide in consensus_seq:
                                for match_group in finditer(peptide, consensus_seq):
                                    mono_di_isobaric_sub_score += 100 - sum(consensus_aa_score[match_group.start(): match_group.end()]) / len(peptide)
                        alg_prediction_dict['mono-di isobaric sub score'] = mono_di_isobaric_sub_score
                    elif k == 'di isobaric sub score':
                        di_isobaric_sub_score = 0
                        for peptide in config.di_isobaric_subs:
                            if peptide in consensus_seq:
                                for match_group in finditer(peptide, consensus_seq):
                                    di_isobaric_sub_score += 100 - sum(consensus_aa_score[match_group.start(): match_group.end()]) / 2
                        alg_prediction_dict['di isobaric sub score'] = di_isobaric_sub_score
                    elif k == 'mono-di near-isobaric sub score':
                        mono_di_near_isobaric_sub_score = 0
                        for peptide in config.mono_di_near_isobaric_subs:
                            if peptide in consensus_seq:
                                for match_group in finditer(peptide, consensus_seq):
                                    mono_di_near_isobaric_sub_score += 100 - sum(consensus_aa_score[match_group.start(): match_group.end()]) / len(peptide)
                        alg_prediction_dict['mono-di near-isobaric sub score'] = mono_di_near_isobaric_sub_score
                    elif k == 'di near-isobaric sub score':
                        di_near_isobaric_sub_score = 0
                        for peptide in config.di_near_isobaric_subs:
                            if peptide in consensus_seq:
                                for match_group in finditer(peptide, consensus_seq):
                                    di_near_isobaric_sub_score += 100 - sum(consensus_aa_score[match_group.start(): match_group.end()]) / 2
                        alg_prediction_dict['di near-isobaric sub score'] = di_near_isobaric_sub_score
            elif alg == 'pn':
                pn_consensus_source_df = alg_consensus_source_df_for_scan_dict['pn']
                for k in alg_prediction_dict:
                    if k == 'is pn seq':
                        alg_prediction_dict['is pn seq'] = 1
                    elif k == 'fraction pn parent len':
                        alg_prediction_dict['fraction pn parent len'] = cs_info_dict['consensus_len'] / pn_consensus_source_df.at[alg_rank, 'encoded seq'].size
                    elif k == 'rank score':
                        alg_prediction_dict['rank score'] = pn_consensus_source_df.at[alg_rank, 'rank score']
                    elif k == 'pn score':
                        alg_prediction_dict['pn score'] = pn_consensus_source_df.at[alg_rank, 'pn score']
                    elif k == 'pn rank':
                        alg_prediction_dict['pn rank'] = alg_rank
                    elif k == 'sqs':
                        alg_prediction_dict['sqs'] = pn_consensus_source_df.at[alg_rank, 'sqs']
            elif alg == 'deepnovo':
                deepnovo_consensus_source_df = alg_consensus_source_df_for_scan_dict['deepnovo']
                consensus_aa_score = deepnovo_consensus_source_df.at[alg_rank, 'aa score'][selection_seq_start: selection_seq_end]
                for k in alg_prediction_dict:
                    if k == 'is deepnovo seq':
                        alg_prediction_dict['is deepnovo seq'] = 1
                    elif k == 'fraction deepnovo parent len':
                        alg_prediction_dict['fraction deepnovo parent len'] = cs_info_dict['consensus_len'] / deepnovo_consensus_source_df.at[alg_rank, 'encoded seq'].size
                    elif k == 'deepnovo rank':
                        alg_prediction_dict['deepnovo rank'] = alg_rank
                    elif k == 'avg deepnovo aa score':
                        alg_prediction_dict['avg deepnovo aa score'] = sum(consensus_aa_score) / cs_info_dict['consensus_len']
                    elif k == 'mono-di isobaric sub score':
                        mono_di_isobaric_sub_score = 0
                        for peptide in config.mono_di_isobaric_subs:
                            if peptide in consensus_seq:
                                for match_group in finditer(peptide, consensus_seq):
                                    mono_di_isobaric_sub_score += 100 - sum(consensus_aa_score[match_group.start(): match_group.end()]) / len(peptide)
                        alg_prediction_dict['mono-di isobaric sub score'] = mono_di_isobaric_sub_score
                    elif k == 'di isobaric sub score':
                        di_isobaric_sub_score = 0
                        for peptide in config.di_isobaric_subs:
                            if peptide in consensus_seq:
                                for match_group in finditer(peptide, consensus_seq):
                                    di_isobaric_sub_score += 100 - sum(consensus_aa_score[match_group.start(): match_group.end()]) / 2
                        alg_prediction_dict['di isobaric sub score'] = di_isobaric_sub_score
                    elif k == 'mono-di near-isobaric sub score':
                        mono_di_near_isobaric_sub_score = 0
                        for peptide in config.mono_di_near_isobaric_subs:
                            if peptide in consensus_seq:
                                for match_group in finditer(peptide, consensus_seq):
                                    mono_di_near_isobaric_sub_score += 100 - sum(consensus_aa_score[match_group.start(): match_group.end()]) / len(peptide)
                        alg_prediction_dict['mono-di near-isobaric sub score'] = mono_di_near_isobaric_sub_score
                    elif k == 'di near-isobaric sub score':
                        di_near_isobaric_sub_score = 0
                        for peptide in config.di_near_isobaric_subs:
                            if peptide in consensus_seq:
                                for match_group in finditer(peptide, consensus_seq):
                                    di_near_isobaric_sub_score += 100 - sum(consensus_aa_score[match_group.start(): match_group.end()]) / 2
                        alg_prediction_dict['di near-isobaric sub score'] = di_near_isobaric_sub_score
            prediction_dict.update(alg_prediction_dict)

        return prediction_dict

def encode_seqs(df, consensus_min_len):

    df['encoded seq'] = df['seq'].where(df['seq'].str.len() >= consensus_min_len)
    df['encoded seq'].fillna('', inplace = True)
    df['encoded seq'] = df['encoded seq'].apply(list)
    map_ord = partial(map, ord)
    df['encoded seq'] = df['encoded seq'].apply(map_ord).apply(list)
    df['encoded seq'] = df['encoded seq'].apply(np.array).apply(lambda x: x - config.unicode_decimal_A)
    return

def make_combo_level_alg_dict(alg_df_dict):
    combo_level_alg_dict = OrderedDict()
    for combo_level in range(2, len(alg_df_dict) + 1):
        combo_level_alg_dict[combo_level] = []
        combo_level_alg_dict[combo_level] += [alg_combo for alg_combo in config.alg_combo_list
                                              if len(alg_combo) == combo_level]
    return combo_level_alg_dict

def add_measured_mass_col(alg_df_dict):

    # mass is reported in novor, pn output
    for alg in ['novor', 'pn']:
        alg_df = alg_df_dict[alg]
        alg_df['measured mass'] = alg_df['m/z'] * alg_df['charge']
    # mass is not reported in deepnovo output,
    # and is therefore transferred into the deepnovo table from novor
    novor_df = alg_df_dict['novor']
    for alg in ['deepnovo']:
        alg_df = alg_df_dict[alg]
        indices = [(scan, 0) for scan in alg_df.index.get_level_values('scan').tolist()]
        alg_df['measured mass'] = novor_df.ix[indices]['measured mass'].tolist()
        
    return alg_df_dict

def make_alg_consensus_source_df_dict(highest_level_alg_combo, alg_df_dict):
    alg_consensus_source_df_dict = OrderedDict().fromkeys(highest_level_alg_combo)
    for alg in alg_consensus_source_df_dict:
        alg_consensus_source_df_dict[alg] = alg_df_dict[alg][config.prediction_dict_source_cols[alg]]
    return alg_consensus_source_df_dict

def setup_scan_info_dicts(combo_level_alg_dict):

    scan_consensus_info_dict = OrderedDict().fromkeys(combo_level_alg_dict)
    scan_generator_fns_dict = OrderedDict().fromkeys(combo_level_alg_dict)
    did_comparison_dict = OrderedDict().fromkeys(combo_level_alg_dict)
    scan_common_substrings_info_dict = OrderedDict().fromkeys(combo_level_alg_dict)

    for i, combo_level in enumerate(combo_level_alg_dict):
        combo_level_alg_dict_for_combo_level = combo_level_alg_dict[combo_level]
        scan_consensus_info_dict[combo_level] = OrderedDict().fromkeys(combo_level_alg_dict_for_combo_level)
        for alg_combo in combo_level_alg_dict_for_combo_level:
            scan_consensus_info_dict[combo_level][alg_combo] = {}
            consensus_info_keys = ['alg_ranks', 'rank_sum', 'seq_starts', 'consensus_len']
            scan_consensus_info_dict[combo_level][alg_combo]['longest_cs'] = {}.fromkeys(consensus_info_keys)
            scan_consensus_info_dict[combo_level][alg_combo]['top_rank_cs'] = {}.fromkeys(consensus_info_keys)

        scan_generator_fns_dict[combo_level] = OrderedDict().fromkeys(combo_level_alg_dict_for_combo_level)
        did_comparison_dict[combo_level] = OrderedDict().fromkeys(combo_level_alg_dict_for_combo_level)
        scan_common_substrings_info_dict[combo_level] = OrderedDict().fromkeys(combo_level_alg_dict_for_combo_level)

    return scan_consensus_info_dict, scan_generator_fns_dict, did_comparison_dict, scan_common_substrings_info_dict

def make_rank_comparison_dict(scan_consensus_info_dict):

    rank_comparison_dict = OrderedDict()

    for combo_level in scan_consensus_info_dict:
        rank_comparison_dict[combo_level] = OrderedDict()
        for alg_combo in scan_consensus_info_dict[combo_level]:
            last_ranks_list = []
            alg_ranks_ranges = []

            for alg in alg_combo:
                last_ranks_list.append(config.seqs_reported_per_alg_dict[alg])
                alg_ranks_ranges.append(range(last_ranks_list[-1]))
            rank_comparison_dict[combo_level][alg_combo] = list(product(*alg_ranks_ranges))

    return rank_comparison_dict

# Join encoded seq cols by merge on scan, retaining seq ranks

# Find various LCS, top-ranking CS for each scan, alg combo

# Consider each alg combo:
# e.g., algs 1x2, algs 1x3, algs 2x3, algs 1x2x3
# Each of these will have a LCS, top-ranking CS
# Those seqs may be the same or different

# Simple, "brute-force" method
# Go through each possible seq comparison with generator functions
# Select T-R CS (using rank sum), LCS
# This selection step is different than the complicated method, I think

# Complicated, faster method

# A generator function generates CS's for each rank combo
# Ex. rank combo: alg 1: 1 rank, alg 2: 2 ranks, alg 3: 3 ranks
# The CS's are stored in a data frame
# Ex. cols: Alg 1 rank  Alg 2 rank  CS      (2 alg combo)
#           1           1           ...
#           1           2           ...
# The rank combo loop is paused if LCS and top-ranking CS are found
# LCS is found if LCS len = lesser parent seq len
# top-ranking CS is found when there can be no other higher rank sum from rank comparisons
# Ex. Alg 2 rank 1 x Alg 3 rank 3 is the first CS satisfying len requirement found
#     Sum rank = 4
#     Alg 2 rank 2 x Alg 3 rank 1 has not been considered (sum rank = 3)
#     Must continue until this rank combo has been considered
# Generator functions are stored in a data-structure
# e.g., {('Novor', 'Peaks'): generator0, ('Novor', 'PN'): generator1, ...}
    
# Paused generators may be restarted when higher-order alg combos are considered
# Lower-order LCS and T-R CS are first considered to find higher-order LCS and T-R CS
# Ex. 2-alg LCS and T-R CS were found in Alg 1 rank 1 x Alg 2 rank 1
#     These two seqs are compared against Alg 3 ranks
# If N-alg LCS creates a CS with Alg N+1, the CS MUST be the N+1-alg LCS
# Likewise, if N-alg T-R CS creates a CS with Alg N+1, the CS MUST be the N+1-alg T-R CS
# But say that the N+1-alg LCS or T-R CS is not found from the N-alg LCS and T-R CS
# Then the unconsidered N-alg rank comparisons may be considered:
# If N+1-alg LCS was not found,
# consider ALL unconsidered lower-order rank combos,
# even if the N+1-alg T-R CS was found
# If the N+1-alg T-R CS was not found despite the N+1-alg LCS being found,
# consider lower-order rank combos until the N+1 order top rank sum criterion is fulfilled

# The previously-generated CS's for lower-order alg combos were stored
# If the lower-order LCS was not found by matching the lesser parent length,
# Then EVERY CS is stored, as every CS was generated in order to find the LCS

# The first set of comparisons to alg N is from the lower-order CS's that didn't make the T-R cut
# Go through the CS's of sequentially lower rank below the T-R CS and compare to alg N
# Ignore the LCS if it is in this set
# These first comparisons do not reactivate the generator
# The second set of comparisons continues with unconsidered rank combos
# For each CS yielded, it is stored in the data structure of generated CS's,
# And it is immediately compared to alg N

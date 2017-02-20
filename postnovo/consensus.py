import numpy as np
import pandas as pd
import sys
import time

from config import *
from utils import (invert_dict_of_lists, save_pkl_objects,
                   load_pkl_objects)

from itertools import groupby, combinations, product
from collections import OrderedDict
from functools import partial
from multiprocessing import Pool

def make_prediction_df(alg_df_name_dict, tol_df_name_dict, alg_tol_dict, user_min_len, train, cores, alg_list):

    if train:
        consensus_min_len = train_consensus_len
    else:
        consensus_min_len = user_min_len

    tol_alg_dict = invert_dict_of_lists(alg_tol_dict)

    tol_prediction_df_list = []
    for tol in tol_alg_dict:
        alg_compar_list = tol_alg_dict[tol]
        
        if len(alg_compar_list) > 1:
            df_name_compar_list = tol_df_name_dict[tol]
            
            alg_df_dict = OrderedDict([(alg, alg_df_name_dict[alg][df_name_compar_list[i]])
                                       for i, alg in enumerate(alg_compar_list)])
            tol_prediction_df = make_prediction_df_for_tol(consensus_min_len, alg_df_dict, tol, cores, tol_alg_dict)
            tol_prediction_df_list.append(tol_prediction_df)

    prediction_df = pd.concat(tol_prediction_df_list)
    grouped_by_scan = prediction_df.groupby(['scan'])
    if 'retention time' in prediction_df.columns:
        prediction_df['retention time'] = grouped_by_scan['retention time'].transform(lambda scan_group: scan_group.max())

    for tol in tol_alg_dict:
        prediction_df[tol].fillna(0, inplace = True)

    alg_combo_group_col_list = []
    for alg in alg_list:
        is_alg_col_name = 'is ' + alg + ' seq'
        prediction_df[is_alg_col_name].fillna(0, inplace = True)
        prediction_df[is_alg_col_name] = prediction_df[is_alg_col_name].astype(int)
        alg_combo_group_col_list.append(is_alg_col_name)
    alg_combo_group_col_list.append('scan')
    prediction_df.set_index(alg_combo_group_col_list, inplace = True)

    return prediction_df

def make_prediction_df_for_tol(consensus_min_len, alg_df_dict, tol, cores, tol_alg_dict):

    for alg, df in alg_df_dict.items():
        df = encode_seqs(df, consensus_min_len)

    combo_level_alg_dict, full_alg_combo_list = get_combo_level_data(alg_df_dict)
    highest_level_alg_combo = list(full_alg_combo_list)[-1]
    consensus_source_df = setup_consensus_source_df(highest_level_alg_combo, alg_df_dict)

    scan_list_for_consensus = list(consensus_source_df.index.levels[0])
    scan_consensus_info_dict, scan_generator_fns_dict, scan_common_substrings_info_dict = setup_scan_info_dicts(full_alg_combo_list, consensus_source_df, combo_level_alg_dict)
    first_seq_second_seq_rank_comparisons_dict, first_seq_second_seq_max_ranks_dict, first_seq_second_seq_alg_positions_dict =\
        make_first_seq_second_seq_comparisons_dicts(scan_consensus_info_dict)
    r = time.time()
    rank_encoded_seqs_lists, max_seq_len_lists = make_scan_info_lists(consensus_source_df, highest_level_alg_combo, cores)
    r = time.time() - r

    consensus_scan_index_list = list(range(len(scan_list_for_consensus)))
    grand_scan_prediction_dict_list = []

    #multiprocessing_pool = Pool(cores)
    #multiprocessing_make_scan_prediction_dicts = partial(
    #    make_scan_prediction_dicts, scan_list_for_consensus = scan_list_for_consensus, rank_encoded_seqs_lists = rank_encoded_seqs_lists,
    #    max_seq_len_lists = max_seq_len_lists, consensus_source_df = consensus_source_df, scan_consensus_info_dict = scan_consensus_info_dict,
    #    scan_generator_fns_dict = scan_generator_fns_dict, scan_common_substrings_info_dict = scan_common_substrings_info_dict,
    #    consensus_min_len = consensus_min_len, highest_level_alg_combo = highest_level_alg_combo,
    #    first_seq_second_seq_rank_comparisons_dict = first_seq_second_seq_rank_comparisons_dict,
    #    first_seq_second_seq_max_ranks_dict = first_seq_second_seq_max_ranks_dict,
    #    first_seq_second_seq_alg_positions_dict = first_seq_second_seq_alg_positions_dict)
    #grand_scan_prediction_dict_list = multiprocessing_pool.map(multiprocessing_make_scan_prediction_dicts, consensus_scan_index_list)
    #multiprocessing_pool.close()
    #multiprocessing_pool.join()

    t = time.time()
    for consensus_scan_index in consensus_scan_index_list:
        grand_scan_prediction_dict_list.append(
            make_scan_prediction_dicts(consensus_scan_index, scan_list_for_consensus, rank_encoded_seqs_lists,
                                       max_seq_len_lists, consensus_source_df, scan_consensus_info_dict, scan_generator_fns_dict,
                                       scan_common_substrings_info_dict, consensus_min_len, highest_level_alg_combo,
                                       first_seq_second_seq_rank_comparisons_dict, first_seq_second_seq_max_ranks_dict,
                                       first_seq_second_seq_alg_positions_dict))
    print(r)
    print(time.time() - t)

    scan_prediction_dict_list = [scan_prediction_dict
                                 for scan_prediction_dict_list in grand_scan_prediction_dict_list
                                 for scan_prediction_dict in scan_prediction_dict_list]
    tol_prediction_df = pd.DataFrame().from_dict(scan_prediction_dict_list)
    tol_prediction_df[tol] = 1
    
    return tol_prediction_df

def make_scan_prediction_dicts(
    consensus_scan_index, scan_list_for_consensus, rank_encoded_seqs_lists, max_seq_len_lists,
    consensus_source_df, scan_consensus_info_dict, scan_generator_fns_dict, scan_common_substrings_info_dict,
    consensus_min_len, highest_level_alg_combo, first_seq_second_seq_rank_comparisons_dict,
    first_seq_second_seq_max_ranks_dict, first_seq_second_seq_alg_positions_dict):

    consensus_scan = scan_list_for_consensus[consensus_scan_index]
    print(consensus_scan)
    scan_rank_encoded_seqs_tuple = rank_encoded_seqs_lists[consensus_scan_index]
    scan_max_seq_len_tuple = max_seq_len_lists[consensus_scan_index]

    scan_prediction_dict_list = []
    scan_consensus_source_df = consensus_source_df.xs(consensus_scan)

    for alg in highest_level_alg_combo:
        scan_prediction_dict_list.append(
            make_seq_prediction_dict(scan_consensus_source_df, consensus_scan, alg = alg))

    for combo_level in scan_consensus_info_dict:
        first_seq_second_seq_alg_positions_for_combo_level_dict = first_seq_second_seq_alg_positions_dict[combo_level]
        first_seq_second_seq_max_ranks_for_combo_level_dict = first_seq_second_seq_max_ranks_dict[combo_level]
        first_seq_second_seq_rank_comparisons_for_combo_level_dict = first_seq_second_seq_rank_comparisons_dict[combo_level]

        for alg_combo in scan_consensus_info_dict[combo_level]:
            first_seq_second_seq_alg_positions_for_alg_combo_dict = first_seq_second_seq_alg_positions_for_combo_level_dict[alg_combo]
            first_seq_second_seq_max_ranks_for_alg_combo_list = first_seq_second_seq_max_ranks_for_combo_level_dict[alg_combo]
            first_seq_second_seq_rank_comparisons_list = first_seq_second_seq_rank_comparisons_for_combo_level_dict[alg_combo]

            scan_common_substrings_info_dict[combo_level][alg_combo] =\
                OrderedDict().fromkeys(first_seq_second_seq_rank_comparisons_list)

            if combo_level == 2:

                first_seq_alg = alg_combo[0]
                first_seq_alg_position_in_scan_info_tuples = first_seq_second_seq_alg_positions_for_alg_combo_dict[first_seq_alg][0]
                second_seq_alg = alg_combo[1]
                second_seq_alg_position_in_scan_info_tuples = first_seq_second_seq_alg_positions_for_alg_combo_dict[second_seq_alg][0]

                #rank_first_encoded_seq_pairs = scan_consensus_source_df[: seqs_reported_per_alg_dict[first_seq_alg]][first_seq_alg]['encoded seq'].to_dict().items()
                #first_encoded_seq_dict = OrderedDict(
                #    [((rank,), encoded_seq) for rank, encoded_seq in rank_first_encoded_seq_pairs])

                #rank_encoded_seq_list = next(rank_encoded_seqs[first_seq_alg])
                #first_encoded_seq_dict = OrderedDict(
                #    [((rank,), encoded_seq) for rank, encoded_seq in enumerate(rank_encoded_seq_list)])

                alg_rank_encoded_seq_list = scan_rank_encoded_seqs_tuple[first_seq_alg_position_in_scan_info_tuples]
                first_encoded_seq_dict = OrderedDict(
                    [((rank,), encoded_seq) for rank, encoded_seq in enumerate(alg_rank_encoded_seq_list)])

                # For numba
                #first_encoded_seq_dict = OrderedDict()
                #for rank, encoded_seq in scan_consensus_source_df[: scan_last_ranks_considered[0] + 1][alg_combo[0]]['encoded seq'].to_dict().items():
                #    first_encoded_seq_dict[(rank,)] = encoded_seq

                #rank_second_encoded_seq_pairs = scan_consensus_source_df[: seqs_reported_per_alg_dict[second_seq_alg]][second_seq_alg]['encoded seq'].to_dict().items()
                #second_encoded_seq_dict = OrderedDict(
                #    [((rank,), encoded_seq) for rank, encoded_seq in rank_second_encoded_seq_pairs])

                #rank_encoded_seq_list = next(rank_encoded_seqs[second_seq_alg])
                #second_encoded_seq_dict = OrderedDict(
                #    [((rank,), encoded_seq) for rank, encoded_seq in enumerate(rank_encoded_seq_list)])

                alg_rank_encoded_seq_list = scan_rank_encoded_seqs_tuple[second_seq_alg_position_in_scan_info_tuples]
                second_encoded_seq_dict = OrderedDict(
                    [((rank,), encoded_seq) for rank, encoded_seq in enumerate(alg_rank_encoded_seq_list)])

                # For numba
                #second_encoded_seq_dict = OrderedDict()
                #for rank, encoded_seq in scan_consensus_source_df[: scan_last_ranks_considered[1] + 1][alg_combo[1]]['encoded seq'].to_dict().items():
                #    second_encoded_seq_dict[(rank,)] = encoded_seq

                max_possible_cs_len = min(scan_max_seq_len_tuple)

            else:
                # scan_common_substrings_info_dict[combo_level - 1][some_alg_combo]
                # provides rank indices and 
                # scan_consensus_info_dict[combo_level - 1][same_alg_combo]
                # lcs and tr cs
                # 

                # If any combo_level - 1 alg_combo doesn't have a lcs (or top_rank_cs),
                # Then there will be NO CS AT ALL for combo_level

                # Two dicts:
                # one is commons_substrings_dict[combo_level - 1][appropriate string not including new alg under consideration]
                # = {(0, 0): encoded seq 0, (0, 1): encoded seq 1, ...}
                # the other is like dict = {0: encoded seq 0, 1: encoded seq 1, ...}
                # Could be useful
                #scan_consensus_source_df.xs(key = 'encoded seq', axis = 1, level = 1, drop_level = False)
                pass

            seq_comparison_generator = do_seq_comparisons(first_encoded_seq_dict, second_encoded_seq_dict, consensus_min_len)
            scan_generator_fns_dict[combo_level][alg_combo] = seq_comparison_generator
            longest_cs_len = 0
            min_cs_rank_sum = 1000

            longest_cs_dict = scan_consensus_info_dict[combo_level][alg_combo]['longest_cs']
            longest_cs_dict = longest_cs_dict.fromkeys(longest_cs_dict, None)
            top_rank_cs_dict = scan_consensus_info_dict[combo_level][alg_combo]['top_rank_cs']
            top_rank_cs_dict = top_rank_cs_dict.fromkeys(top_rank_cs_dict, None)

            for first_seq_cs_start_position, second_seq_cs_start_position, cs_len, first_seq_rank_index, second_seq_rank_index in seq_comparison_generator:

                if cs_len is not None:
                    scan_common_substrings_info_dict[(first_seq_rank_index, second_seq_rank_index)] =\
                        first_encoded_seq_dict[first_seq_rank_index][first_seq_cs_start_position: first_seq_cs_start_position + cs_len]

                    first_rank_sum = sum(first_seq_rank_index)
                    second_rank_sum = sum(second_seq_rank_index)
                    cs_rank_sum = first_rank_sum + second_rank_sum

                    if cs_len > longest_cs_len:
                        longest_cs_len = cs_len
                        longest_cs_dict['alg_ranks'] = (first_seq_rank_index, second_seq_rank_index)
                        longest_cs_dict['rank_sum'] = cs_rank_sum
                        longest_cs_dict['seq_starts'] = (first_seq_cs_start_position, second_seq_cs_start_position)
                        longest_cs_dict['consensus_len'] = cs_len

                    if cs_rank_sum < min_cs_rank_sum:
                        min_cs_rank_sum = cs_rank_sum
                        top_rank_cs_dict['alg_ranks'] = (first_seq_rank_index, second_seq_rank_index)
                        top_rank_cs_dict['rank_sum'] = cs_rank_sum
                        top_rank_cs_dict['seq_starts'] = (first_seq_cs_start_position, second_seq_cs_start_position)
                        top_rank_cs_dict['consensus_len'] = cs_len

                    if longest_cs_len == max_possible_cs_len:
                        for first_seq_parent_index, first_seq_parent_rank in enumerate(first_seq_rank_index):
                            if first_seq_parent_rank + sum(first_seq_rank_index[: first_seq_parent_index]) < cs_rank_sum\
                                and first_seq_parent_rank + 1 < first_seq_second_seq_max_ranks_for_alg_combo_list[first_seq_parent_index]:
                                break
                        else:
                            break

            if longest_cs_dict['alg_ranks'] is not None:
                first_seq_second_seq_alg_positions_subdict = first_seq_second_seq_alg_positions_dict[combo_level][alg_combo]

                if longest_cs_dict['alg_ranks'] == top_rank_cs_dict['alg_ranks']:
                    cs_prediction_dict, longest_cs_dict = make_seq_prediction_dict(
                        scan_consensus_source_df, consensus_scan, cs_info_dict = longest_cs_dict,
                        cs_type_list = ['longest', 'top rank'], alg_combo = alg_combo,
                        first_seq_second_seq_alg_positions_subdict = first_seq_second_seq_alg_positions_subdict)
                    scan_prediction_dict_list.append(cs_prediction_dict)
                    top_rank_cs_dict.update(longest_cs_dict)

                else:
                    longest_cs_prediction_dict, longest_cs_dict = make_seq_prediction_dict(
                        scan_consensus_source_df, consensus_scan, cs_info_dict = longest_cs_dict,
                        cs_type_list = ['longest'], alg_combo = alg_combo,
                        first_seq_second_seq_alg_positions_subdict = first_seq_second_seq_alg_positions_subdict)
                    top_rank_cs_prediction_dict, top_rank_cs_dict = make_seq_prediction_dict(
                        scan_consensus_source_df, consensus_scan, cs_info_dict = top_rank_cs_dict,
                        cs_type_list = ['top rank'], alg_combo = alg_combo,
                        first_seq_second_seq_alg_positions_subdict = first_seq_second_seq_alg_positions_subdict)
                    scan_prediction_dict_list.append(longest_cs_prediction_dict)
                    scan_prediction_dict_list.append(top_rank_cs_prediction_dict)

    return scan_prediction_dict_list

def make_seq_prediction_dict(scan_consensus_source_df, scan, alg = None, cs_info_dict = None, cs_type_list = None, alg_combo = None, first_seq_second_seq_alg_positions_subdict = None):

    if cs_info_dict is None:
        prediction_dict = {}.fromkeys(single_alg_prediction_dict_cols['general'])
        for k in prediction_dict:
            if k == 'scan':
                prediction_dict['scan'] = scan
            elif k == 'is top rank single alg':
                prediction_dict['is top rank single alg'] = 1
            elif k == 'seq':
                prediction_dict['seq'] = scan_consensus_source_df.at[0, (alg, 'seq')]
            elif k == 'len':
                prediction_dict['len'] = scan_consensus_source_df.at[0, (alg, 'encoded seq')].size
            elif k == 'avg rank':
                prediction_dict['avg rank'] = 0
        
        alg_prediction_dict = {}.fromkeys(single_alg_prediction_dict_cols[alg])
        if alg == 'novor':
            for k in alg_prediction_dict:
                if k == 'retention time':
                    alg_prediction_dict['retention time'] = scan_consensus_source_df.at[0, (alg, 'retention time')]
                elif k == 'is novor seq':
                    alg_prediction_dict['is novor seq'] = 1
                elif k == 'avg novor aa score':
                    alg_prediction_dict['avg novor aa score'] = scan_consensus_source_df.at[0, (alg, 'avg aa score')]
        elif alg == 'peaks':
            pass
        elif alg == 'pn':
            for k in alg_prediction_dict:
                if k == 'is pn seq':
                    alg_prediction_dict['is pn seq'] = 1
                elif k == 'rank score':
                    alg_prediction_dict['rank score'] = scan_consensus_source_df.at[0, (alg, 'rank score')]
                elif k == 'pn score':
                    alg_prediction_dict['pn score'] = scan_consensus_source_df.at[0, (alg, 'pn score')]
                elif k == 'pn rank':
                    alg_prediction_dict['pn rank'] = 0
                elif k == 'sqs':
                    alg_prediction_dict['sqs'] = scan_consensus_source_df.at[0, (alg, 'sqs')]
        prediction_dict.update(alg_prediction_dict)

        return prediction_dict

    else:
        prediction_dict = {}.fromkeys(consensus_prediction_dict_cols['general'])
        selection_seq_start = cs_info_dict['seq_starts'][1]
        selection_seq_end = selection_seq_start + cs_info_dict['consensus_len']
        for k in prediction_dict:
            if k == 'scan':
                prediction_dict['scan'] = scan
            elif k == 'seq':
                cs_info_dict['encoded_consensus_seq'] = scan_consensus_source_df.at[cs_info_dict['alg_ranks'][1][0], (alg_combo[-1], 'encoded seq')][selection_seq_start: selection_seq_end]
                prediction_dict['seq'] = cs_info_dict['consensus_seq'] = scan_consensus_source_df.at[cs_info_dict['alg_ranks'][1][0], (alg_combo[-1], 'seq')][selection_seq_start: selection_seq_end]
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

        for alg in alg_combo:
            alg_rank = cs_info_dict['alg_ranks'][first_seq_second_seq_alg_positions_subdict[alg][0]][first_seq_second_seq_alg_positions_subdict[alg][1]]
            alg_prediction_dict = {}.fromkeys(consensus_prediction_dict_cols[alg])
            if alg == 'novor':
                for k in alg_prediction_dict:
                    if k == 'retention time':
                        alg_prediction_dict['retention time'] = scan_consensus_source_df.at[alg_rank, (alg, 'retention time')]
                    elif k == 'is novor seq':
                        alg_prediction_dict['is novor seq'] = 1
                    elif k == 'fraction novor parent len':
                        alg_prediction_dict['fraction novor parent len'] = cs_info_dict['consensus_len'] / scan_consensus_source_df.at[alg_rank, (alg, 'encoded seq')].size
                    elif k == 'avg novor aa score':
                        alg_prediction_dict['avg novor aa score'] = sum(scan_consensus_source_df.at[alg_rank, (alg, 'aa score')][selection_seq_start: selection_seq_end]) / cs_info_dict['consensus_len']
            elif alg == 'peaks':
                pass
            elif alg == 'pn':
                for k in alg_prediction_dict:
                    if k == 'is pn seq':
                        alg_prediction_dict['is pn seq'] = 1
                    elif k == 'fraction pn parent len':
                        alg_prediction_dict['fraction pn parent len'] = cs_info_dict['consensus_len'] / scan_consensus_source_df.at[alg_rank, (alg, 'encoded seq')].size
                    elif k == 'rank score':
                        alg_prediction_dict['rank score'] = scan_consensus_source_df.at[alg_rank, (alg, 'rank score')]
                    elif k == 'pn score':
                        alg_prediction_dict['pn score'] = scan_consensus_source_df.at[alg_rank, (alg, 'pn score')]
                    elif k == 'pn rank':
                        alg_prediction_dict['pn rank'] = alg_rank
                    elif k == 'sqs':
                        alg_prediction_dict['sqs'] = scan_consensus_source_df.at[alg_rank, (alg, 'sqs')]
            prediction_dict.update(alg_prediction_dict)

        return prediction_dict, cs_info_dict

    return

def setup_consensus_source_df(highest_level_alg_combo, alg_df_dict):

    alg = highest_level_alg_combo[0]
    consensus_source_df = alg_df_dict[alg][prediction_dict_source_cols[alg]]
    consensus_source_df.columns = [[alg] * len(prediction_dict_source_cols[alg]),
                                   consensus_source_df.columns]

    for alg in highest_level_alg_combo[1:]:
        right_join_df = alg_df_dict[alg][prediction_dict_source_cols[alg]]
        right_join_df.columns = [[alg] * len(prediction_dict_source_cols[alg]),
                                 right_join_df.columns]
        consensus_source_df = consensus_source_df.join(right_join_df, how = 'outer')

    for alg in highest_level_alg_combo:
        seqs_reported_per_alg = seqs_reported_per_alg_dict[alg]
        consensus_source_df = consensus_source_df[
            consensus_source_df[alg]['seq'].groupby(level = 0).transform(
                lambda scan_group: scan_group.count() == seqs_reported_per_alg)]
        # pandas does not automatically drop levels from the index
        consensus_source_df.set_index(
            pd.MultiIndex.from_tuples(consensus_source_df.index.values, names = ['scan', 'rank']), inplace = True)

    return consensus_source_df

def setup_scan_info_dicts(full_alg_combo_list, consensus_source_df, combo_level_alg_dict):

    scan_consensus_info_dict = OrderedDict().fromkeys(combo_level_alg_dict)
    scan_generator_fns_dict = OrderedDict().fromkeys(combo_level_alg_dict)
    scan_common_substrings_info_dict = OrderedDict().fromkeys(combo_level_alg_dict)

    for combo_level in combo_level_alg_dict:
        scan_consensus_info_dict[combo_level] = OrderedDict().fromkeys(combo_level_alg_dict[combo_level])
        scan_generator_fns_dict[combo_level] = OrderedDict().fromkeys(combo_level_alg_dict[combo_level])
        scan_common_substrings_info_dict[combo_level] = OrderedDict().fromkeys(combo_level_alg_dict[combo_level])

    for alg_combo in full_alg_combo_list:
        scan_consensus_info_dict[combo_level][alg_combo] = {}
        consensus_info_keys = ['alg_ranks', 'rank_sum', 'seq_starts', 'consensus_len']
        scan_consensus_info_dict[combo_level][alg_combo]['longest_cs'] = {}.fromkeys(consensus_info_keys)
        scan_consensus_info_dict[combo_level][alg_combo]['top_rank_cs'] = {}.fromkeys(consensus_info_keys)

    return scan_consensus_info_dict, scan_generator_fns_dict, scan_common_substrings_info_dict

def extract_consensus_strings(scan_consensus_info_dict, full_alg_combo_list, consensus_source_df, alg_df_dict):

    for alg_combo in full_alg_combo_list:
        scan_info_dict = scan_consensus_info_dict[alg_combo]

        #consensus_df_cols = general_consensus_df_col_names
        #for alg in alg_combo:
        #    consensus_df_cols.append(consensus_df_col_names_for_alg

        for scan in scan_info_dict:
            selection_alg = alg_combo[-1]
            scan_info = scan_info_dict[scan]

            for consensus_type in ['longest_cs', 'top_rank_cs']:
                cs_info = scan_info[consensus_type]
                try:
                    selection_seq_start = cs_info['seq_starts'][1]
                    selection_seq_end = selection_seq_start + cs_info['consensus_len']
                    cs_info['consensus_seq'] = consensus_source_df.ix[
                        (scan, cs_info['alg_ranks'][1])][selection_alg]['seq'][selection_seq_start: selection_seq_end]
                except TypeError:
                    cs_info['consensus_seq'] = None

    return scan_consensus_info_dict

def fill_tol_prediction_df(tol_prediction_df, alg_df_dict, scan_consensus_info_dict):
    # Loop through each scan in tol_prediction_df
    # Add single alg top rank data
    # Add consensus data
    # Extract consensus seqs from seq strings in alg_df_dict
    # 

    ## Get rank 0 rows
    #single_alg_top_rank_df = alg_df_dict[alg][alg_df_dict[alg].index.get_level_values(1) == 0]
    #for drop_col in drop_cols:
    #    try:
    #        single_alg_top_rank_df.drop(drop_col, axis = 1, inplace = True)
    #    except ValueError:
    #        pass

    ## To slice seqs by start and stop value
    #df.reset_index().apply(lambda x: str.__getitem__(x.loc['seq'], slice(*(d[x.loc['index']], e[x.loc['index']]))), 1)

    return tol_prediction_df

#@jit
def do_seq_comparisons(first_encoded_seq_dict, second_encoded_seq_dict, consensus_min_len):

    not_seq_prediction = np.nan

    for first_seq_rank_index, first_encoded_seq in first_encoded_seq_dict.items():
        for second_seq_rank_index, second_encoded_seq in second_encoded_seq_dict.items():

            if first_encoded_seq is not_seq_prediction or second_encoded_seq is not_seq_prediction:
                yield None, None, None, None, None
            else:
                first_encoded_seq = first_encoded_seq.reshape(first_encoded_seq.size, 1)
                tiled_second_encoded_seq = np.tile(second_encoded_seq, (first_encoded_seq.size, 1))
                match_arr = np.equal(first_encoded_seq, tiled_second_encoded_seq).astype(int)

                diags = [match_arr.diagonal(d)
                         for d in range(-len(first_encoded_seq) + consensus_min_len,
                                        len(second_encoded_seq) - consensus_min_len)]
                #diags = []
                #for d in range(-first_encoded_seq.size + consensus_min_len, second_encoded_seq.size - consensus_min_len):
                #    diags.append(match_arr.diagonal(d))

                ## For numba
                #lcs_diag_index = 0
                #lcs_diag = np.arange(0)

                lcs_len = consensus_min_len
                found_long_consensus = False
                for diag_index, diag in enumerate(diags):
                    for match_status, diag_group in groupby(diag):
                        if match_status:
                            consensus_len = sum(diag_group)
                            if consensus_len >= lcs_len:
                                found_long_consensus = True
                                lcs_len = consensus_len
                                lcs_diag_index = diag_index
                                lcs_diag = diag

                if found_long_consensus:

                    ## For numba
                    #diag_lcs_start_position = 0

                    for diag_aa_position in range(lcs_diag.size - lcs_len + 1):
                        for lcs_aa_position in range(lcs_len):
                            if not lcs_diag[diag_aa_position + lcs_aa_position]:
                                break
                        else:
                            diag_lcs_start_position = diag_aa_position
                            break

                    upper_left_diag_index = first_encoded_seq.size - consensus_min_len
                    relative_lcs_diag_index = lcs_diag_index - upper_left_diag_index
                    if relative_lcs_diag_index < 0:
                        first_seq_lcs_start_position = diag_lcs_start_position - relative_lcs_diag_index
                        second_seq_lcs_start_position = diag_lcs_start_position
                    else:
                        first_seq_lcs_start_position = diag_lcs_start_position
                        second_seq_lcs_start_position = relative_lcs_diag_index + diag_lcs_start_position

                    yield first_seq_lcs_start_position, second_seq_lcs_start_position, lcs_len, first_seq_rank_index, second_seq_rank_index
                else:
                    yield None, None, None, None, None

def make_first_seq_second_seq_comparisons_dicts(scan_consensus_info_dict):

    first_seq_second_seq_rank_comparisons_dict = OrderedDict()
    first_seq_second_seq_max_ranks_dict = OrderedDict()
    first_seq_second_seq_alg_positions_dict = OrderedDict()
    for combo_level in scan_consensus_info_dict:
        first_seq_second_seq_rank_comparisons_dict[combo_level] = OrderedDict()
        first_seq_second_seq_max_ranks_dict[combo_level] = OrderedDict()
        first_seq_second_seq_alg_positions_dict[combo_level] = OrderedDict()
        for alg_combo in scan_consensus_info_dict[combo_level]:
            alg_ranks_ranges = []
            last_ranks_list = []
            for alg in alg_combo:
                last_ranks_list.append(seqs_reported_per_alg_dict[alg])
                alg_ranks_ranges.append(range(last_ranks_list[-1]))
            first_seq_second_seq_max_ranks_dict[combo_level][alg_combo] = last_ranks_list
            rank_comparisons = list(product(*alg_ranks_ranges))
            first_seq_second_seq_rank_comparisons_dict[combo_level][alg_combo] = [
                ((rank_comparison[:-1]), (rank_comparison[-1],)) for rank_comparison in rank_comparisons]

            first_seq_second_seq_alg_positions_dict[combo_level][alg_combo] = OrderedDict()
            for i, alg in enumerate(alg_combo[:-1]):
                first_seq_second_seq_alg_positions_dict[combo_level][alg_combo][alg] = (0, i)
            first_seq_second_seq_alg_positions_dict[combo_level][alg_combo][alg_combo[-1]] = (1, 0)

    return first_seq_second_seq_rank_comparisons_dict, first_seq_second_seq_max_ranks_dict, first_seq_second_seq_alg_positions_dict

def make_scan_info_lists(consensus_source_df, highest_level_alg_combo, cores):

    rank_encoded_seqs_lists = []
    max_seq_len_lists = []

    ### TEMPORARY ###
    cores = 1
    #################

    if cores > 1:
        total_scans = consensus_source_df.index.levels[0].size
        first_index_number_split = int(total_scans / cores)

        index_number_splits = [(0, first_index_number_split)]
        scan_index_splits = [(consensus_source_df.index.levels[0][0],
                              consensus_source_df.index.levels[0][first_index_number_split])]

        for core_number in range(2, cores):
            index_number_splits.append((index_number_splits[-1][1] + 1, first_index_number_split * core_number))
            scan_index_splits.append(
                (consensus_source_df.index.levels[0][index_number_splits[-1][0]],
                 consensus_source_df.index.levels[0][index_number_splits[-1][1]]))

        index_number_splits.append(
            (index_number_splits[-1][1] + 1, total_scans - 1))
        scan_index_splits.append(
            (consensus_source_df.index.levels[0][index_number_splits[-1][0]],
             consensus_source_df.index.levels[0][index_number_splits[-1][1]]))

    for alg in highest_level_alg_combo:
        if cores == 1:
            encoded_seq_group_for_alg = consensus_source_df[alg].groupby(level = 0)['encoded seq']
            rank_encoded_seqs_lists.append(encoded_seq_group_for_alg.agg(lambda g: g.dropna().tolist()).tolist())
            max_seq_len_lists.append(encoded_seq_group_for_alg.agg(lambda g: g.map(np.size).max()).tolist())

        else:
            multiprocessing_pool = Pool(cores)
            multiprocessing_make_scan_info_lists_for_alg_split = partial(make_scan_info_lists_for_alg_split, alg_consensus_source_df = consensus_source_df[alg])
            multiprocessing_info_list = multiprocessing_pool.map(multiprocessing_make_scan_info_lists_for_alg_split, scan_index_splits)
            multiprocessing_pool.close()
            multiprocessing_pool.join()

            rank_encoded_seqs_lists_for_alg, max_seq_len_lists_for_alg = zip(*multiprocessing_info_list)
            combined_rank_encoded_seqs_lists_for_alg = []
            combined_max_seq_lens_lists_for_alg = []
            for split in range(len(index_number_splits)):
                combined_rank_encoded_seqs_lists_for_alg += rank_encoded_seqs_lists_for_alg[split]
                combined_max_seq_lens_lists_for_alg += max_seq_len_lists_for_alg[split]
            rank_encoded_seqs_lists.append(combined_rank_encoded_seqs_lists_for_alg)
            max_seq_len_lists.append(combined_max_seq_lens_lists_for_alg)
        
    rank_encoded_seqs_lists = list(zip(*rank_encoded_seqs_lists))
    max_seq_len_lists = list(zip(*max_seq_len_lists))

    return rank_encoded_seqs_lists, max_seq_len_lists

def make_scan_info_lists_for_alg_split(scan_index_split, alg_consensus_source_df):

    encoded_seq_group_for_alg_split = alg_consensus_source_df.loc[scan_index_split[0]: scan_index_split[1]].groupby(level = 0)['encoded seq']
    rank_encoded_seqs_list_for_alg_split = encoded_seq_group_for_alg_split.agg(lambda g: g.dropna().tolist()).tolist()
    max_seq_len_list_for_alg_split = encoded_seq_group_for_alg_split.agg(lambda g: g.map(np.size).max()).tolist()

    return (rank_encoded_seqs_list_for_alg_split, max_seq_len_list_for_alg_split)

def encode_seqs(df, consensus_min_len):

    df['encoded seq'] = df['seq'].where(df['seq'].str.len() >= consensus_min_len)
    df['encoded seq'].fillna('', inplace = True)
    df['encoded seq'] = df['encoded seq'].apply(list)
    map_ord = partial(map, ord)
    df['encoded seq'] = df['encoded seq'].apply(map_ord).apply(list)
    df['encoded seq'] = df['encoded seq'].apply(np.array).apply(lambda x: x - unicode_decimal_A)

    return df

def get_combo_level_data(alg_df_dict):

    combo_level_alg_dict = OrderedDict()
    full_alg_combo_list = []
    for combo_level in range(2, len(alg_df_dict) + 1):
        combo_level_alg_dict[combo_level] = []
        alg_combo_list = [combo for combo in combinations(alg_df_dict, combo_level)]
        combo_level_alg_dict[combo_level] += alg_combo_list
        full_alg_combo_list += alg_combo_list

    return combo_level_alg_dict, full_alg_combo_list




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
    # Likewise, if N-alg T-R CS creates a CS with Alg N+1, the CS MUST be the N+1-alg LCS
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

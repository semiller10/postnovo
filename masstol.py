''' Compare sequences predicted with different fragment mass tolerances '''

import numpy as np
import pandas as pd
import sys

from functools import partial
from multiprocessing import Pool

if 'postnovo' in sys.modules:
    import postnovo.config as config
    import postnovo.utils as utils
else:
    import config
    import utils

def update_prediction_df(prediction_df):
    utils.verbose_print()

    if len(config.globals['frag_mass_tols']) == 1:
        return prediction_df

    utils.verbose_print('setting up mass tolerance comparison')
    prediction_df.reset_index(inplace = True)
    # combo level col = sum of 'is novor seq', 'is pn seq', 'is deepnovo seq' values
    prediction_df['combo level'] = prediction_df.iloc[:, :len(config.globals['algs'])].sum(axis = 1)
    scan_list = sorted(list(set(prediction_df['scan'])))
    one_percent_number_scans = len(scan_list) / 100 / config.globals['cpus']
    tol_group_key_list = []
    for i, tol in enumerate(config.globals['frag_mass_tols']):
        tol_group_key = [0] * len(config.globals['frag_mass_tols'])
        tol_group_key[-(i + 1)] = 1
        tol_group_key_list.append(tuple(tol_group_key))
    # set index as scan, '0.2' -> '0.7', combo level
    prediction_df.set_index(['scan'] + config.globals['frag_mass_tols'], inplace = True)
    # tol list indices are sorted backwards: 0.7 predictions come before 0.2 in scan group
    prediction_df.sort_index(level = ['scan'] + config.globals['frag_mass_tols'], inplace = True)
    mass_tol_compar_df = prediction_df[['seq', 'combo level']]
    scan_groups = mass_tol_compar_df.groupby(level = 'scan')

    ## Single process
    #tol_match_array_list = []
    #print_percent_progress_fn = partial(utils.print_percent_progress_singlethreaded,
    #                                    procedure_str = 'mass tolerance comparison progress: ',
    #                                    one_percent_total_count = one_percent_number_scans)
    #child_initialize(scan_groups,
    #                 config.globals['frag_mass_tols'],
    #                 tol_group_key_list,
    #                 print_percent_progress_fn)
    #utils.verbose_print('performing mass tolerance comparison')
    #for scan in scan_list:
    #    tol_match_array_list.append(make_mass_tol_match_array(scan))

    # Multiprocess
    print_percent_progress_fn = partial(utils.print_percent_progress_multithreaded,
                                        procedure_str = 'mass tolerance comparison progress: ',
                                        one_percent_total_count = one_percent_number_scans,
                                        cores = config.globals['cpus'])
    multiprocessing_pool = Pool(config.globals['cpus'],
                                initializer = child_initialize,
                                initargs = (scan_groups,
                                            config.globals['frag_mass_tols'],
                                            tol_group_key_list,
                                            print_percent_progress_fn)
                                )
    utils.verbose_print('performing mass tolerance comparison')
    tol_match_array_list = multiprocessing_pool.map(make_mass_tol_match_array, scan_list)
    multiprocessing_pool.close()
    multiprocessing_pool.join()

    tol_match_cols = [tol + ' seq match' for tol in config.globals['frag_mass_tols']]
    tol_match_df = pd.DataFrame(np.fliplr(np.concatenate(tol_match_array_list)),
                                index = prediction_df.index,
                                columns = tol_match_cols)
    prediction_df = pd.concat([prediction_df, tol_match_df], axis = 1)
    prediction_df.drop(['combo level'], axis = 1, inplace = True)
    prediction_df.reset_index(inplace = True)
    prediction_df.set_index(config.globals['is_alg_names'] + ['scan'], inplace = True)
    prediction_df.sort_index(level = ['scan'] + config.globals['is_alg_names'], inplace = True)

    return prediction_df

def child_initialize(_scan_groups, _frag_mass_tols, _tol_group_key_list, _print_percent_progress_fn):

    global scan_groups, frag_mass_tols, tol_group_key_list, print_percent_progress_fn

    scan_groups = _scan_groups
    frag_mass_tols = _frag_mass_tols
    tol_group_key_list = _tol_group_key_list
    print_percent_progress_fn = _print_percent_progress_fn

def make_mass_tol_match_array(scan):

    print_percent_progress_fn()

    scan_group_df = scan_groups.get_group(scan)
    scan_group_df.index = scan_group_df.index.droplevel('scan')
    tol_match_array = np.zeros([len(scan_group_df), len(frag_mass_tols)])
    tol_groups = scan_group_df.groupby(level = frag_mass_tols)

    for first_tol_group_key_index, first_tol_group_key in enumerate(tol_group_key_list[:-1]):
        try:
            first_tol_group_df = tol_groups.get_group(first_tol_group_key)
            try:
                first_tol_group_first_row_position_in_scan_group = scan_group_df.index.get_loc(first_tol_group_key).start
            # certain tols may have only one prediction
            except AttributeError:
                first_tol_group_first_row_position_in_scan_group = scan_group_df.index.get_loc(first_tol_group_key)


            for first_prediction_index, first_prediction_row in enumerate(first_tol_group_df.values):
                first_prediction_seq = first_prediction_row[0]
                first_prediction_combo_level = first_prediction_row[1]

                # compare seqs to those from each subsequent tol
                for second_tol_group_key_index, second_tol_group_key in enumerate(tol_group_key_list[first_tol_group_key_index + 1:]):
                    try:
                        second_tol_group_df = tol_groups.get_group(second_tol_group_key)

                        # for the first row of tol group df, find that row's position in scan group df
                        try:
                            second_tol_group_first_row_position_in_scan_group = scan_group_df.index.get_loc(second_tol_group_key).start
                        # the tol index may correspond to only one prediction instead of multiple
                        except AttributeError:
                            second_tol_group_first_row_position_in_scan_group = scan_group_df.index.get_loc(second_tol_group_key)

                        for second_prediction_index, second_prediction_row in enumerate(second_tol_group_df.values):
                            second_prediction_seq = second_prediction_row[0]
                            second_prediction_combo_level = second_prediction_row[1]

                            if first_prediction_seq in second_prediction_seq:
                                # if combo level > current highest level for tol comparison, as recorded in tol_match_array
                                if first_prediction_combo_level > tol_match_array[
                                    first_tol_group_first_row_position_in_scan_group + first_prediction_index,
                                    first_tol_group_key_index + second_tol_group_key_index + 1]:

                                    tol_match_array[
                                    first_tol_group_first_row_position_in_scan_group + first_prediction_index,
                                    first_tol_group_key_index + second_tol_group_key_index + 1] = second_prediction_combo_level

                            if second_prediction_seq in first_prediction_seq:
                                if second_prediction_combo_level > tol_match_array[
                                    second_tol_group_first_row_position_in_scan_group + second_prediction_index,
                                    first_tol_group_key_index]:

                                    tol_match_array[
                                    second_tol_group_first_row_position_in_scan_group + second_prediction_index,
                                    first_tol_group_key_index] = first_prediction_combo_level

                    # certain tols may not have predictions
                    except KeyError:
                        pass

        # certain tols may not have predictions
        except KeyError:
            pass

    return tol_match_array
''' Compare sequences predicted with different fragment mass tolerances '''

import numpy as np
import pandas as pd

from config import *
from utils import *

from multiprocessing import Pool, current_process

multiprocessing_scan_count = 0


def update_prediction_df(prediction_df):
    verbose_print()

    if len(tol_list) == 1:
        return prediction_df

    verbose_print('setting up comparison over mass tolerance parameterization')
    prediction_df.reset_index(inplace = True)
    # combo level col = sum of 'is novor seq', 'is peaks seq', 'is pn seq' values
    prediction_df['combo level'] = prediction_df.iloc[:, :len(alg_list)].sum(axis = 1)
    scan_list = sorted(list(set(prediction_df['scan'])))
    one_percent_number_scans = len(scan_list) / 100 / cores[0]
    tol_group_key_list = []
    for i, tol in enumerate(tol_list):
        tol_group_key = [0] * len(tol_list)
        tol_group_key[i] = 1
        tol_group_key_list.append(tuple(tol_group_key))
    # set index as scan, '0.2' -> '0.7', combo level
    prediction_df.set_index(['scan'] + tol_list, inplace = True)
    prediction_df.sort_index(level = ['scan'] + sorted(tol_list, reverse = True), inplace = True)
    mass_tol_comparison_df = prediction_df[['seq', 'combo level']]
    scan_groups = mass_tol_comparison_df.groupby(level = 'scan')

    ## single processor method
    #child_initialize(scan_groups, tol_list, tol_group_key_list, cores[0], one_percent_number_scans)
    #tol_match_array_list = []
    #verbose_print('comparing over mass tolerance parameterization')
    #for scan in scan_list:
    #    tol_match_array_list.append(make_mass_tol_match_array(scan))

    multiprocessing_pool = Pool(cores[0],
                                initializer = child_initialize,
                                initargs = (scan_groups, tol_list, tol_group_key_list,
                                            cores[0], one_percent_number_scans)
                                )
    verbose_print('comparing over mass tolerance parameterization')
    tol_match_array_list = multiprocessing_pool.map(make_mass_tol_match_array, scan_list)
    multiprocessing_pool.close()
    multiprocessing_pool.join()

    tol_match_cols = [tol + ' seq match' for tol in tol_list]
    tol_match_df = pd.DataFrame(np.concatenate(tol_match_array_list),
                                index = prediction_df.index,
                                columns = tol_match_cols)
    prediction_df = pd.concat([prediction_df, tol_match_df], axis = 1)
    prediction_df.drop(['combo level'], axis = 1, inplace = True)
    prediction_df.reset_index(inplace = True)
    prediction_df.set_index(alg_combo_group_col_list, inplace = True)
    prediction_df.sort_index(level = ['scan'] + alg_combo_group_col_list[:-1], inplace = True)

    sys.exit(0)

    return prediction_df

def child_initialize(_scan_groups, _tol_list, _tol_group_key_list, _cores, _one_percent_number_scans):

    global scan_groups, tol_list, tol_group_key_list, cores, one_percent_number_scans

    scan_groups = _scan_groups
    tol_list = _tol_list
    tol_group_key_list = _tol_group_key_list
    cores = _cores
    one_percent_number_scans = _one_percent_number_scans

    return

def make_mass_tol_match_array(scan):

    if current_process()._identity[0] % cores == 1:
        global multiprocessing_scan_count
        multiprocessing_scan_count += 1
        if int(multiprocessing_scan_count % one_percent_number_scans) == 0:
            percent_complete = int(multiprocessing_scan_count / one_percent_number_scans)
            if percent_complete <= 100:
                verbose_print_over_same_line('mass tolerance comparison progress: ' + str(percent_complete) + '%')

    scan_group_df = scan_groups.get_group(scan)
    scan_group_df.index = scan_group_df.index.droplevel('scan')
    tol_match_array = np.zeros([len(scan_group_df), len(tol_list)])
    tol_groups = scan_group_df.groupby(level = tol_list)

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
                                    first_tol_group_key_index]:

                                    tol_match_array[
                                    first_tol_group_first_row_position_in_scan_group + first_prediction_index,
                                    first_tol_group_key_index] = second_prediction_combo_level

                            if second_prediction_seq in first_prediction_seq:
                                if second_prediction_combo_level > tol_match_array[
                                    second_tol_group_first_row_position_in_scan_group + second_prediction_index,
                                    second_tol_group_key_index + 1]:

                                    tol_match_array[
                                    second_tol_group_first_row_position_in_scan_group + second_prediction_index,
                                    second_tol_group_key_index + 1] = first_prediction_combo_level

                    # certain tols may not have predictions
                    except KeyError:
                        pass

        # certain tols may not have predictions
        except KeyError:
            pass

    return tol_match_array
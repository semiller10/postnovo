''' Compare sequences predicted with different fragment mass tolerances '''

import numpy as np
import pandas as pd

from config import *
from utils import *


def update_prediction_df(prediction_df):

    #if len(tol_list) == 1:
    #    return prediction_df

    #scan_groups = prediction_df.groupby('scan')
    #for g in scan_groups:
    #    for r in g.iloc[:-1]:
    #        seq = r['seq']

    # Reset index
    prediction_df.reset_index(inplace = True)
    # Add a column for combo level = sum of 'is novor seq', 'is peaks seq', 'is pn seq'
    prediction_df['combo level'] = prediction_df.iloc[:, :len(alg_list)].sum(axis = 1)
    scan_list = sorted(list(set(prediction_df['scan'])))
    tol_group_key_list = []
    for i, tol in enumerate(tol_list):
        tol_group_key = [0] * len(tol_list)
        tol_group_key[i] = 1
        tol_group_key_list.append(tuple(tol_group_key))
    # Set index as scan, '0.2' -> '0.7', combo level
    prediction_df.set_index(['scan'] + tol_list, inplace = True)
    prediction_df.sort_index(level = ['scan'] + sorted(tol_list, reverse = True), inplace = True)
    mass_tol_comparison_df = prediction_df[['seq', 'combo level']]
    # Group by scan
    scan_groups = mass_tol_comparison_df.groupby(level = 'scan')
    # In new fn, analyze each scan group based on list of scans, allowing multiprocessing
    tol_match_array_list = []
    # Retrieve df for scan group
    for scan in scan_list:
        scan_group_df = scan_groups.get_group(scan)
    # Drop scan level from index
        scan_group_df.index = scan_group_df.index.droplevel('scan')
    # Make a numpy array of zeros with dims = # scan group df rows x len tol_list
        tol_match_array = np.zeros([len(scan_group_df), len(tol_list)])
    # Group by tols from tol_list, '0.2' -> '0.7'
        tol_groups = scan_group_df.groupby(level = tol_list)
    # Enumerate tol_list, ignoring last tol, '0.2' -> '0.6'
        for first_tol_group_key_index, first_tol_group_key in enumerate(tol_group_key_list[:-1]):
    # Retrieve df for each tol
            try:
                first_tol_group_df = tol_groups.get_group(first_tol_group_key)
            except KeyError:
                break
    # For the first row of tol group df, find where it is in scan group df
            try:
                first_tol_group_first_row_position_in_scan_group = scan_group_df.index.get_loc(first_tol_group_key).start
            except AttributeError:
                first_tol_group_first_row_position_in_scan_group = scan_group_df.index.get_loc(first_tol_group_key)
    # Enumerate each seq prediction row for the tol
            for first_prediction_index, first_prediction_row in enumerate(first_tol_group_df.values):
    # Retrieve seq and combo level for row
                first_prediction_seq = first_prediction_row[0]
                first_prediction_combo_level = first_prediction_row[1]
    # Enumerate each subsequent tol in tol_list, '0.3' -> '0.7'
                for second_tol_group_key_index, second_tol_group_key in enumerate(tol_group_key_list[first_tol_group_key_index + 1:]):
    # Retrieve df for tol
                    try:
                        second_tol_group_df = tol_groups.get_group(second_tol_group_key)
                    except KeyError:
                        break
    # For the first row of tol group df, find where it is in scan group df
                    try:
                        second_tol_group_first_row_position_in_scan_group = scan_group_df.index.get_loc(second_tol_group_key).start
                    except AttributeError:
                        second_tol_group_first_row_position_in_scan_group = scan_group_df.index.get_loc(second_tol_group_key)
    # Enumerate each seq prediction row for the tol
                    for second_prediction_index, second_prediction_row in enumerate(second_tol_group_df.values):
    # Retrieve combo level and seq for row
                        second_prediction_seq = second_prediction_row[0]
                        second_prediction_combo_level = second_prediction_row[1]
    # Is seq 0 in seq 1?
                        if first_prediction_seq in second_prediction_seq:
    # If so, is combo level > current highest level for tol comparison in np array element (scan group row #, comparison tol col #)
                            if first_prediction_combo_level > tol_match_array[
                                first_tol_group_first_row_position_in_scan_group + first_prediction_index,
                                first_tol_group_key_index]:
    # Then replace element with level
                                tol_match_array[
                                first_tol_group_first_row_position_in_scan_group + first_prediction_index,
                                first_tol_group_key_index] = second_prediction_combo_level
    # Is seq 1 in seq 0?
                        if second_prediction_seq in first_prediction_seq:
    # If so, is combo level > current highest level for tol comparison in np array element (scan group row #, comparison tol col #)
                            if second_prediction_combo_level > tol_match_array[
                                second_tol_group_first_row_position_in_scan_group + second_prediction_index,
                                second_tol_group_key_index + 1]:
    # Then replace element with level
                                tol_match_array[
                                second_tol_group_first_row_position_in_scan_group + second_prediction_index,
                                second_tol_group_key_index + 1] = first_prediction_combo_level
    # Return np array
        tol_match_array_list.append(tol_match_array)
    tol_match_cols = [tol + ' seq match' for tol in tol_list]
    # After multiprocessing closed, stack np arrays, covert to df
    tol_match_df = pd.DataFrame(np.concatenate(tol_match_array_list),
                                index = prediction_df.index,
                                columns = tol_match_cols)
    # Concatenate df column-wise with prediction_df
    prediction_df = pd.concat([prediction_df, tol_match_df], axis = 1)
    # Delete combo level col
    prediction_df.drop(['combo level'], axis = 1, inplace = True)
    # Reset index
    prediction_df.reset_index(inplace = True)
    # Set index again to 'is novor seq', etc. and 'scan'
    prediction_df.set_index(alg_combo_group_col_list, inplace = True)
    prediction_df.sort_index(level = ['scan'] + alg_combo_group_col_list[:-1], inplace = True)

    return prediction_df
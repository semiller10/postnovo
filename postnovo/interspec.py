''' Compare sequences predicted from different spectra but the same peptide '''

import numpy as np
import pandas as pd

from config import *
from utils import *

def update_prediction_df(prediction_df):
    verbose_print()

    verbose_print('setting up inter-spectrum comparison')
    prediction_df['mass error'] = prediction_df['measured mass'] * precursor_mass_tol[0] * 10**-6
    prediction_df.reset_index(inplace = True)
    prediction_df.set_index(is_alg_col_names, inplace = True)
    tol_group_key_list = []
    for i, tol in enumerate(tol_list):
        tol_group_key = [0] * len(tol_list)
        tol_group_key[i] = 1
        tol_group_key_list.append(tuple(tol_group_key))
    full_precursor_array_list = []

    for multiindex_key in is_alg_col_multiindex_keys:
        alg_combo = '-'.join([alg for i, alg in enumerate(alg_list) if multiindex_key[i]])

        alg_group_precursor_array_list = []
        alg_combo_df = prediction_df.xs(multiindex_key)
        alg_combo_df.reset_index(inplace = True)
        alg_combo_df.set_index(tol_list, inplace = True)

        for tol_group_key in tol_group_key_list:
            tol = tol_list[tol_group_key.index(1)]

            tol_group_precursor_array_list = []
            try:
                tol_df = alg_combo_df.xs(tol_group_key)[['seq', 'measured mass', 'mass error']]
            except TypeError:
                tol_df = alg_combo_df.xs(tol_group_key[0])[['seq', 'measured mass', 'mass error']]
            tol_df.sort('measured mass', inplace = True)
            measured_masses = tol_df['measured mass'].tolist()
            mass_errors = tol_df['mass error'].tolist()

            # precursor indices represent different spectra clustered by mass
            precursor_indices = []
            precursor_index = 0
            # assign the first seq prediction to precursor 0
            precursor_indices.append(0)
            previous_mass = tol_df.iat[0, 1]

            for mass_index, mass in enumerate(measured_masses[1:]):
                if mass - mass_errors[mass_index] > previous_mass:
                    precursor_index += 1
                    previous_mass = mass
                precursor_indices.append(precursor_index)
            tol_df['precursor index'] = precursor_indices
            precursor_groups = tol_df.groupby('precursor index')

            ## single processor method
            for precursor_index in range(precursor_indices[-1] + 1):
                child_initialize(precursor_groups)
                verbose_print('performing inter-spectrum comparison for', alg_combo, ',', tol, 'Da seqs')
                tol_group_precursor_array_list.append(find_precursor_array(precursor_index))

    # return array
    # append array to list of interspectrum arrays
                tol_group_precursor_array_list.append(precursor_array)
            alg_group_precursor_array_list += tol_group_precursor_array_list
    # concatenate arrays
        full_precursor_array_list += alg_group_precursor_array_list
    interspec_df = pd.DataFrame(np.concatenate(full_precursor_array_list),
                                    index = prediction_df.index,
                                    columns = ['mass group agreement', 'mass group size'])
    # concatenate full array columnwise with prediction_df
    prediction_df = pd.concat([prediction_df, interspec_df], axis = 1)

    return prediction_df

def child_initialize(_precursor_groups):
    global precursor_groups
    precursor_groups = _precursor_groups

def make_precursor_info_array(precursor_index):

    precursor_seqs = precursor_groups.get_group(precursor_index)['seq']
# make a nested list, with number of inner lists = group size
    precursor_group_size = len(precursor_seqs)
    all_seq_matches_list = [[] for i in range(precursor_group_size)]
# make an empty array of dims (# rows in group, 2)
    precursor_array = np.zeros((precursor_group_size, 2))
# first col of array is proportion of matching seqs in group
# second col is group size for each row
# loop through each seq row in group, ending with second to last
    for first_seq_index, first_seq in enumerate(precursor_seqs):
# record first group row index and seq
# inner list # (first group row index) append 1, representing self
        all_seq_matches_list[first_seq_index].append(1)
# loop through each subsequent seq row in group
        for second_seq_index, second_seq in enumerate(precursor_seqs[first_seq_index + 1:], start = 1):
# record second group row index and seq
# if first seq in second seq, inner list # (first group row index) append 1
            if first_seq in second_seq:
                all_seq_matches_list[first_seq_index].append(1)
# else append 0
            else:
                all_seq_matches_list[first_seq_index].append(0)
# if second seq in first seq, inner list # (second group row index) append 1
            if second_seq in first_seq:
                all_seq_matches_list[first_seq_index + second_seq_index].append(1)
# else append 0
            else:
                all_seq_matches_list[first_seq_index + second_seq_index].append(0)
# loop through rows of group again
    for seq_index, seq_matches in enumerate(all_seq_matches_list):
# record row index
# sum inner list corresponding to row index and record in first col of array
        precursor_array[seq_index, 0] = sum(seq_matches) / precursor_group_size
        precursor_array[seq_index, 1] = precursor_group_size

    return precursor_array
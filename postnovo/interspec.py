''' Compare sequences predicted from different spectra but the same peptide '''

import numpy as np

from config import *
from utils import *

def update_prediction_df(prediction_df):

    # make mass error col by multiplying ppm error by mass
    prediction_df['mass error'] = prediction_df['measured mass'] * precursor_mass_tol[0] * 10**-6
    # make a list for interspectrum matrices
    alg_combo_group_interspec_matrix_list = []
    # set index as alg and tol cols
    prediction_df.reset_index(inplace = True)
    prediction_df.set_index(is_alg_col_names, inplace = True)

    tol_group_key_list = []
    for i, tol in enumerate(tol_list):
        tol_group_key = [0] * len(tol_list)
        tol_group_key[i] = 1
        tol_group_key_list.append(tuple(tol_group_key))

    # extract xs for alg combo/tol, e.g., 0.2 novor-pn consensus seqs
    for multiindex_key in is_alg_col_multiindex_keys:
        alg_combo_df = prediction_df.xs(multiindex_key)
        alg_combo_df.reset_index(inplace = True)
        alg_combo_df.set_index(tol_list, inplace = True)
        measured_mass_col_index = tol_df.columns.tolist().index('measured mass')

        for tol_group_key in tol_group_key_list:
            try:
                tol_df = alg_combo_df.xs(tol_group_key)
            except TypeError:
                tol_df = alg_combo_df.xs(tol_group_key[0])
    # make a mass group col
            mass_indices = []
            measured_masses = tol_df.loc['measured mass'].tolist()
            mass_errors = tol_df.loc['mass error'].tolist()
    # sort df by mass
            tol_df.sort('measured mass', inplace = True)
    # start with mass group 0
            mass_index = 0
    # assign the first row to group 0
            mass_indices.append(0)
    # record the mass as previous mass
            previous_mass = tol_df.iat[0, -1]
    # loop through each mass, starting with second row
            for i, mass in enumerate(measured_masses):
    # if mass - error <= previous mass
                if mass - mass_errors[i] > previous_mass:
    # assign row to group under consideration
                    mass_index += 1
    # else increment group and assign to group
                mass_indices.append(mass_group)
    # groupby mass groups
            tol_df['mass index'] = mass_indices
            mass_groups = tol_df.groupby('mass index')
    # loop through each group
            for mass_index in mass_indices:
                mass_group_df = mass_groups.get_group(mass_index)
    # make a nested list, with number of inner lists = group size
                mass_group_size = len(mass_group_df)
                seq_matches = [[] for i in range(mass_group_size)]
    # make an empty matrix of dims (# rows in group, 2)
                
    # first col of matrix is proportion of matching seqs in group
    # second col is group size for each row
    # loop through each seq row in group, ending with second to last
    # record first group row index and seq
    # inner list # (first group row index) append 1, representing self
    # loop through each subsequent seq row in group
    # record second group row index and seq
    # if first seq in second seq, inner list # (first group row index) append 1
    # else append 0
    # if second seq in first seq, inner list # (second group row index) append 1
    # else append 0
    # loop through rows of group again
    # record row index
    # sum inner list corresponding to row index and record in first col of matrix
    # return matrix
    # append matrix to list of interspectrum matrices
    # concatenate matrices
    # concatenate full matrix columnwise with prediction_df

    return prediction_df
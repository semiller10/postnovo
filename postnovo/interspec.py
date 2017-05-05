''' Compare sequences predicted from different spectra but the same peptide '''

import numpy as np
import pandas as pd

import postnovo.config as config
import postnovo.utils as utils

#import config
#import utils

from multiprocessing import Pool, current_process

multiprocessing_precursor_count = 0


def update_prediction_df(prediction_df):
    utils.verbose_print()

    utils.verbose_print('setting up inter-spectrum comparison')
    prediction_df['mass error'] = prediction_df['measured mass'] * config.precursor_mass_tol[0] * 10**-6
    prediction_df.reset_index(inplace = True)
    prediction_df.set_index(config.is_alg_col_names, inplace = True)
    tol_group_key_list = []
    for i, tol in enumerate(config.frag_mass_tols):
        tol_group_key = [0] * len(config.frag_mass_tols)
        tol_group_key[i] = 1
        tol_group_key_list.append(tuple(tol_group_key))
    full_precursor_array_list = []

    for multiindex_key in config.is_alg_col_multiindex_keys:
        alg_combo = '-'.join([alg for i, alg in enumerate(config.alg_list) if multiindex_key[i]])

        alg_group_precursor_array_list = []
        alg_combo_df = prediction_df.xs(multiindex_key)
        alg_combo_df.reset_index(inplace = True)
        alg_combo_df.set_index(config.frag_mass_tols, inplace = True)

        for tol_group_key in tol_group_key_list:
            tol = config.frag_mass_tols[tol_group_key.index(1)]

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
            #tol_group_precursor_array_list = []
            #utils.verbose_print('performing inter-spectrum comparison for', alg_combo + ',', tol, 'Da seqs')
            #for precursor_index in range(precursor_indices[-1] + 1):
            #    child_initialize(precursor_groups)
            #    tol_group_precursor_array_list.append(make_precursor_info_array(precursor_index))

            ## multiprocessing method
            precursor_range = range(precursor_indices[-1] + 1)
            one_percent_number_precursors = len(precursor_range) / 100 / config.cores[0]
            multiprocessing_pool = Pool(config.cores[0],
                                        initializer = child_initialize,
                                        initargs = (precursor_groups, config.cores[0], one_percent_number_precursors)
                                        )
            utils.verbose_print('performing inter-spectrum comparison for', alg_combo + ',', tol, 'Da seqs')
            tol_group_precursor_array_list = multiprocessing_pool.map(make_precursor_info_array,
                                                                      precursor_range)
            multiprocessing_pool.close()
            multiprocessing_pool.join()

            alg_group_precursor_array_list += tol_group_precursor_array_list
        full_precursor_array_list += alg_group_precursor_array_list
    interspec_df = pd.DataFrame(np.concatenate(full_precursor_array_list),
                                index = prediction_df.index,
                                columns = ['precursor seq agreement', 'precursor seq count'])
    # concatenate full array columnwise with prediction_df
    prediction_df = pd.concat([prediction_df, interspec_df], axis = 1)

    prediction_df.drop(['measured mass', 'mass error'], axis = 1, inplace = True)
    prediction_df.reset_index(inplace = True)
    prediction_df.set_index(config.is_alg_col_names + ['scan'], inplace = True)
    prediction_df.sort_index(level = ['scan'] + config.is_alg_col_names, inplace = True)

    return prediction_df

def child_initialize(_precursor_groups, _cores = 1, _one_percent_number_precursors = None):
    global precursor_groups, cores, one_percent_number_precursors
    precursor_groups = _precursor_groups
    cores = _cores
    one_percent_number_precursors = _one_percent_number_precursors

def make_precursor_info_array(precursor_index):

    if current_process()._identity[0] % cores == 1:
        global multiprocessing_precursor_count
        multiprocessing_precursor_count += 1
        if int(multiprocessing_precursor_count % one_percent_number_precursors) == 0:
            percent_complete = int(multiprocessing_precursor_count / one_percent_number_precursors)
            if percent_complete <= 100:
                utils.verbose_print_over_same_line('inter-spectrum comparison progress: ' + str(percent_complete) + '%')

    precursor_seqs = precursor_groups.get_group(precursor_index)['seq']
    precursor_group_size = len(precursor_seqs)
    all_seq_matches_list = [[] for i in range(precursor_group_size)]
    precursor_array = np.zeros((precursor_group_size, 2))

    for first_seq_index, first_seq in enumerate(precursor_seqs):
        all_seq_matches_list[first_seq_index].append(1)

        for second_seq_index, second_seq in enumerate(precursor_seqs[first_seq_index + 1:], start = 1):
            if first_seq in second_seq:
                all_seq_matches_list[first_seq_index].append(1)
            else:
                all_seq_matches_list[first_seq_index].append(0)
            if second_seq in first_seq:
                all_seq_matches_list[first_seq_index + second_seq_index].append(1)
            else:
                all_seq_matches_list[first_seq_index + second_seq_index].append(0)

    for seq_index, seq_matches in enumerate(all_seq_matches_list):
        precursor_array[seq_index, 0] = sum(seq_matches) / precursor_group_size
        precursor_array[seq_index, 1] = precursor_group_size

    return precursor_array
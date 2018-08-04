''' Compare sequences predicted from different spectra but the same peptide '''

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

    utils.verbose_print('setting up inter-spectrum comparison')
    prediction_df['mass error'] = prediction_df['measured mass'] * config.globals['pre_mass_tol'] * 10**-6
    prediction_df.reset_index(inplace = True)
    prediction_df.set_index(config.globals['is_alg_names'], inplace = True)
    tol_group_key_list = []
    for i, tol in enumerate(config.globals['frag_mass_tols']):
        tol_group_key = [0] * len(config.globals['frag_mass_tols'])
        tol_group_key[i] = 1
        tol_group_key_list.append(tuple(tol_group_key))
    full_precursor_array_list = []

    for multiindex_key in config.globals['is_alg_keys']:
        alg_combo = '-'.join([alg for i, alg in enumerate(config.globals['algs']) if multiindex_key[i]])

        alg_group_precursor_array_list = []
        alg_combo_df = prediction_df.xs(multiindex_key)
        alg_combo_df.reset_index(inplace = True)
        alg_combo_df.set_index(config.globals['frag_mass_tols'], inplace = True)

        for tol_group_key in tol_group_key_list:
            tol = config.globals['frag_mass_tols'][tol_group_key.index(1)]

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
            precursor_range = range(precursor_indices[-1] + 1)
            one_percent_number_precursors = len(precursor_range) / 100 / config.globals['cpus']

            ## Single process
            #tol_group_precursor_array_list = []
            #print_percent_progress_fn = partial(utils.print_percent_progress_singlethreaded, 
            #                                    procedure_str = 'inter-spectrum comparison progress: ', 
            #                                    one_percent_total_count = one_percent_number_precursors)
            #utils.verbose_print('performing inter-spectrum comparison for', alg_combo + ',', tol, 'Da seqs')
            #child_initialize(precursor_groups,
            #                 print_percent_progress_fn)
            #for precursor_index in precursor_range:
            #    tol_group_precursor_array_list.append(make_precursor_info_array(precursor_index))

            # Multiprocess
            print_percent_progress_fn = partial(utils.print_percent_progress_multithreaded,
                                                procedure_str = 'inter-spectrum comparison progress: ',
                                                one_percent_total_count = one_percent_number_precursors,
                                                cores = config.globals['cpus'])
            multiprocessing_pool = Pool(config.globals['cpus'],
                                        initializer = child_initialize,
                                        initargs = (precursor_groups,
                                                    print_percent_progress_fn)
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

    # prediction_df.drop(['measured mass', 'mass error'], axis = 1, inplace = True)
    prediction_df.reset_index(inplace = True)
    prediction_df.set_index(config.globals['is_alg_names'] + ['scan'], inplace = True)
    prediction_df.sort_index(level = ['scan'] + config.globals['is_alg_names'], inplace = True)

    return prediction_df

def child_initialize(_precursor_groups, _print_percent_progress_fn):

    global precursor_groups, print_percent_progress_fn

    precursor_groups = _precursor_groups
    print_percent_progress_fn = _print_percent_progress_fn

def make_precursor_info_array(precursor_index):

    print_percent_progress_fn()

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
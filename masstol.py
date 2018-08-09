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

    utils.verbose_print('Setting up mass tolerance comparison')
    prediction_df.reset_index(inplace=True)
    #combo level col = sum of 'is novor seq', 'is pn seq', 'is deepnovo seq' values
    prediction_df['combo level'] = prediction_df.iloc[:, :len(config.globals['algs'])].sum(axis=1)
    spec_list = sorted(list(set(prediction_df['spec_id'])))
    one_percent_number_specs = len(spec_list) / 100 / config.globals['cpus']
    tol_group_key_list = []
    for i, tol in enumerate(config.globals['frag_mass_tols']):
        tol_group_key = [0] * len(config.globals['frag_mass_tols'])
        tol_group_key[-(i + 1)] = 1
        tol_group_key_list.append(tuple(tol_group_key))
    #Set index as spec_id, '0.2' -> '0.7', combo level
    prediction_df.set_index(['spec_id'] + config.globals['frag_mass_tols'], inplace=True)
    #Tol list indices are sorted backwards: 0.7 predictions come before 0.2 in spec group.
    prediction_df.sort_index(level=['spec_id'] + config.globals['frag_mass_tols'], inplace=True)
    mass_tol_compar_df = prediction_df[['seq', 'combo level']]
    spec_groups = mass_tol_compar_df.groupby(level='spec_id')

    ##Single process
    #tol_match_array_list = []
    #print_percent_progress_fn = partial(
    #    utils.print_percent_progress_singlethreaded, 
    #    procedure_str='mass tolerance comparison progress: ', 
    #    one_percent_total_count=one_percent_number_specs
    #)
    #child_initialize(
    #    spec_groups, 
    #    config.globals['frag_mass_tols'], 
    #    tol_group_key_list, 
    #    print_percent_progress_fn
    #)
    #utils.verbose_print('performing mass tolerance comparison')
    #for spec_id in spec_list:
    #    tol_match_array_list.append(make_mass_tol_match_array(spec_id))

    #Multiprocess
    print_percent_progress_fn = partial(
        utils.print_percent_progress_multithreaded, 
        procedure_str='Mass tolerance comparison progress: ', 
        one_percent_total_count=one_percent_number_specs, 
        cores=config.globals['cpus']
    )
    multiprocessing_pool = Pool(
        config.globals['cpus'], 
        initializer=child_initialize, 
        initargs=(
            spec_groups, 
            config.globals['frag_mass_tols'], 
            tol_group_key_list, 
            print_percent_progress_fn
        )
    )
    utils.verbose_print('Performing mass tolerance comparison')
    tol_match_array_list=multiprocessing_pool.map(
        make_mass_tol_match_array, spec_list
    )
    multiprocessing_pool.close()
    multiprocessing_pool.join()

    tol_match_cols = [tol + ' seq match' for tol in config.globals['frag_mass_tols']]
    tol_match_df = pd.DataFrame(
        np.fliplr(np.concatenate(tol_match_array_list)), 
        index=prediction_df.index, 
        columns=tol_match_cols
    )
    prediction_df = pd.concat([prediction_df, tol_match_df], axis=1)
    prediction_df.drop(['combo level'], axis=1, inplace=True)
    prediction_df.reset_index(inplace=True)
    prediction_df.set_index(config.globals['is_alg_names'] + ['spec_id'], inplace=True)
    prediction_df.sort_index(level=['spec_id'] + config.globals['is_alg_names'], inplace=True)

    return prediction_df

def child_initialize(
    _spec_groups, 
    _frag_mass_tols, 
    _tol_group_key_list, 
    _print_percent_progress_fn
):

    global spec_groups, frag_mass_tols, tol_group_key_list, print_percent_progress_fn

    spec_groups = _spec_groups
    frag_mass_tols = _frag_mass_tols
    tol_group_key_list = _tol_group_key_list
    print_percent_progress_fn = _print_percent_progress_fn

def make_mass_tol_match_array(spec_id):

    print_percent_progress_fn()

    spec_group_df = spec_groups.get_group(spec_id)
    spec_group_df.index = spec_group_df.index.droplevel('spec_id')
    tol_match_array = np.zeros([len(spec_group_df), len(frag_mass_tols)])
    tol_groups = spec_group_df.groupby(level=frag_mass_tols)

    for first_tol_group_key_index, first_tol_group_key in enumerate(tol_group_key_list[:-1]):
        try:
            first_tol_group_df = tol_groups.get_group(first_tol_group_key)
            try:
                first_tol_group_first_row_position_in_spec_group = spec_group_df.index.get_loc(
                    first_tol_group_key
                ).start
            #Certain tols may have only one prediction.
            except AttributeError:
                first_tol_group_first_row_position_in_spec_group = spec_group_df.index.get_loc(
                    first_tol_group_key
                )

            for first_prediction_index, first_prediction_row in enumerate(
                first_tol_group_df.values
            ):
                first_prediction_seq = first_prediction_row[0]
                first_prediction_combo_level = first_prediction_row[1]

                #Compare seqs to those from each subsequent tol.
                for second_tol_group_key_index, second_tol_group_key in enumerate(
                    tol_group_key_list[first_tol_group_key_index + 1:]
                ):
                    try:
                        second_tol_group_df = tol_groups.get_group(second_tol_group_key)

                        #For the first row of tol group df, 
                        #find that row's position in spec group df.
                        try:
                            second_tol_group_first_row_position_in_spec_group = \
                                spec_group_df.index.get_loc(second_tol_group_key).start
                        #The tol index may correspond to only one prediction instead of multiple.
                        except AttributeError:
                            second_tol_group_first_row_position_in_spec_group = \
                                spec_group_df.index.get_loc(second_tol_group_key)

                        for second_prediction_index, second_prediction_row in enumerate(
                            second_tol_group_df.values
                        ):
                            second_prediction_seq = second_prediction_row[0]
                            second_prediction_combo_level = second_prediction_row[1]

                            if first_prediction_seq in second_prediction_seq:
                                #If combo level > current highest level for tol comparison, 
                                #as recorded in tol_match_array
                                if first_prediction_combo_level > tol_match_array[
                                    first_tol_group_first_row_position_in_spec_group + \
                                        first_prediction_index,
                                    first_tol_group_key_index + second_tol_group_key_index + 1
                                ]:
                                    tol_match_array[
                                        first_tol_group_first_row_position_in_spec_group + \
                                            first_prediction_index, 
                                        first_tol_group_key_index + second_tol_group_key_index + 1
                                    ] = second_prediction_combo_level

                            if second_prediction_seq in first_prediction_seq:
                                if second_prediction_combo_level > tol_match_array[
                                    second_tol_group_first_row_position_in_spec_group + \
                                        second_prediction_index, 
                                    first_tol_group_key_index
                                ]:
                                    tol_match_array[
                                        second_tol_group_first_row_position_in_spec_group + \
                                            second_prediction_index, 
                                        first_tol_group_key_index
                                    ] = first_prediction_combo_level

                    #Certain tols may not have predictions.
                    except KeyError:
                        pass

        #Certain tols may not have predictions.
        except KeyError:
            pass

    return tol_match_array
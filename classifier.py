''' Sequence accuracy classification model '''

import datetime
import matplotlib
# Set backend to make image files on server
matplotlib.use('Agg')
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os.path
import numpy as np
import pandas as pd
import pickle as pkl
import sklearn as sk
import sys
import warnings
warnings.filterwarnings('ignore')

from collections import Counter, OrderedDict
from functools import partial
from multiprocessing import Pool
from os.path import join, basename
from scipy.stats import norm
from sklearn.cluster import Birch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if 'postnovo' in sys.modules:
    import postnovo.config as config
    import postnovo.dbsearch as dbsearch
    import postnovo.utils as utils
else:
    import config
    import dbsearch
    import utils

total_ref_len = 0

def classify(prediction_df=None, input_df_dict=None):
    utils.verbose_print()
    if config.globals['mode'] in ['train', 'test', 'optimize']:
        prediction_df, ref_correspondence_df, db_search_ref = find_target_accuracy(prediction_df)
        with open(os.path.join(config.globals['iodir'], 'ref_correspondence_df.pkl'), 'wb') as f:
            pkl.dump(ref_correspondence_df, f, 2)
        with open(os.path.join(config.globals['iodir'], 'db_search_ref.pkl'), 'wb') as f:
            pkl.dump(db_search_ref, f, 2)
        #with open(os.path.join(config.globals['iodir'], 'ref_correspondence_df.pkl'), 'rb') as f:
        #    ref_correspondence_df = pkl.load(f)
        #with open(os.path.join(config.globals['iodir'], 'db_search_ref.pkl'), 'rb') as f:
        #    db_search_ref = pkl.load(f)

    utils.verbose_print('formatting data for compatability with model')
    prediction_df = standardize_prediction_df_cols(prediction_df)
    utils.save_pkl_objects(config.globals['iodir'], **{'prediction_df': prediction_df})
    #prediction_df = utils.load_pkl_objects(config.globals['iodir'], 'prediction_df')

    if config.globals['mode'] == 'predict':
        reported_prediction_df = make_predictions(prediction_df)
        reported_prediction_df.to_csv(os.path.join(config.globals['iodir'], 'best_predictions.csv'))
        #reported_prediction_df = pd.read_csv(os.path.join(config.globals['iodir'], 'best_predictions.csv'), header=0)

        reported_prediction_df = reported_prediction_df[
            reported_prediction_df['probability'] >= config.globals['min_prob']
        ]

        df = reported_prediction_df.reset_index()
        if config.psm_fp_list:
            df = dbsearch.merge_predictions(df)
        mass_grouped_df = dbsearch.group_predictions(df)
        retained_seq_dict = dbsearch.lengthen_seqs(mass_grouped_df)
        dbsearch.make_fasta(retained_seq_dict)

    elif config.globals['mode'] == 'test':
        reported_prediction_df = make_predictions(prediction_df, input_df_dict, db_search_ref)
        reported_prediction_df = reported_prediction_df.reset_index().merge(
            ref_correspondence_df.reset_index(), 
            how='left', 
            on=(
                config.is_alg_col_names 
                + ['scan'] 
                + config.globals['frag_mass_tols'] 
                + ['is longest consensus', 'is top rank consensus']
            )
        )
        reported_prediction_df.set_index('scan', inplace = True)
        reported_cols_in_order = []
        for reported_df_col in config.reported_df_cols:
            if reported_df_col in reported_prediction_df.columns:
                reported_cols_in_order.append(reported_df_col)
        reported_prediction_df = reported_prediction_df.reindex_axis(reported_cols_in_order, axis = 1)
        reported_prediction_df.to_csv(os.path.join(config.globals['iodir'], 'best_predictions.csv'))

        # Loop through each predic
    
    elif config.globals['mode'] in ['train', 'optimize']:
        
        utils.verbose_print('updating training database')
        training_df = update_training_data(prediction_df)
        #training_df = pd.read_csv(
        #    os.path.join(config.data_dir, 'training_df.csv'),
        #    header=0,
        #    index_col=config.is_alg_col_names
        #)
        # REMOVE
        #training_df = utils.load_pkl_objects(config.data_dir, 'training_df')

        forest_dict = make_training_forests(training_df)
        utils.save_pkl_objects(config.data_dir, **{'forest_dict': forest_dict})
        #forest_dict = utils.load_pkl_objects(config.data_dir, 'forest_dict')

def find_seqs_in_paired_seqs(seq_list1, seq_list2):
    seq_pairs = list(zip(seq_list1, seq_list2))
    matches = []
    for seq_pair in seq_pairs:
        if seq_pair[0] in seq_pair[1]:
            matches.append(1)
        else:
            matches.append(0)
    return matches

def find_target_accuracy(prediction_df):

    utils.verbose_print('loading', basename(config.globals['db_search_fp']))
    db_search_ref = load_db_search_ref_file(config.globals['db_search_fp'])
    utils.verbose_print('loading', basename(config.globals['ref_fasta_fp']))
    fasta_ref = load_fasta_ref_file(config.globals['ref_fasta_fp'])

    #Determine the number of amino acids in the reference seqs.
    for ref_seq in fasta_ref:
        total_ref_len += len(ref_seq)

    utils.verbose_print('finding sequence matches to database search reference')

    prediction_df.reset_index(inplace=True)
    comparison_df = prediction_df.merge(db_search_ref, how='left', on='scan')
    #Null entries exist for scans with a de novo but not a db search sequence.
    prediction_df['scan has db search PSM'] = comparison_df['ref seq'].notnull().astype(int)
    comparison_df['ref seq'][comparison_df['ref seq'].isnull()] = ''
    #Add a column recording whether de novo seq is in db search reference (1 if true, 0 if false).
    prediction_df['de novo seq matches db search seq'] = find_seqs_in_paired_seqs(
        comparison_df['seq'].tolist(),
        comparison_df['ref seq'].tolist()
    )
    #Add column for the length of the de novo seq.
    prediction_df['predict len'] = comparison_df['seq'].apply(len)

    utils.verbose_print(
        'finding de novo sequence matches to fasta reference for scans lacking database search PSM'
    )

    #Determine the number of de novo seqs meeting the length criterion without ref matches.
    no_db_search_psm_df = prediction_df[prediction_df['scan has db search PSM'] == 0]
    no_db_search_psm_df = no_db_search_psm_df[no_db_search_psm_df['seq'].apply(len) >= config.min_ref_match_len]
    unique_long_denovo_seqs = list(set(no_db_search_psm_df['seq']))

    utils.verbose_print('finding minimum de novo sequence length to uniquely match fasta reference')
    #config.min_ref_match_len = find_min_seq_len(fasta_ref = fasta_ref, cores = config.globals['cpus'])
    one_percent_number_denovo_seqs = len(unique_long_denovo_seqs) / 100 / config.globals['cpus']

    #Match de novo to the reference fasta file.
    # Multiprocess
    multiprocessing_pool = Pool(config.globals['cpus'])
    print_percent_progress_fn = partial(
        utils.print_percent_progress_multithreaded, 
        procedure_str='reference sequence matching progress: ', 
        one_percent_total_count=one_percent_number_denovo_seqs, 
        cores=config.globals['cpus']
    )
    single_var_match_seq_to_fasta_ref = partial(
        match_seq_to_fasta_ref, 
        fasta_ref=fasta_ref, 
        print_percent_progress_fn=print_percent_progress_fn
    )
    #Return a 1 if there is a match, 0 if there isn't.
    fasta_matches = multiprocessing_pool.map(
        single_var_match_seq_to_fasta_ref, unique_long_denovo_seqs
    )
    multiprocessing_pool.close()
    multiprocessing_pool.join()

    fasta_match_dict = dict(zip(unique_long_denovo_seqs, fasta_matches))
    single_var_get_match_from_dict = partial(get_match_from_dict, match_dict=fasta_match_dict)
    no_db_search_psm_df['correct de novo seq not found in db search'] = \
        no_db_search_psm_df['seq'].apply(single_var_get_match_from_dict)
    prediction_df = prediction_df.merge(
        no_db_search_psm_df['correct de novo seq not found in db search'].to_frame(), 
        left_index=True, 
        right_index=True, 
        how='left'
    )
    prediction_df['correct de novo seq not found in db search'].fillna(0, inplace = True)

    #Count as correct all de novo seqs that matched either the db search results or the ref fasta.
    prediction_df['ref match'] = prediction_df['de novo seq matches db search seq'] + \
        prediction_df['correct de novo seq not found in db search']
    prediction_df.set_index(
        config.is_alg_col_names + 
        ['scan'] + 
        config.globals['frag_mass_tols'] + 
        ['is longest consensus', 'is top rank consensus'], 
        inplace=True
    )
    #Store information on de novo seq accuracy in a separate structure.
    ref_correspondence_df = pd.concat(
        [
            prediction_df['scan has db search PSM'], 
            prediction_df['de novo seq matches db search seq'], 
            prediction_df['correct de novo seq not found in db search']
        ], 
        axis=1
    )
    prediction_df.drop(
        [
            'scan has db search PSM', 
            'de novo seq matches db search seq', 
            'correct de novo seq not found in db search'
        ], 
        axis=1, 
        inplace=True
    )
    prediction_df = prediction_df.reset_index().set_index(config.is_alg_col_names + ['scan'])

    return prediction_df, ref_correspondence_df, db_search_ref

def get_match_from_dict(seq, match_dict):
    return match_dict[seq]

def load_db_search_ref_file(db_search_ref_file_path):
    db_search_ref_df = pd.read_csv(db_search_ref_file_path, '\t')
    # Proteome Discoverer PSM table
    try:
        db_search_ref_df = pd.concat([db_search_ref_df['First Scan'],
                         db_search_ref_df['Annotated Sequence'],
                         db_search_ref_df['Percolator q-Value'],
                         db_search_ref_df['PSM Ambiguity']], 1)
        # Scans may have multiple PSMs -- one is "Selected" while the others are "Rejected": keep the "Selected" peptide
        db_search_ref_df.sort_values(['First Scan', 'PSM Ambiguity'], ascending = [True, False], inplace = True)
        db_search_ref_df.drop_duplicates(subset = 'First Scan', inplace = True)
        db_search_ref_df.drop('PSM Ambiguity', axis = 1, inplace = True)
        db_search_ref_df.columns = ['scan', 'ref seq', 'fdr']
    # Other ref source, in which the columns must be 1. scan, 2. seq, 3. FDR
    except KeyError:
        db_search_ref_df.columns = ['scan', 'ref seq', 'fdr']
        db_search_ref_df.drop_duplicates(subset = 'scan', inplace = True)

    db_search_ref_df = db_search_ref_df[db_search_ref_df['fdr'] <= config.max_fdr]
    db_search_ref_df['ref seq'] = db_search_ref_df['ref seq'].apply(lambda seq: seq.upper())
    db_search_ref_df['ref seq'] = db_search_ref_df['ref seq'].apply(lambda seq: seq.replace('I', 'L'))

    return db_search_ref_df

def load_fasta_ref_file(fasta_ref_file_path):

    with open(fasta_ref_file_path) as f:
        lines = f.readlines()

    fasta_ref = []
    for line in lines:
        if line[0] == '>':
            next_seq_in_next_line = True
        elif line != '\n':
            if next_seq_in_next_line:
                fasta_ref.append(line.strip().replace('I', 'L'))
                next_seq_in_next_line = False
            else:
                fasta_ref[-1] += line.strip().replace('I', 'L')

    return fasta_ref

def match_seq_to_fasta_ref(denovo_seq, fasta_ref, print_percent_progress_fn):

    print_percent_progress_fn()

    for fasta_seq in fasta_ref:
        if denovo_seq in fasta_seq:
            return 1
    return 0

def standardize_prediction_df_cols(prediction_df):

    prediction_df.drop('is top rank single alg', inplace=True)
    min_retention_time = prediction_df['retention time'].min()
    max_retention_time = prediction_df['retention time'].max()
    prediction_df['retention time'] = \
        (prediction_df['retention time'] - min_retention_time) / \
        (max_retention_time - min_retention_time)
    prediction_df.sort_index(1, inplace=True)

    return prediction_df

def update_training_data(prediction_df):

    prediction_df['timestamp'] = str(datetime.datetime.now()).split('.')[0]
    prediction_df.reset_index(inplace = True)
    try:
        training_df = pd.read_csv(join(config.data_dir, 'training_df.csv'))
        training_df = pd.concat([training_df, prediction_df])
    except (FileNotFoundError, OSError) as e:
        training_df = prediction_df
    training_df.set_index(['timestamp', 'scan'], inplace = True)
    training_df.to_csv(join(config.data_dir, 'training_df.csv'))
    training_df.reset_index(inplace=True)
    training_df.set_index(config.is_alg_col_names, inplace=True)

    return training_df

def make_predictions(prediction_df, input_df_dict=None, db_search_ref=None):

    def retrieve_single_alg_score_accuracy_df(alg, frag_analyzer):
        '''
        Calculate the accuracy of individual algorithms' de novo sequences.
        '''

        if alg == 'novor':
            score_col_name = 'avg aa score'
        elif alg == 'pn':
            score_col_name = 'rank score'
        elif alg == 'deepnovo':
            score_col_name = 'avg aa score'

        #Consider the de novo sequences predicted at standard fragment mass tolerances.
        if frag_analyzer == 'Trap':
            frag_mass_tol = '0.5'
        elif frag_analyzer == 'FT':
            frag_mass_tol = '0.05'

        single_alg_df = input_df_dict[alg][frag_mass_tol]
        single_alg_df = single_alg_df.groupby(single_alg_df.index.get_level_values(0)).first()
        single_alg_df = single_alg_df[['seq', score_col_name]]
        single_alg_df.reset_index(inplace=True)
        #Record the lengths of predictions and references to calculate amino acid level statistics.
        single_alg_df['predict len'] = single_alg_df['seq'].apply(lambda x: len(x))
        single_alg_df = single_alg_df[single_alg_df['predict len'] >= config.globals['min_len']]
        single_alg_df = single_alg_df.merge(db_search_ref, how='left', on='scan')
        single_alg_df['ref seq'][single_alg_df['ref seq'].isnull()] = ''
        single_alg_df['ref match'] = find_seqs_in_paired_seqs(
            single_alg_df['seq'].tolist(), 
            single_alg_df['ref seq'].tolist()
        )

        single_alg_df = single_alg_df.set_index('scan')[
            [score_col_name, 'ref match', 'predict len']
        ]
        single_alg_df.columns = ['score', 'ref match', 'predict len']

        return single_alg_df

    # Full comparison plots
    # Make a reported prediction df that goes down to score = 0
    # Retrieve the three top single-alg predictions
    # Make a new df with probability cols and paired ref match cols
    # Use the df in tailored precision-recall and precision-yield fns

    # REMOVE
    print(config.data_dir, flush=True)
    forest_dict = utils.load_pkl_objects(config.data_dir, 'forest_dict')

    #Run Postnovo model for each combination of algorithms (single alg seqs, consensus seqs).
    prediction_df['probability'] = np.nan
    for multiindex_key in config.is_alg_col_multiindex_keys:
        # REMOVE
        print(str(multiindex_key), flush=True)
        alg_group = tuple([alg for i, alg in enumerate(config.globals['algs']) if multiindex_key[i]])
        # REMOVE
        print(alg_group, flush=True)

        alg_group_data = prediction_df.xs(multiindex_key)
        if config.globals['mode'] == 'predict':
            # Remove the cols that are not features in ANY rf
            alg_group_data.drop(
                ['seq', 'probability', 'measured mass', 'mass error'], 
                axis=1, 
                inplace=True
            )
        elif config.globals['mode'] == 'test':                
            # Remove the cols that are not features in ANY rf
            alg_group_data.drop(
                ['seq', 'ref match', 'probability', 'measured mass', 'mass error'], 
                axis=1, 
                inplace=True
            )
        #Remove empty cols that are not features in the appropriate rf.
        alg_group_data.dropna(1, inplace=True)
        forest_dict[alg_group].n_jobs = config.globals['cpus']
        # REMOVE
        print(alg_group_data.columns, flush=True)
        probabilities = forest_dict[alg_group].predict_proba(alg_group_data.as_matrix())[:, 1]

        #Add the predicted probabilities for the alg combination to the full table of results.
        prediction_df.loc[multiindex_key, 'probability'] = probabilities

        # UNCOMMENT
        #if config.globals['mode'] == 'test':
        #    utils.verbose_print('making', '_'.join(alg_group), 'test plots')
        #    #plot_roc_curve(accuracy_labels, probabilities, alg_group, alg_group_data)
        #    postnovo_alg_combo_df = prediction_df.xs(multiindex_key)
        #    postnovo_alg_combo_df = postnovo_alg_combo_df.reset_index().set_index('scan')[['probability', 'ref match']]
        #    max_probabilities = postnovo_alg_combo_df.groupby(level='scan')['probability'].transform(max)
        #    postnovo_alg_combo_df = postnovo_alg_combo_df[postnovo_alg_combo_df['probability'] == max_probabilities][['probability', 'ref match']]
        #    postnovo_alg_combo_df = postnovo_alg_combo_df.groupby(level='scan').first()

        #    plot_precision_recall_curve(postnovo_alg_combo_df, alg_score_accuracy_df_dict)
        #    plot_precision_yield_curve(postnovo_alg_combo_df, alg_score_accuracy_df_dict, len(db_search_ref))

    prediction_df = prediction_df.reset_index().set_index('scan')

    if config.globals['max_total_sacrifice']:
        # Recover the longest seq prediction that fulfills the score sacrifice conditions
        # First, filter to those predictions with prob score above <sacrifice_floor>
        above_sac_floor_df = prediction_df[prediction_df['probability'] >= config.globals['sacrifice_floor']]
        # Loop through the predictions for each spectrum
        above_sac_floor_scan_dfs = [df for _, df in above_sac_floor_df.groupby(level='scan')]

        one_percent_number_scans_above_sac_floor = \
            len(above_sac_floor_scan_dfs) / 100 / config.globals['cpus']
        max_total_sacrifice = config.globals['max_total_sacrifice']
        sacrifice_extension_ratio = config.globals['max_sacrifice_per_percent_extension'] * 100

        ## Single-threaded
        #for scan_df in above_sac_floor_scan_dfs:
        #    one_percent_number_scans_above_sac_floor * config.globals['cpus']
        #    print_percent_progress_fn = partial(
        #        utils.print_percent_progress_multithreaded,
        #        procedure_str='Score-length tradeoff progress: ',
        #        one_percent_total_count=one_percent_number_scans_above_sac_floor
        #        )
        #initialize_workers(
        #    print_percent_progress_fn, 
        #    max_total_sacrifice, 
        #    sacrifice_extension_ratio
        #    )
        #    reported_above_sac_floor_scan_dfs = []
        #    for scan_df in above_sac_floor_scan_dfs:
        #        reported_above_sac_floor_scan_dfs.append(do_score_sacrifice_extension(scan_df))

        # Multiprocessing
        print_percent_progress_fn = partial(
            utils.print_percent_progress_multithreaded,
            procedure_str='Score-length tradeoff progress: ',
            one_percent_total_count=one_percent_number_scans_above_sac_floor,
            cores=config.globals['cpus']
        )
        mp_pool = multiprocessing.Pool(
            config.globals['cpus'], 
            initializer=initialize_workers, 
            initargs=(
                print_percent_progress_fn, 
                max_total_sacrifice, 
                sacrifice_extension_ratio
            )
        )
        reported_above_sac_floor_scan_dfs = mp_pool.map(
            do_score_sacrifice_extension, above_sac_floor_scan_dfs
        )
        mp_pool.close()
        mp_pool.join()
        reported_above_sac_floor_df = pd.concat(reported_above_sac_floor_scan_dfs)

        # Find the best predictions from below the sacrifice floor
        predictions_below_floor_df = pd.concat(
            [scan_df for _, scan_df in prediction_df.groupby(level='scan')
             if (scan_df['probability'] <= config.globals['sacrifice_floor']).all()]
        )
        max_probabilities = \
            predictions_below_floor_df.groupby(level='scan')['probability'].transform(max)
        best_predictions_below_floor_df = predictions_below_floor_df[
            predictions_below_floor_df['probability'] == max_probabilities
        ]
        reported_below_sac_floor_df = \
            best_predictions_below_floor_df.groupby(level='scan').first()

        # Concatenate the data from above and below the floor
        reported_prediction_df = pd.concat([
            reported_above_sac_floor_df, 
            reported_below_sac_floor_df
        ]).sort_index()
    else:
        max_probabilities = prediction_df.groupby(level='scan')['probability'].transform(max)
        best_prediction_df = prediction_df[prediction_df['probability'] == max_probabilities]
        best_prediction_df = best_prediction_df.groupby(level='scan').first()
        reported_prediction_df = best_prediction_df

    #Make plots from test data.
    if config.globals['mode'] == 'test':
        #Prepare single-alg "raw" de novo data for precision-recall and precision-yield plots.
        alg_score_accuracy_df_dict = OrderedDict().fromkeys(alg_group)
        for alg in alg_group:
            alg_score_accuracy_df_dict[alg] = retrieve_single_alg_score_accuracy_df(
                alg, config.globals['frag_analyzer']
            )

        plot_precision_recall_curve(
            reported_prediction_df[['probability', 'ref match']], 
            alg_score_accuracy_df_dict, 
            all_postnovo_predictions=True
        )
        plot_precision_yield_curve(
            reported_prediction_df[['probability', 'ref match', 'ref fraction']], 
            alg_score_accuracy_df_dict, 
            len(db_search_ref), 
            all_postnovo_predictions=True
        )

        plot_aa_fraction_recall_curve(
            reported_prediction_df[['probability', 'ref match', 'predict len']], 
            alg_score_accuracy_df_dict
        )

    reported_cols_in_order = []
    for reported_df_col in config.reported_df_cols:
        if reported_df_col in reported_prediction_df.columns:
            reported_cols_in_order.append(reported_df_col)
    reported_prediction_df = reported_prediction_df.reindex_axis(reported_cols_in_order, axis = 1)

    return reported_prediction_df

def initialize_workers(
    _print_percent_progress_fn, 
    _max_total_sacrifice, 
    _sacrifice_extension_ratio
    ):

    global print_percent_progress_fn, max_total_sacrifice, sacrifice_extension_ratio

    print_percent_progress_fn = _print_percent_progress_fn
    max_total_sacrifice = _max_total_sacrifice
    sacrifice_extension_ratio = _sacrifice_extension_ratio

    return

def do_score_sacrifice_extension(scan_df):

    scan_df.sort_values('probability', ascending=False, inplace=True)
    probs = scan_df['probability'].tolist()
    highest_prob = probs[0]
    lower_prob_seqs = scan_df['seq'].tolist()[1:]
    current_longest_seq = scan_df.iloc[0]['seq']
    current_prob = highest_prob
    longest_row_index = 0
    for i, seq in enumerate(lower_prob_seqs):
        if len(seq) > len(current_longest_seq):
            # The longer seq must contain the shorter, higher-prob seq
            if current_longest_seq in seq:
                lower_prob = probs[i + 1]
                # The longer seq must be within <max_total_sacrifice> score of highest score
                if highest_prob - lower_prob <= max_total_sacrifice:
                    # Finally, the loss in score per percent extension must meet the criterion
                    length_weighted_max_sacrifice = sacrifice_extension_ratio * \
                        np.sum(np.reciprocal(np.arange(
                            len(current_longest_seq) + 1, len(seq) + 1, dtype=np.float16
                            )))
                    if current_prob - lower_prob <= length_weighted_max_sacrifice:
                        longest_row_index = i + 1
                        current_longest_seq = seq
                        current_prob = lower_prob

    return scan_df.iloc[[longest_row_index]]

def make_training_forests(training_df):

    train_target_arr_dict = make_train_target_arr_dict(training_df)
    
    if config.globals['mode'] == 'train':
        forest_dict = make_forest_dict(train_target_arr_dict, config.rf_default_params)

        ## REMOVE
        for alg_key in forest_dict:
            data_train_split, data_validation_split, target_train_split, target_validation_split =\
                train_test_split(train_target_arr_dict[alg_key]['train'], train_target_arr_dict[alg_key]['target'], stratify = train_target_arr_dict[alg_key]['target'])
        #    #plot_feature_importances(forest_dict[alg_key], alg_key, train_target_arr_dict[alg_key]['feature_names'])
            plot_binned_feature_importances(forest_dict[alg_key], alg_key, train_target_arr_dict[alg_key]['feature_names'])
        #    plot_errors(data_train_split, data_validation_split, target_train_split, target_validation_split, alg_key)

    elif config.globals['mode'] == 'optimize':
        utils.verbose_print('optimizing random forest parameters')
        optimized_params = optimize_model(train_target_arr_dict)
        forest_dict = make_forest_dict(train_target_arr_dict, optimized_params)

    return forest_dict

def make_train_target_arr_dict(training_df):

    training_df.sort_index(inplace = True)
    model_keys_used = []
    train_target_arr_dict = {}
    for multiindex in config.is_alg_col_multiindex_keys:
        print(str(multiindex), flush=True)
        model_key = tuple([alg for i, alg in enumerate(config.globals['algs']) if multiindex[i]])
        print(model_key, flush=True)
        model_keys_used.append(model_key)
        train_target_arr_dict[model_keys_used[-1]] = {}.fromkeys(['train', 'target'])
        try:
            alg_group_df = training_df.xs(multiindex).reset_index().set_index(['scan', 'seq'])
            alg_group_df.dropna(1, inplace = True)
            train_columns = alg_group_df.columns.tolist()
            for c in config.is_alg_col_names + ['ref match', 'mass error', 'measured mass', 'timestamp']:
                train_columns.remove(c)
            print(train_columns)
            train_target_arr_dict[model_key]['train'] = alg_group_df.as_matrix(train_columns)
            train_target_arr_dict[model_key]['target'] = alg_group_df['ref match'].tolist()
            train_target_arr_dict[model_key]['feature_names'] = train_columns
        except KeyError:
            print(str(model_keys_used[-1]) + ' predictions were not found')

    return train_target_arr_dict

def make_forest_dict(train_target_arr_dict, rf_params):

    forest_dict = {}.fromkeys(train_target_arr_dict)
    for alg_key in forest_dict:
        if len(alg_key) > 1:
            utils.verbose_print('making random forest for', '-'.join(alg_key), 'consensus sequences')
        else:
            utils.verbose_print('making random forest for', alg_key[0], 'sequences')

        train_data = train_target_arr_dict[alg_key]['train']
        target_data = train_target_arr_dict[alg_key]['target']
        forest = RandomForestClassifier(n_estimators = config.rf_n_estimators,
                                        max_depth = rf_params[alg_key]['max_depth'],
                                        max_features = rf_params[alg_key]['max_features'],
                                        oob_score = True,
                                        n_jobs = config.globals['cpus'])
        forest.fit(train_data, target_data)
        forest_dict[alg_key] = forest

    return forest_dict

def optimize_model(train_target_arr_dict):

    optimized_params = {}
    for alg_key in train_target_arr_dict:
        optimized_params[alg_key] = {}

        data_train_split, data_validation_split, target_train_split, target_validation_split =\
            train_test_split(train_target_arr_dict[alg_key]['train'], train_target_arr_dict[alg_key]['target'], stratify = train_target_arr_dict[alg_key]['target'])
        forest_grid = GridSearchCV(RandomForestClassifier(n_estimators = config.rf_n_estimators, oob_score = True),
                                   {'max_features': ['sqrt', None], 'max_depth': [depth for depth in range(11, 20)]},
                                   n_jobs = config.globals['cpus'])
        forest_grid.fit(data_train_split, target_train_split)
        optimized_forest = forest_grid.best_estimator_
        optimized_params[alg_key]['max_depth'] = optimized_forest.max_depth
        utils.verbose_print(alg_key, 'optimized max depth:', optimized_forest.max_depth)
        optimized_params[alg_key]['max_features'] = optimized_forest.max_features
        utils.verbose_print(alg_key, 'optimized max features:', optimized_forest.max_features)

        plot_feature_importances(optimized_forest, alg_key, train_target_arr_dict[alg_key]['feature_names'])
        plot_binned_feature_importances(optimized_forest, alg_key, train_target_arr_dict[alg_key]['feature_names'])
        plot_errors(data_train_split, data_validation_split, target_train_split, target_validation_split, alg_key)

    return optimized_params

def plot_feature_importances(forest, alg_key, feature_names):
    if len(alg_key) > 1:
        utils.verbose_print('plotting feature importances for', '-'.join(alg_key), 'consensus sequences')
    else:
        utils.verbose_print('plotting feature importances for', alg_key[0], 'sequences')

    importances = forest.feature_importances_
    feature_std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis = 0)
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots()
    ax.set_title('Feature importances')
    x = np.arange(len(importances))
    ax.bar(left = x, height = importances[indices], color = 'r', yerr = feature_std[indices], width = 0.9, align = 'center')
    ax.set_xticks(x)
    labels = np.array(feature_names)[indices]
    ax.set_xticklabels(labels, rotation = -45, ha = 'left')
    ax.set_xlim([-1, len(importances)])
    ax.set_ylim(ymin = 0)
    fig.set_tight_layout(True)

    alg_key_str = '_'.join(alg_key)
    save_path = join(config.globals['iodir'], alg_key_str + '_feature_importances.pdf')
    fig.savefig(save_path, bbox_inches = 'tight')

def plot_binned_feature_importances(forest, alg_key, feature_names):
    if len(alg_key) > 1:
        utils.verbose_print('plotting feature importances for', '-'.join(alg_key), 'consensus sequences')
    else:
        utils.verbose_print('plotting feature importances for', alg_key[0], 'sequences')

    feature_importances = forest.feature_importances_
    feature_group_importances = []
    feature_group_stds = []
    for feature_group, features in config.feature_groups.items():
        feature_group_importance = 0.0
        feature_group_var = 0.0
        for i, feature_name in enumerate(feature_names):
            if feature_name in features:
                feature_group_importance += feature_importances[i]
                feature_group_var += np.var(
                    [tree.feature_importances_[i] for tree in forest.estimators_], axis = 0)
        feature_group_importances.append(feature_group_importance)
        feature_group_stds.append(np.sqrt(feature_group_var))

    feature_group_importances = np.array(feature_group_importances)
    feature_group_stds = np.array(feature_group_stds)
    indices = np.argsort(feature_group_importances)[::-1]

    fig, ax = plt.subplots()
    ax.set_title('Binned feature importances')
    x = np.arange(len(feature_group_importances))
    ax.bar(left = x, height = feature_group_importances[indices], color = 'r', yerr = feature_group_stds[indices], width = 0.9, align = 'center')
    ax.set_xticks(x)
    labels = np.array(list(config.feature_groups))[indices]
    ax.set_xticklabels(labels, rotation = -45, ha = 'left')
    ax.set_xlim([-1, len(feature_group_importances)])
    ax.set_ylim(ymin = 0)
    fig.set_tight_layout(True)

    alg_key_str = '_'.join(alg_key)
    save_path = join(config.globals['iodir'], alg_key_str + '_binned_feature_importances.pdf')
    fig.savefig(save_path, bbox_inches = 'tight')

def plot_errors(data_train_split, data_validation_split, target_train_split, target_validation_split, alg_key):
    if len(alg_key) > 1:
        utils.verbose_print('plotting errors vs tree size for', '-'.join(alg_key), 'consensus sequences')
    else:
        utils.verbose_print('plotting errors vs tree size for', alg_key[0], 'sequences')

    ensemble_clfs = [
        #('max_features=\'sqrt\'',
        # RandomForestClassifier(warm_start = True, max_features = 'sqrt', oob_score = True, max_depth = 15, n_jobs = config.globals['cpus'], random_state = 1)),
        ('max_features=None',
         RandomForestClassifier(warm_start = True, max_features = None, oob_score = True, max_depth = 15, n_jobs = config.globals['cpus'], random_state = 1))
    ]

    oob_errors = OrderedDict((label, []) for label, _ in ensemble_clfs)
    #validation_errors = OrderedDict((label, []) for label, _ in ensemble_clfs)
    min_estimators = 10
    max_estimators = 500

    for label, clf in ensemble_clfs:
        for tree_number in range(min_estimators, max_estimators + 1, 100):
            clf.set_params(n_estimators = tree_number)
            clf.fit(data_train_split, target_train_split)

            oob_error = 1 - clf.oob_score_
            oob_errors[label].append((tree_number, oob_error))

            #validation_error = 1 - clf.score(data_validation_split, target_validation_split)
            #validation_errors[label].append((tree_number, validation_error))

    fig, ax1 = plt.subplots()
    for label, oob_error in oob_errors.items():
        xs, ys = zip(*oob_error)
        ax1.plot(xs, ys, label = 'oob error: ' + label)
    #for label, validation_error in validation_errors.items():
    #    xs, ys = zip(*validation_error)
    #    ax1.plot(xs, ys, label = 'validation error: ' + label)

    ax1.set_xlim(min_estimators, max_estimators)
    ax1.set_xlabel('n_estimators')
    ax1.set_ylabel('error rate')
    ax1.legend(loc = 'upper right')
    fig.set_tight_layout(True)

    alg_key_str = '_'.join(alg_key)
    save_path = join(config.globals['iodir'], alg_key_str + '_error.pdf')
    fig.savefig(save_path, bbox_inches = 'tight')

def plot_roc_curve(accuracy_labels, probabilities, alg_group, alg_group_data):

    false_positive_rate, true_positive_rate, thresholds = roc_curve(accuracy_labels, probabilities, pos_label = 1)
    model_auc = roc_auc_score(accuracy_labels, probabilities)

    alg_scores_dict = {}
    for alg in alg_group:
        if alg == 'novor':
            alg_scores_dict[alg] = alg_group_data['avg novor aa score']
        elif alg == 'pn':
            alg_scores_dict[alg] = alg_group_data['rank score']

    alg_roc_dict = {}
    alg_auc_dict = {}
    for alg in alg_scores_dict:
        alg_roc_dict[alg] = roc_curve(accuracy_labels, alg_scores_dict[alg], pos_label = 1)
        alg_auc_dict[alg] = roc_auc_score(accuracy_labels, alg_scores_dict[alg])

    fig, ax = plt.subplots()

    model_line_collection = colorline(false_positive_rate, true_positive_rate, thresholds)
    plt.colorbar(model_line_collection, label = 'moving threshold:\nrandom forest probability or\nde novo algorithm score percentile')
    annotation_x = false_positive_rate[len(false_positive_rate) // 2]
    annotation_y = true_positive_rate[len(true_positive_rate) // 2]
    plt.annotate('random forest\nauc = ' + str(round(model_auc, 2)),
                 xy = (annotation_x, annotation_y),
                 xycoords='data',
                 xytext = (annotation_x + 50, annotation_y - 50),
                 textcoords = 'offset pixels',
                 arrowprops = dict(facecolor = 'black', shrink = 0.01, width = 1, headwidth = 6),
                 horizontalalignment = 'left', verticalalignment = 'top',
                 )

    arrow_position = 3
    for alg in alg_roc_dict:
        alg_fpr = alg_roc_dict[alg][0]
        alg_tpr = alg_roc_dict[alg][1]
        alg_thresh = alg_roc_dict[alg][2].argsort() / alg_roc_dict[alg][2].size
        annotation_x = alg_fpr[len(alg_fpr) // arrow_position]
        annotation_y = alg_tpr[len(alg_tpr) // arrow_position]
        arrow_position -= 1
        colorline(alg_fpr, alg_tpr, alg_thresh)
        plt.annotate(alg + '\nauc = ' + str(round(alg_auc_dict[alg], 2)),
                     xy = (annotation_x, annotation_y),
                     xycoords='data',
                     xytext = (annotation_x + 50, annotation_y - 50),
                     textcoords = 'offset pixels',
                     arrowprops = dict(facecolor = 'black', shrink = 0.01, width = 1, headwidth = 6),
                     horizontalalignment = 'left', verticalalignment = 'top',
                     )

    plt.plot([0, 1], [0, 1], linestyle = '--', c = 'black')

    if len(alg_group) == 1:
        plt.title('roc curve: ' + alg_group[0] + ' sequences')
    else:
        plt.title('roc curve: ' + '-'.join(alg_group) + ' consensus sequences')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('false positive rate = ' + r'$\frac{F_p}{F_p + T_n}$')
    plt.ylabel('true positive rate = ' + r'$\frac{T_p}{T_p + F_n}$')
    plt.tight_layout(True)

    save_path = join(config.globals['iodir'], '_'.join(alg_group) + '_roc.pdf')
    fig.savefig(save_path, bbox_inches = 'tight')

def plot_aa_fraction_recall_curve(
    postnovo_alg_combo_df, 
    alg_score_accuracy_df_dict
):

    #Sort seqs by probability.
    #Loop through predict lens, ref matches.
    #Calculate running sum of predict lens if ref match == 1.

    aa_recalls = []
    cumulative_len = 0
    for _, ref_match, predict_len in sorted(zip(
        postnovo_alg_combo_df['probability'].tolist(), 
        postnovo_alg_combo_df['ref match'].tolist(), 
        postnovo_alg_combo_df['predict len'].tolist()
    ), lambda t: t[0]):
        if ref_match:
            cumulative_len += predict_len
        aa_recalls.append(cumulative_len / total_ref_len)
    
    alg_aa_recall_dict = dict()
    for alg, alg_score_accuracy_df in alg_score_accuracy_df_dict.items():
        cumulative_len = 0
        for _, ref_match, predict_len in sorted(zip(
            alg_score_accuracy_df['score'].tolist(), 
            alg_score_accuracy_df['ref match'].tolist(), 
            alg_score_accuracy_df['predict len'].tolist()
        ), lambda t: t[0]):
            if ref_match:
                cumulative_len += predict_len
            aa_recalls.append(cumulative_len / total_ref_len)
        alg_aa_recall_dict[alg] = aa_recalls

def plot_precision_recall_curve(
    postnovo_alg_combo_df, 
    alg_score_accuracy_df_dict, 
    all_postnovo_predictions=False
):

    true_positive_rate, recall, thresholds = precision_recall_curve(
        postnovo_alg_combo_df['ref match'].tolist(), 
        postnovo_alg_combo_df['probability'].tolist(), 
        pos_label = 1
    )

    alg_pr_dict = {}
    alg_auc_dict = {}
    for alg, alg_score_accuracy_df in alg_score_accuracy_df_dict.items():
        alg_pr_dict[alg] = precision_recall_curve(
            alg_score_accuracy_df['ref match'].tolist(), 
            alg_score_accuracy_df['score'].tolist(), 
            pos_label = 1
        )
        alg_auc_dict[alg] = average_precision_score(
            alg_score_accuracy_df['ref match'].tolist(), 
            alg_score_accuracy_df['score'].tolist()
        )

    fig, ax = plt.subplots()

    line_collection = colorline(recall, true_positive_rate, thresholds)
    cb = plt.colorbar(
        line_collection, 
        label=(
            'moving threshold:\npostnovo probability score or\n'
            'de novo algorithm score percentile'
        )
    )
    #annotation_x = recall[int(len(recall) / 1.2)]
    #annotation_y = true_positive_rate[int(len(true_positive_rate) / 1.2)]
    #plt.annotate('random forest\nauc = ' + str(round(model_auc, 2)),
    #             xy = (annotation_x, annotation_y),
    #             xycoords = 'data',
    #             xytext = (annotation_x + 25, annotation_y + 25),
    #             textcoords = 'offset pixels',
    #             arrowprops = dict(facecolor = 'black', shrink = 0.01, width = 1, headwidth = 6),
    #             horizontalalignment = 'right', verticalalignment = 'bottom',
    #             )

    # Tabulate the probability score cutoffs required for different levels of precision
    if all_postnovo_predictions:
        next_precision = 0.5
        precision_index = 0
        score_cutoffs = []
        for prob_index, prob in enumerate(thresholds):
            while true_positive_rate[precision_index] > next_precision:
                score_cutoffs.append(thresholds[prob_index])
                next_precision += 0.05
            precision_index += 1
        cutoff_precision_table = pd.DataFrame()
        cutoff_precision_table['precision'] = np.arange(0.5, 1, 0.05)
        cutoff_precision_table['score cutoff'] = score_cutoffs
        cutoff_precision_table.to_csv(
            os.path.join(config.globals['iodir'], 'cutoff_precision.tsv'), 
            sep='\t', 
            index=False
        )

    #COMMENT
    arrow_position = 1.2

    for alg in alg_pr_dict:
        alg_recall = alg_pr_dict[alg][1]
        alg_tpr = alg_pr_dict[alg][0]
       
        #alg_thresh = alg_pr_dict[alg][2].argsort() / alg_pr_dict[alg][2].size
        alg_thresh = alg_pr_dict[alg][2]

        #COMMENT
        annotation_x = alg_recall[int(len(alg_recall) / arrow_position)]
        annotation_y = alg_tpr[int(len(alg_tpr) / arrow_position)]

        if alg == 'novor':
            line_collection = colorline(alg_recall, alg_tpr, alg_thresh, norm=plt.Normalize(0, 100))
            tick_locs = [0, 20, 40, 60, 80, 100]
        elif alg == 'pn':
            line_collection = colorline(alg_recall, alg_tpr, alg_thresh, norm=plt.Normalize(-10, 15))
            tick_locs = [-10, -5, 0, 5, 10, 15]
            #line_collection = colorline(alg_recall, alg_tpr, alg_thresh, norm=plt.Normalize(alg_thresh[0], alg_thresh[-1]))
            #tick_locs = []
            #tick_interval = (alg_thresh[-1] - alg_thresh[0]) / 5
            #for i in range(6):
            #    tick_locs.append(alg_thresh[0] + i * tick_interval)
        elif alg == 'deepnovo':
            line_collection = colorline(alg_recall, alg_tpr, alg_thresh, norm=plt.Normalize(0, 1))
            tick_locs = [0, 0.2, 0.4, 0.6, 0.8, 1]
        #plt.colorbar(line_collection, ticks=tick_locs)

        #COMMENT
        plt.annotate(
            alg + '\nauc = ' + str(round(alg_auc_dict[alg], 2)), 
            xy = (annotation_x, annotation_y), 
            xycoords = 'data', 
            xytext = (annotation_x - 25, annotation_y - 25), 
            textcoords = 'offset pixels', 
            arrowprops = dict(facecolor = 'black', shrink = 0.01, width = 1, headwidth = 6), 
            horizontalalignment = 'right', verticalalignment = 'top',
        )

    #if all_postnovo_predictions:
    #    plt.title('all postnovo predictions')
    #else:
    #    alg_group = list(alg_score_accuracy_df_dict.keys())
    #    if len(alg_score_accuracy_df_dict) == 1:
    #        # e.g., title is 'novor sequences'
    #        plt.title(alg_group[0] + ' sequences')
    #    else:
    #        # e.g., title is 'novor-pn consensus sequences'
    #        plt.title('-'.join(alg_group) + ' consensus sequences')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('recall')
    #plt.xlabel('recall (true positive rate) = ' + r'$\frac{T_p}{T_p + F_n}$')
    plt.ylabel('precision')
    #plt.ylabel('precision = ' + r'$\frac{T_p}{T_p + F_p}$')
    plt.tight_layout(True)

    if all_postnovo_predictions:
        save_path = join(config.globals['iodir'], 'full_precision_recall.pdf')
    else:
        save_path = join(config.globals['iodir'], '_'.join(alg_group) + '_precision_recall.pdf')
    fig.savefig(save_path, bbox_inches = 'tight')

    return

def plot_precision_yield_curve(
    postnovo_alg_combo_df, 
    alg_score_accuracy_df_dict, 
    db_search_yield, 
    all_postnovo_predictions=False
    ):

    fig, ax = plt.subplots()
    x_min = 1
    x_max = 0
    plt.ylim([0, 1])
    plt.xlabel('sequence yield')
    plt.ylabel('precision')
    #plt.ylabel('precision = ' + r'$\frac{T_p}{T_p + F_p}$')

    db_search_x = db_search_yield
    db_search_y = 1 - config.max_fdr

    alg_group = list(alg_score_accuracy_df_dict.keys())

    # Plot postnovo results
    # Get the numbers from 1 to N postnovo seq predictions
    sample_size_list = list(range(1, len(postnovo_alg_combo_df) + 1))
    precision_list = make_precision_list(postnovo_alg_combo_df, 'probability')

    x = sample_size_list[::100]
    y = precision_list[::100]
    # Color is probability score (ranges from 0 to 1)
    z = sorted(postnovo_alg_combo_df['probability'].tolist(), reverse=True)[::100]
    line_collection = colorline(x, y, z)

    plt.colorbar(
        line_collection, 
        label=(
            'moving threshold:\npostnovo probability score or\n'
            'de novo algorithm score percentile'
            )
        )
    #annotation_x = x[len(x) // 2]
    #annotation_y = y[len(y) // 2]
    #plt.annotate(
    #    '_'.join(alg_group) + '\n' + 'random forest', 
    #    xy=(annotation_x, annotation_y), 
    #    xycoords='data', 
    #    xytext=(25, 25), 
    #    textcoords = 'offset pixels', 
    #    arrowprops = dict(facecolor='black', shrink=0.01, width=1, headwidth=6), 
    #    horizontalalignment='left', 
    #    verticalalignment='bottom'
    #    )

    # Push the x-axis maximum to the highest yield
    # Yield can theoretically be higher than db search yield, 
    # if more de novo seqs are found than db search PSMs
    if x[-1] > x_max:
        x_max = x[-1]

    #COMMENT
    arrow_position = 2.5

    for alg, alg_score_accuracy_df in alg_score_accuracy_df_dict.items():

        sample_size_list = list(range(1, len(alg_score_accuracy_df) + 1))
        precision_list = make_precision_list(alg_score_accuracy_df, 'score')

        x = sample_size_list[::100]
        y = precision_list[::100]

        # Color is raw score
        z = sorted(alg_score_accuracy_df['score'].tolist(), reverse=True)[::100]
        ## Color is score percentile (0 to 1)
        #score_count = len(alg_score_accuracy_df)
        #z = [i / score_count for i in range(score_count, 0, -100)]
        #line_collection = colorline(x, y, z)
        if alg == 'novor':
            line_collection = colorline(x, y, z, norm=plt.Normalize(0, 100))
            tick_locs = [0, 20, 40, 60, 80, 100]
        elif alg == 'pn':
            line_collection = colorline(x, y, z, norm=plt.Normalize(-10, 15))
            tick_locs = [-10, -5, 0, 5, 10, 15]
            #line_collection = colorline(x, y, z, norm=plt.Normalize(z[-1], z[0]))
            #tick_locs = []
            #tick_interval = (z[0] - z[-1]) / 5
            #for i in range(6):
            #    tick_locs.append(z[-1] + i * tick_interval)
        elif alg == 'deepnovo':
            line_collection = colorline(x, y, z, norm=plt.Normalize(0, 1))
            tick_locs = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        #plt.colorbar(
        #    line_collection, 
        #    ticks=tick_locs, 
        #    #label = (
        #    #    'moving threshold:\npostnovo probability score or\n'
        #    #    'de novo algorithm score percentile'
        #    #    )
        #    )

        #COMMENT
        annotation_x = x[int(len(x) / arrow_position)]
        annotation_y = y[int(len(y) / arrow_position)]
        arrow_position -= 0.5
        plt.annotate(
            alg,
            xy=(annotation_x, annotation_y),
            xycoords='data',
            xytext = (-25, -25),
            textcoords='offset pixels',
            arrowprops=dict(facecolor='black', shrink=0.01, width=1, headwidth=6),
            horizontalalignment='right', 
            verticalalignment='top'
            )

        if x[-1] > x_max:
            x_max = x[-1]

    ax.plot(db_search_x, db_search_y, color='r', marker='*', markersize=10)
    if db_search_x > x_max:
        x_max = db_search_x + 1000
    plt.xlim([x_min, x_max])

    #if all_postnovo_predictions:
    #    plt.title('all postnovo predictions')
    #else:
    #    alg_group = list(alg_score_accuracy_df_dict.keys())
    #    if len(alg_score_accuracy_df_dict) == 1:
    #        # e.g., title is 'novor sequences'
    #        plt.title(alg_group[0] + ' sequences')
    #    else:
    #        # e.g., title is 'novor-pn consensus sequences'
    #        plt.title('-'.join(alg_group) + ' consensus sequences')

    plt.tight_layout(True)
    if all_postnovo_predictions:
        save_path = join(config.globals['iodir'], 'full_precision_yield.pdf')
    else:
        save_path = join(config.globals['iodir'], '_'.join(alg_group) + '_precision_yield.pdf')
    fig.savefig(save_path, bbox_inches = 'tight')
    plt.close()

    return

def colorline(x, y, z, cmap='jet', norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):

    z = np.asarray(z)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    line_collection = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(line_collection)
    return line_collection

def make_precision_list(df, sort_col):
    # Make a list of cumulative precision values as successively lower-scoring seqs are considered

    df.sort_values(sort_col, ascending=False, inplace=True)
    ref_matches = df['ref match'].tolist()
    cumulative_ref_matches = [ref_matches[0]]
    precision_list = [cumulative_ref_matches[0] / 1]
    for i, ref_match in enumerate(ref_matches[1:]):
        cumulative_ref_match_sum = cumulative_ref_matches[-1] + ref_match
        cumulative_ref_matches.append(cumulative_ref_match_sum)
        precision_list.append(cumulative_ref_match_sum / (i + 2))

    return precision_list
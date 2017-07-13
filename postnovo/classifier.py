''' Sequence accuracy classification model '''

import difflib
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib
# Set backend to make image files on server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import datetime
import warnings
warnings.filterwarnings('ignore')
import os.path

import postnovo.config as config
import postnovo.utils as utils

#import config
#import utils

from functools import partial
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import Birch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from os.path import join, basename
from multiprocessing import Pool, current_process
from collections import Counter
from scipy.stats import norm

multiprocessing_seq_matching_count = 0

def classify(prediction_df = None):
    utils.verbose_print()

    if config.mode[0] in ['train', 'test', 'optimize']:
        prediction_df, ref_correspondence_df, db_search_ref = find_target_accuracy(prediction_df)

    utils.verbose_print('formatting data for compatability with model')
    prediction_df = standardize_prediction_df_cols(prediction_df)
    utils.save_pkl_objects(config.iodir[0], **{'prediction_df': prediction_df})
    #prediction_df = utils.load_pkl_objects(config.iodir[0], 'prediction_df')

    if config.mode[0] == 'predict':
        reported_prediction_df = make_predictions(prediction_df)
        reported_prediction_df.to_csv(os.path.join(config.iodir[0], 'best_predictions.csv'))
        make_fasta(reported_prediction_df)

    elif config.mode[0] == 'test':
        reported_prediction_df = make_predictions(prediction_df, db_search_ref)
        reported_prediction_df = reported_prediction_df.reset_index().\
            merge(ref_correspondence_df.reset_index(),
                  how = 'left',
                  on = config.is_alg_col_names + ['scan'] + config.frag_mass_tols + ['is longest consensus', 'is top rank consensus'])
        reported_prediction_df.set_index('scan', inplace = True)
        reported_cols_in_order = []
        for reported_df_col in config.reported_df_cols:
            if reported_df_col in reported_prediction_df.columns:
                reported_cols_in_order.append(reported_df_col)
        reported_prediction_df = reported_prediction_df.reindex_axis(reported_cols_in_order, axis = 1)
        reported_prediction_df.to_csv(os.path.join(config.iodir[0], 'best_predictions.csv'))
    
    elif config.mode[0] in ['train', 'optimize']:
        
        #subsampled_df = subsample_training_data(prediction_df)
        #utils.save_pkl_objects(config.iodir[0], **{'subsampled_df': subsampled_df})
        #subsampled_df = utils.load_pkl_objects(config.iodir[0], 'subsampled_df')

        utils.verbose_print('updating training database')
        training_df = update_training_data(prediction_df)
        #training_df = utils.load_pkl_objects(config.data_dir, 'training_df')

        forest_dict = make_training_forests(training_df)
        utils.save_pkl_objects(config.data_dir, **{'forest_dict': forest_dict})
        #forest_dict = utils.load_pkl_objects(config.data_dir, 'forest_dict')


def find_target_accuracy(prediction_df):
    utils.verbose_print('loading', basename(config.db_search_ref_file[0]))
    db_search_ref = load_db_search_ref_file(config.db_search_ref_file[0])
    utils.verbose_print('loading', basename(config.db_search_ref_file[0]))
    fasta_ref = load_fasta_ref_file(config.db_search_ref_file[0])

    utils.verbose_print('finding sequence matches to database search reference')

    prediction_df.reset_index(inplace = True)
    comparison_df = prediction_df.merge(db_search_ref, how = 'left', on = 'scan')
    prediction_df['scan has db search PSM'] = comparison_df['ref seq'].notnull().astype(int)
    comparison_df['ref seq'][comparison_df['ref seq'].isnull()] = ''
    denovo_seqs = comparison_df['seq'].tolist()
    psm_seqs = comparison_df['ref seq'].tolist()
    seq_pairs = list(zip(denovo_seqs, psm_seqs))
    matches = []
    for seq_pair in seq_pairs:
        if seq_pair[0] in seq_pair[1]:
            matches.append(1)
        else:
            matches.append(0)
    prediction_df['de novo seq matches db search seq'] = matches

    utils.verbose_print('finding de novo sequence matches to fasta reference for scans lacking database search PSM')

    no_db_search_psm_df = prediction_df[prediction_df['scan has db search PSM'] == 0]
    no_db_search_psm_df = no_db_search_psm_df[no_db_search_psm_df['seq'].apply(len) >= config.min_ref_match_len]
    unique_long_denovo_seqs = list(set(no_db_search_psm_df['seq']))

    utils.verbose_print('finding minimum de novo sequence length to uniquely match fasta reference')
    #config.min_ref_match_len = find_min_seq_len(fasta_ref = fasta_ref, cores = config.cores[0])
    one_percent_number_denovo_seqs = len(unique_long_denovo_seqs) / 100 / config.cores[0]

    multiprocessing_pool = Pool(config.cores[0])
    single_var_match_seq = partial(match_seq_to_fasta_ref, fasta_ref = fasta_ref,
                                   one_percent_number_denovo_seqs = one_percent_number_denovo_seqs, cores = config.cores[0])
    fasta_matches = multiprocessing_pool.map(single_var_match_seq, unique_long_denovo_seqs)
    multiprocessing_pool.close()
    multiprocessing_pool.join()

    fasta_match_dict = dict(zip(unique_long_denovo_seqs, fasta_matches))
    single_var_get_match_from_dict = partial(get_match_from_dict, match_dict = fasta_match_dict)
    no_db_search_psm_df['correct de novo seq not found in db search'] = no_db_search_psm_df['seq'].apply(single_var_get_match_from_dict)
    prediction_df = prediction_df.merge(no_db_search_psm_df['correct de novo seq not found in db search'].to_frame(),
                                        left_index = True, right_index = True, how = 'left')
    prediction_df['correct de novo seq not found in db search'].fillna(0, inplace = True)

    prediction_df['ref match'] = prediction_df['de novo seq matches db search seq'] +\
        prediction_df['correct de novo seq not found in db search']
    prediction_df.set_index(config.is_alg_col_names + ['scan'] + config.frag_mass_tols + ['is longest consensus', 'is top rank consensus'],
                            inplace = True)
    ref_correspondence_df = pd.concat([prediction_df['scan has db search PSM'],
                                       prediction_df['de novo seq matches db search seq'],
                                       prediction_df['correct de novo seq not found in db search']], 1)
    prediction_df.drop(['scan has db search PSM',
                        'de novo seq matches db search seq',
                        'correct de novo seq not found in db search'], axis = 1, inplace = True)
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

    db_search_ref_df = db_search_ref_df[db_search_ref_df['fdr'] <= config.min_fdr]
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

def match_seq_to_fasta_ref(denovo_seq, fasta_ref, one_percent_number_denovo_seqs, cores):

    if current_process()._identity[0] % cores == 1:
        global multiprocessing_seq_matching_count
        multiprocessing_seq_matching_count += 1
        if int(multiprocessing_seq_matching_count % one_percent_number_denovo_seqs) == 0:
            percent_complete = int(multiprocessing_seq_matching_count / one_percent_number_denovo_seqs)
            if percent_complete <= 100:
                utils.verbose_print_over_same_line('reference sequence matching progress: ' + str(percent_complete) + '%')

    for fasta_seq in fasta_ref:
        if denovo_seq in fasta_seq:
            return 1
    return 0

def standardize_prediction_df_cols(prediction_df):

    # No longer necessary with preset mass tol list
    #for accepted_mass_tol in config.accepted_mass_tols:
    #    if accepted_mass_tol not in prediction_df.columns:
    #        prediction_df[accepted_mass_tol] = 0
    prediction_df.drop('is top rank single alg', inplace = True)
    min_retention_time = prediction_df['retention time'].min()
    max_retention_time = prediction_df['retention time'].max()
    prediction_df['retention time'] = (prediction_df['retention time'] - min_retention_time) / (max_retention_time - min_retention_time)
    prediction_df.sort_index(1, inplace = True)
    return prediction_df

def subsample_training_data(prediction_df_orig):

    subsample_row_indices = []
    prediction_df_orig['unique index'] = [i for i in range(prediction_df_orig.shape[0])]
    prediction_df_orig.set_index('unique index', append = True, inplace = True)
    prediction_df = prediction_df_orig.copy()
    prediction_df.drop(['is top rank single alg', 'seq'], axis = 1, inplace = True)

    accuracy_bins = sorted([round(x / config.subsample_accuracy_divisor, 1) for x in range(config.subsample_accuracy_divisor)], reverse = True)
    
    lower = config.subsample_accuracy_distribution_lower_bound
    upper = config.subsample_accuracy_distribution_upper_bound
    weight_bins = np.arange(lower, upper + (upper - lower) / config.subsample_accuracy_divisor, (upper - lower) / config.subsample_accuracy_divisor)
    sigma = config.subsample_accuracy_distribution_sigma
    mu_location = config.subsample_accuracy_distribution_mu_location
    accuracy_weights = (norm.cdf(weight_bins[1: 1 + config.subsample_accuracy_divisor], loc = mu_location, scale = sigma)
                        - norm.cdf(weight_bins[: config.subsample_accuracy_divisor], loc = mu_location, scale = sigma))\
                            / (norm.cdf(upper, loc = mu_location, scale = sigma)
                               - norm.cdf(lower, loc = mu_location, scale = sigma))
    accuracy_subsample_weights = {acc_bin: weight for acc_bin, weight in zip(accuracy_bins, accuracy_weights)}
    accuracy_subsample_sizes = {acc_bin: int(weight * config.subsample_size) for acc_bin, weight in accuracy_subsample_weights.items()}
    while sum(accuracy_subsample_sizes.values()) != config.subsample_size:
        accuracy_subsample_sizes[accuracy_bins[0]] += 1

    for multiindex_key in config.is_alg_col_multiindex_keys:
        multiindex_list = list(multiindex_key)
        alg_group_df_key = tuple([alg for i, alg in enumerate(config.alg_list) if multiindex_key[i]])
        if sum(multiindex_key) == 1:
            utils.verbose_print('subsampling', alg_group_df_key[0], 'top-ranking sequences')
        else:
            utils.verbose_print('subsampling', '-'.join(alg_group_df_key), 'consensus sequences')
        alg_group_df = prediction_df.xs(multiindex_key)
        alg_group_unique_index = alg_group_df.index.get_level_values('unique index')
        alg_group_df.reset_index(inplace = True)
        alg_group_df.set_index(['scan'], inplace = True)
        alg_group_df.dropna(1, inplace = True)
        ref_match_col = alg_group_df['ref match'].copy()

        retained_features_target = round(config.clustering_feature_retention_factor_dict[sum(multiindex_key)] / alg_group_df.shape[0], 0)
        if retained_features_target < config.clustering_min_retained_features:
            retained_features_target = config.clustering_min_retained_features
        retained_features_list = []
        retained_feature_count = 0
        for feature in config.features_ordered_by_importance:
            if feature in alg_group_df.columns:
                retained_features_list.append(feature)
                retained_feature_count += 1
            if retained_feature_count == retained_features_target:
                break
        alg_group_df = alg_group_df[retained_features_list]

        if alg_group_df.shape[0] > config.subsample_size:

            pipe = make_pipeline(StandardScaler(),
                                 Birch(threshold = config.clustering_birch_threshold, n_clusters = None))
            cluster_assignments = pipe.fit_predict(alg_group_df.as_matrix())

            cluster_assignment_accuracies = zip(cluster_assignments, ref_match_col)
            sum_cluster_accuracies = {}.fromkeys(cluster_assignments, 0)
            for cluster, acc in cluster_assignment_accuracies:
                sum_cluster_accuracies[cluster] += acc
            cluster_counts = Counter(cluster_assignments)
            mean_cluster_accuracies = {}.fromkeys(sum_cluster_accuracies, 0)
            for cluster in cluster_counts:
                mean_cluster_accuracies[cluster] = min(
                    accuracy_bins,
                    key = lambda accuracy_bin: abs(accuracy_bin - int(
                        sum_cluster_accuracies[cluster] * 10 / cluster_counts[cluster]) / 10))
            ordered_clusters_accuracies = sorted(
                mean_cluster_accuracies.items(), key = lambda cluster_accuracy_tuple: cluster_accuracy_tuple[1], reverse = True)
            cluster_assignments_row_indices = [(cluster, index) for index, cluster in enumerate(cluster_assignments)]
            cluster_row_indices_dict = {cluster: [] for cluster in mean_cluster_accuracies}
            for cluster, index in cluster_assignments_row_indices:
                cluster_row_indices_dict[cluster].append(index)
            cluster_accuracies_ordered_by_cluster = [cluster_acc_tuple[1] for cluster_acc_tuple in
                                                     sorted(ordered_clusters_accuracies, key = lambda cluster_acc_tuple: cluster_acc_tuple[0])]
            cluster_accuracies_row_indices = [(x[1], x[0][1]) for x in sorted(
                zip(cluster_row_indices_dict.items(), cluster_accuracies_ordered_by_cluster),
                key = lambda cluster_acc_tuple: cluster_acc_tuple[1], reverse = True)]

            accuracy_row_indices_dict = {acc: [] for acc in cluster_accuracies_ordered_by_cluster}
            for acc_row_indices_tuple in cluster_accuracies_row_indices:
                accuracy_row_indices_dict[acc_row_indices_tuple[0]] += acc_row_indices_tuple[1]

            alg_group_subsample_indices = []
            remaining_subsample_size = config.subsample_size
            remaining_accuracy_bins = [acc_bin for acc_bin in accuracy_bins]
            remaining_accuracy_subsample_sizes = {acc_bin: size for acc_bin, size in accuracy_subsample_sizes.items()}
            loop_remaining_accuracy_bins = [acc_bin for acc_bin in remaining_accuracy_bins]
            while remaining_subsample_size > 0:
                for acc_bin in loop_remaining_accuracy_bins:
                    if acc_bin not in accuracy_row_indices_dict:
                        residual = remaining_accuracy_subsample_sizes[acc_bin]
                        remaining_accuracy_bins.remove(acc_bin)
                        remaining_accuracy_subsample_sizes[acc_bin] = 0
                        remaining_accuracy_subsample_sizes = redistribute_residual_subsample(
                            residual, remaining_accuracy_bins, accuracy_subsample_weights, remaining_accuracy_subsample_sizes)
                    elif remaining_accuracy_subsample_sizes[acc_bin] > len(accuracy_row_indices_dict[acc_bin]):
                        acc_bin_subsample_size = len(accuracy_row_indices_dict[acc_bin])
                        remaining_subsample_size -= acc_bin_subsample_size
                        residual = remaining_accuracy_subsample_sizes[acc_bin] - acc_bin_subsample_size
                        remaining_accuracy_bins.remove(acc_bin)
                        remaining_accuracy_subsample_sizes[acc_bin] = 0
                        alg_group_subsample_indices += accuracy_row_indices_dict[acc_bin]
                        remaining_accuracy_subsample_sizes = redistribute_residual_subsample(
                            residual, remaining_accuracy_bins, accuracy_subsample_weights, remaining_accuracy_subsample_sizes)
                    else:
                        alg_group_subsample_indices += np.random.choice(
                            accuracy_row_indices_dict[acc_bin], remaining_accuracy_subsample_sizes[acc_bin], replace = False).tolist()
                        remaining_subsample_size -= remaining_accuracy_subsample_sizes[acc_bin]
                        remaining_accuracy_subsample_sizes[acc_bin] = 0
                loop_remaining_accuracy_bins = remaining_accuracy_bins

            scan_index = alg_group_df.index
            for i in alg_group_subsample_indices:
                subsample_row_indices.append(tuple(multiindex_list + [scan_index[i], alg_group_unique_index[i]])) 

        else:
            for i, scan in enumerate(alg_group_df.index):
                subsample_row_indices.append(tuple(multiindex_list + [scan, alg_group_unique_index[i]]))

    subsampled_df = prediction_df_orig.loc[sorted(subsample_row_indices)]
    retained_multiindices = subsampled_df.index.names[:-1]
    subsampled_df.reset_index(inplace = True)
    subsampled_df.drop('unique index', axis = 1, inplace = True)
    subsampled_df.set_index(retained_multiindices, inplace = True)

    return subsampled_df

def redistribute_residual_subsample(residual, remaining_accuracy_bins, accuracy_subsample_weights, remaining_accuracy_subsample_sizes):

    residual_of_residual = residual
    sum_remaining_weights = 0
    for acc_bin in remaining_accuracy_bins:
        sum_remaining_weights += accuracy_subsample_weights[acc_bin]
    for acc_bin in remaining_accuracy_bins:
        acc_bin_redistributed_residual = int(residual * accuracy_subsample_weights[acc_bin] / sum_remaining_weights)
        remaining_accuracy_subsample_sizes[acc_bin] += acc_bin_redistributed_residual
        residual_of_residual -= acc_bin_redistributed_residual
    remaining_accuracy_subsample_sizes[remaining_accuracy_bins[0]] += residual_of_residual

    return remaining_accuracy_subsample_sizes

def update_training_data(prediction_df):

    try:
        training_df = utils.load_pkl_objects(config.data_dir, 'training_df')
        training_df = pd.concat([training_df, prediction_df])
    except (FileNotFoundError, OSError) as e:
        training_df = prediction_df
    utils.save_pkl_objects(config.data_dir, **{'training_df': training_df})

    prediction_df_csv = prediction_df.copy()
    prediction_df_csv['timestamp'] = str(datetime.datetime.now()).split('.')[0]
    prediction_df_csv.reset_index(inplace = True)
    try:
        training_df_csv = pd.read_csv(join(config.data_dir, 'training_df.csv'))
        training_df_csv = pd.concat([training_df_csv, prediction_df_csv])
    except (FileNotFoundError, OSError) as e:
        training_df_csv = prediction_df_csv
    training_df_csv.set_index(['timestamp', 'scan'], inplace = True)
    training_df_csv.to_csv(join(config.data_dir, 'training_df.csv'))
    # save prediction df as csv in case training df is too big to open in Excel
    # NEEDS FIXING: KeyError: 'timestamp'
    #prediction_df_csv.set_index(['timestamp', 'scan'], inplace = True)
    #prediction_df_csv.to_csv(join(config.test_dir, 'last_added_training_df.csv'))

    return training_df

def make_predictions(prediction_df, db_search_ref = None):

    forest_dict = utils.load_pkl_objects(config.data_dir, 'forest_dict')

    prediction_df['probability'] = np.nan
    for multiindex_key in config.is_alg_col_multiindex_keys:
        alg_group = tuple([alg for i, alg in enumerate(config.alg_list) if multiindex_key[i]])

        alg_group_data = prediction_df.xs(multiindex_key)
        if config.mode[0] == 'predict':
            alg_group_data.drop(['seq', 'probability', 'measured mass', 'mass error'], axis = 1, inplace = True)
        elif config.mode[0] == 'test':
            accuracy_labels = alg_group_data['ref match'].tolist()
            alg_group_data.drop(['seq', 'ref match', 'probability', 'measured mass', 'mass error'], axis = 1, inplace = True)
        alg_group_data.dropna(1, inplace = True)
        forest_dict[alg_group].n_jobs = config.cores[0]
        probabilities = forest_dict[alg_group].predict_proba(alg_group_data.as_matrix())[:, 1]

        if config.mode[0] == 'test':
            utils.verbose_print('making', '_'.join(alg_group), 'test plots')
            #plot_roc_curve(accuracy_labels, probabilities, alg_group, alg_group_data)
            plot_precision_recall_curve(accuracy_labels, probabilities, alg_group, alg_group_data)

        prediction_df.loc[multiindex_key, 'probability'] = probabilities

    if config.mode[0] == 'test':
        plot_precision_yield(prediction_df, db_search_ref)

    prediction_df = prediction_df.reset_index().set_index('scan')
    max_probabilities = prediction_df.groupby(level = 'scan')['probability'].transform(max)
    best_prediction_df = prediction_df[prediction_df['probability'] == max_probabilities]
    best_prediction_df = best_prediction_df.groupby(level = 'scan').first()
    reported_prediction_df = best_prediction_df[best_prediction_df['probability'] >= config.min_prob[0]]
    
    reported_cols_in_order = []
    for reported_df_col in config.reported_df_cols:
        if reported_df_col in reported_prediction_df.columns:
            reported_cols_in_order.append(reported_df_col)
    reported_prediction_df = reported_prediction_df.reindex_axis(reported_cols_in_order, axis = 1)

    return reported_prediction_df

def make_fasta(reported_prediction_df):
    
    # after assembling fasta seqs, check to make sure no seqs are in any other seqs
    # remove duplicate shorter seqs

    # df with cols: scan, seq, prob, mass, mass error
    reduced_df = reported_prediction_df.reset_index()[
        ['scan', 'probability', 'seq', 'measured mass', 'mass error']]
    # sort by mass
    reduced_df.sort('measured mass', inplace=True)
    # extract mass list, mass error list
    mass_list = reduced_df['measured mass'].apply(float).tolist()
    mass_error_list = reduced_df['mass error'].apply(float).tolist()
    # assign mass 0 to group 0
    current_prelim_mass_group = 0
    prelim_mass_group_list = [current_prelim_mass_group]
    # loop through each mass, to mass n-1
    for i in range(len(mass_list[:-1])):
        # if next mass outside mass error of mass
        if mass_list[i] + mass_error_list[i] < mass_list[i+1] - mass_error_list[i+1]:
            current_prelim_mass_group += 1
        prelim_mass_group_list.append(current_prelim_mass_group)
    # add mass group col to df
    reduced_df['prelim mass group'] = prelim_mass_group_list
    current_final_mass_group = -1
    final_mass_group_list = []
    # min seq similarity = 7/9 rounded down to third digit
    min_seq_similarity = 0.777
    scan_proximity = 300
    # scan proximity bonus = 1/9 rounded down to third digit
    scan_proximity_bonus = 0.111
    # length difference penalty = 1/9 rounded down to third digit
    length_diff_penalty = 0.111
    length_diff_multiplier = 3
    for prelim_mass_group in set(prelim_mass_group_list):
        prelim_mass_group_df = reduced_df[reduced_df['prelim mass group'] == prelim_mass_group]

        #print(prelim_mass_group_df)

        local_final_mass_group_list = [-1] * len(prelim_mass_group_df)
        for first_row_index in range(len(prelim_mass_group_df)):
            first_seq = prelim_mass_group_df['seq'].iloc[first_row_index]
            first_scan = prelim_mass_group_df['scan'].iloc[first_row_index]
            if local_final_mass_group_list[first_row_index] == -1:
                current_final_mass_group += 1

            #print('current final mass group: ' + str(current_final_mass_group))
            #print(list(first_seq))

            for second_row_index in range(first_row_index + 1, len(prelim_mass_group_df)):
                if (local_final_mass_group_list[first_row_index] == -1 or
                    local_final_mass_group_list[second_row_index] == -1):
                    first_seq_list = list(first_seq)
                    second_seq = prelim_mass_group_df['seq'].iloc[second_row_index]
                    second_seq_list = list(second_seq)
                    second_scan = prelim_mass_group_df['scan'].iloc[second_row_index]

                    #print(second_seq_list)

                    if len(first_seq) <= len(second_seq):
                        shorter_seq_list = first_seq_list
                        longer_seq_list = second_seq_list
                    else:
                        shorter_seq_list = second_seq_list
                        longer_seq_list = first_seq_list
                    match_count = 0
                    mismatch_count = 0
                    for aa in shorter_seq_list:
                        try:
                            del(longer_seq_list[longer_seq_list.index(aa)])
                            match_count += 1
                        except ValueError:
                            mismatch_count += 1
                    seq_similarity = match_count / len(shorter_seq_list)
                    if abs(first_scan - second_scan) <= scan_proximity:
                        adjusted_seq_similarity = \
                            (seq_similarity + 
                             scan_proximity_bonus - 
                             (abs(len(first_seq) - len(second_seq)) // length_diff_multiplier * length_diff_penalty))
                    else:
                        adjusted_seq_similarity = \
                            (seq_similarity - 
                             (abs(len(first_seq) - len(second_seq)) // length_diff_multiplier * length_diff_penalty))

                    #print('seq similarity: ' + str(seq_similarity))
                    #print('adjusted seq similarity: ' + str(adjusted_seq_similarity))

                    if adjusted_seq_similarity >= min_seq_similarity:
                        if ((local_final_mass_group_list[first_row_index] == -1) and 
                            (local_final_mass_group_list[second_row_index] == -1)):
                            local_final_mass_group_list[first_row_index] = current_final_mass_group
                            local_final_mass_group_list[second_row_index] = current_final_mass_group
                        else:
                            if local_final_mass_group_list[first_row_index] == -1:
                                local_final_mass_group_list[first_row_index] = \
                                    local_final_mass_group_list[second_row_index]
                            if local_final_mass_group_list[second_row_index] == -1:
                                local_final_mass_group_list[second_row_index] = \
                                    local_final_mass_group_list[first_row_index]
            if local_final_mass_group_list[first_row_index] == -1:
                local_final_mass_group_list[first_row_index] = current_final_mass_group
        final_mass_group_list += local_final_mass_group_list

    #print(str(len(final_mass_group_list)))
    #print(str(len(reduced_df)))

    reduced_df['final mass group'] = final_mass_group_list

    # make preliminary mass groups
    # have a list of final mass groups
    # keep track of the last assigned final group (bigger than each preliminary mass group)
    # loop through group to determine seq similarity
    # seq similarity > 
    # ex. preliminary mass group
    # scan  seq         prob        mass        error       final group
    # 18920	LTVEEAK	    0.610231138	1075.575276	0.004302301 N
    # 24760	LTEDLGGLEK	0.892213125	1075.575276	0.004302301 N+1
    # 18825	VDALTVEEAK	0.574638	1075.576276	0.004302305 N
    # 24856	LTEDLGGLEK	0.819234005	1075.576276	0.004302305 N+1
    # 24953	LTEDLELNK	0.7073737	1075.576276	0.004302305 N+1
    # assign group N to row 0
    # make a list of chars in seq 0: [L, T, V, E, E, A, K]
    # make a list of chars in seq 1: [L, T, E, D, L, G, G, L, E, K]
    # loop through chars in shorter seq (seq 0)
    # search for char in longer seq (seq 1)
    # count matches and mismatches
    # if found in longer seq list, del char
    # at the end of this example: seq 1 list = [D, L, G, G, L], matches = 5, mismatches = 2
    # if matches / # aa in shorter seq >= 0.77 (7/9), add to group N
    # else do not add to a new final group
    # final groups are recorded in corresponding list
    # proceeding to the next comparison
    # seq 0 list: [L, T, V, E, E, A, K]
    # seq 2 list: [V, D, A, L, T, V, E, E, A, K]
    # matches = 7, mismatches = 0 => add to group N
    # ... moving onto second loop starting with second seq:
    # this seq is assigned to group N+1 (last assigned final group + 1)
    # ignore seq comparisons to seqs with final group assignment
    # first comparison: seq 1, seq 3 => 100% match
    # second (last) comparison: seq 1, seq 4
    # seq 1 list: [L, T, E, D, L, G, G, L, E, K]
    # seq 4 list: [L, T, E, D, L, E, L, N, K]
    # matches = 8, mismatches = 1

    # faa file from most probable, longest overlapping seq of each mass group
    # loop through each mass group
    fasta_seq_list = []
    scan_list = []
    mass_list = []
    mass_error_list = []
    min_overlap = 5
    for current_final_mass_group in set(final_mass_group_list):
        # xs mass group
        final_mass_group_df = reduced_df[reduced_df['final mass group'] == current_final_mass_group]
        # sort by prob, highest to lowest
        final_mass_group_df.sort('probability', ascending=False, inplace=True)

        #print('final mass group df:')
        #print(final_mass_group_df)

        # take highest prob seq as faa seq
        final_mass_group_seq_list = final_mass_group_df['seq'].tolist()
        fasta_seq = final_mass_group_seq_list[0]

        #print('fasta seq: ' + fasta_seq)

        # loop through subsequent seqs
        for seq in final_mass_group_seq_list[1:]:
            # if seq is not in faa seq
            if seq not in fasta_seq:
                seq_matcher_obj = difflib.SequenceMatcher(None, fasta_seq, seq)
                # d = difflib.SequenceMatcher(None, faa seq, seq)
                # l = d.get_matching_blocks()
                matching_blocks_list = seq_matcher_obj.get_matching_blocks()
                # for m in l
                for block in matching_blocks_list:
                    # if (m[0] == 0) and (m[1] + m[2] == len(seq))
                    if ((block[0] == 0) and 
                        (block[1] + block[2] == len(seq)) and 
                        (block[2] >= min_overlap)):
                        # this means overlap of seq with left side of faa seq
                        # ex. faa seq = 'bcde', seq = 'abcd'
                        # if m[2] >= min overlap (5)
                        # faa seq = seq[:m[1]] + faa seq
                        fasta_seq = seq[:block[1]] + fasta_seq

                        #print('overlapping seq: ' + seq)
                        #print('updated fasta seq: ' + fasta_seq)

                    # elif (m[1] == 0) and (m[0] + m[2] == len(faa seq)
                    elif ((block[1] == 0) and
                            (block[0] + block[2] == len(fasta_seq)) and
                            (block[2] >= min_overlap)):
                        # this means overlap of seq with right side of faa seq
                        # ex. faa seq = 'abcd', seq = 'bcde'
                        # if m[2] >= min overlap (5)
                        # faa seq = faa seq + seq[m[2]:]
                        fasta_seq = fasta_seq + seq[block[2]:]

                        #print('overlapping seq: ' + seq)
                        #print('updated fasta seq: ' + fasta_seq)

        fasta_seq_list.append(fasta_seq)
        scan_list.append(str(final_mass_group_df['scan'].iloc[0]))
        mass_list.append(str(final_mass_group_df['measured mass'].iloc[0]))
        mass_error_list.append(str(final_mass_group_df['mass error'].iloc[0]))

    # Have a list the length of the number of fasta seqs
    # Initialized with 0's
    # Change to 1's for seqs that are present in other seqs
    # Loop through each fasta seq
    # Compare to every other fasta seq that is not slated for removal
    # Is the first seq in the second?
    # Is the second seq in the first?
    # If one seq is in the other, mark the removal list with 1 at the proper position
    # Loop through the removal list to write to file
    # If 0, write the L/I permuted seqs to file
    removal_list = [0 for i in range(len(fasta_seq_list))]
    for i, first_fasta_seq in enumerate(fasta_seq_list):
        for j in range(i+1, len(fasta_seq_list)):
            second_fasta_seq = fasta_seq_list[j]
            if (first_fasta_seq in second_fasta_seq) and (removal_list[j] == 0):
                
                #print('first scan: ' + scan_list[i])
                #print('first mass: ' + mass_list[i])
                #print('first seq: ' + first_fasta_seq)
                #print('second scan: ' + scan_list[j])
                #print('second mass: ' + mass_list[j])
                #print('second seq: ' + second_fasta_seq)

                removal_list[i] = 1
                break
            elif (second_fasta_seq in first_fasta_seq) and (removal_list[i] == 0):

                #print('first scan: ' + scan_list[i])
                #print('first mass: ' + mass_list[i])
                #print('first seq: ' + first_fasta_seq)
                #print('second scan: ' + scan_list[j])
                #print('second mass: ' + mass_list[j])
                #print('second seq: ' + second_fasta_seq)

                removal_list[j] = 1
                break
                    
    # open faa file
    with open(os.path.join(config.iodir[0], 'postnovo_seqs.faa'), 'w') as fasta_file:
        for i, remove in enumerate(removal_list):
            if not remove:
                fasta_seq = fasta_seq_list[i]
                if len(fasta_seq) >= config.min_blast_query_len:
                    permuted_seqs = make_l_i_permutations(fasta_seq, permuted_seqs = [])[::-1]
                    for j, permuted_seq in enumerate(permuted_seqs):
                        fasta_header = ('>' + 
                                        '(scan)' + scan_list[i] + 
                                        '(l_i_permutation)' + str(j) + 
                                        '(mass)' + mass_list[i] + 
                                        '(mass_error)' + mass_error_list[i] + 
                                        '\n')
                        fasta_file.write(fasta_header)
                        fasta_file.write(permuted_seq + '\n')

def make_l_i_permutations(seq, residue_number = 0, permuted_seqs = None):

    if permuted_seqs == None:
        permuted_seqs = []

    if residue_number == len(seq):
        permuted_seqs.append(seq)
        return permuted_seqs
    else:
        if seq[residue_number] == 'L':
            permuted_seq = seq[: residue_number] + 'I' + seq[residue_number + 1:]
            permuted_seqs = make_l_i_permutations(permuted_seq, residue_number + 1, permuted_seqs)
        permuted_seqs = make_l_i_permutations(seq, residue_number + 1, permuted_seqs)
        return permuted_seqs

def make_training_forests(training_df):

    train_target_arr_dict = make_train_target_arr_dict(training_df)
    
    if config.mode[0] == 'train':
        forest_dict = make_forest_dict(train_target_arr_dict, config.rf_default_params)

        ## REMOVE
        for alg_key in forest_dict:
            data_train_split, data_validation_split, target_train_split, target_validation_split =\
                train_test_split(train_target_arr_dict[alg_key]['train'], train_target_arr_dict[alg_key]['target'], stratify = train_target_arr_dict[alg_key]['target'])
        #    #plot_feature_importances(forest_dict[alg_key], alg_key, train_target_arr_dict[alg_key]['feature_names'])
            plot_binned_feature_importances(forest_dict[alg_key], alg_key, train_target_arr_dict[alg_key]['feature_names'])
        #    plot_errors(data_train_split, data_validation_split, target_train_split, target_validation_split, alg_key)

    elif config.mode[0] == 'optimize':
        utils.verbose_print('optimizing random forest parameters')
        optimized_params = optimize_model(train_target_arr_dict)
        forest_dict = make_forest_dict(train_target_arr_dict, optimized_params)

    return forest_dict

def make_train_target_arr_dict(training_df):

    training_df.sort_index(inplace = True)
    model_keys_used = []
    train_target_arr_dict = {}
    for multiindex in config.is_alg_col_multiindex_keys:
        model_key = tuple([alg for i, alg in enumerate(config.alg_list) if multiindex[i]])
        model_keys_used.append(model_key)
        train_target_arr_dict[model_keys_used[-1]] = {}.fromkeys(['train', 'target'])
        try:
            alg_group_df = training_df.xs(multiindex).reset_index().set_index(['scan', 'seq'])
            alg_group_df.dropna(1, inplace = True)
            train_columns = alg_group_df.columns.tolist()
            train_columns.remove('ref match')
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
                                        n_jobs = config.cores[0])
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
                                   n_jobs = config.cores[0])
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
    save_path = join(config.iodir[0], alg_key_str + '_feature_importances.pdf')
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
    save_path = join(config.iodir[0], alg_key_str + '_binned_feature_importances.pdf')
    fig.savefig(save_path, bbox_inches = 'tight')

def plot_errors(data_train_split, data_validation_split, target_train_split, target_validation_split, alg_key):
    if len(alg_key) > 1:
        utils.verbose_print('plotting errors vs tree size for', '-'.join(alg_key), 'consensus sequences')
    else:
        utils.verbose_print('plotting errors vs tree size for', alg_key[0], 'sequences')

    ensemble_clfs = [
        #('max_features=\'sqrt\'',
        # RandomForestClassifier(warm_start = True, max_features = 'sqrt', oob_score = True, max_depth = 15, n_jobs = config.cores[0], random_state = 1)),
        ('max_features=None',
         RandomForestClassifier(warm_start = True, max_features = None, oob_score = True, max_depth = 15, n_jobs = config.cores[0], random_state = 1))
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
    save_path = join(config.iodir[0], alg_key_str + '_error.pdf')
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

    save_path = join(config.iodir[0], '_'.join(alg_group) + '_roc.pdf')
    fig.savefig(save_path, bbox_inches = 'tight')

def plot_precision_recall_curve(accuracy_labels, probabilities, alg_group, alg_group_data):

    true_positive_rate, recall, thresholds = precision_recall_curve(accuracy_labels, probabilities, pos_label = 1)
    model_auc = average_precision_score(accuracy_labels, probabilities)

    alg_scores_dict = {}
    for alg in alg_group:
        if alg == 'novor':
            alg_scores_dict[alg] = alg_group_data['avg novor aa score']
        elif alg == 'pn':
            alg_scores_dict[alg] = alg_group_data['rank score']

    alg_pr_dict = {}
    alg_auc_dict = {}
    for alg in alg_scores_dict:
        alg_pr_dict[alg] = precision_recall_curve(accuracy_labels, alg_scores_dict[alg], pos_label = 1)
        alg_auc_dict[alg] = average_precision_score(accuracy_labels, alg_scores_dict[alg])

    fig, ax = plt.subplots()

    model_line_collection = colorline(recall, true_positive_rate, thresholds)
    plt.colorbar(model_line_collection, label = 'moving threshold:\nrandom forest probability or\nde novo algorithm score percentile')
    annotation_x = recall[int(len(recall) / 1.2)]
    annotation_y = true_positive_rate[int(len(true_positive_rate) / 1.2)]
    plt.annotate('random forest\nauc = ' + str(round(model_auc, 2)),
                 xy = (annotation_x, annotation_y),
                 xycoords = 'data',
                 xytext = (annotation_x + 25, annotation_y + 25),
                 textcoords = 'offset pixels',
                 arrowprops = dict(facecolor = 'black', shrink = 0.01, width = 1, headwidth = 6),
                 horizontalalignment = 'right', verticalalignment = 'bottom',
                 )

    arrow_position = 1.2
    for alg in alg_pr_dict:
        alg_recall = alg_pr_dict[alg][1]
        alg_tpr = alg_pr_dict[alg][0]
        alg_thresh = alg_pr_dict[alg][2].argsort() / alg_pr_dict[alg][2].size
        annotation_x = alg_recall[int(len(alg_recall) / arrow_position)]
        annotation_y = alg_tpr[int(len(alg_tpr) / arrow_position)]
        colorline(alg_recall, alg_tpr, alg_thresh)
        plt.annotate(alg + '\nauc = ' + str(round(alg_auc_dict[alg], 2)),
                     xy = (annotation_x, annotation_y),
                     xycoords = 'data',
                     xytext = (annotation_x - 25, annotation_y - 25),
                     textcoords = 'offset pixels',
                     arrowprops = dict(facecolor = 'black', shrink = 0.01, width = 1, headwidth = 6),
                     horizontalalignment = 'right', verticalalignment = 'top',
                     )

    if len(alg_group) == 1:
        plt.title('precision-recall curve: ' + alg_group[0] + ' sequences')
    else:
        plt.title('precision-recall curve: ' + '-'.join(alg_group) + ' consensus sequences')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('recall (true positive rate) = ' + r'$\frac{T_p}{T_p + F_n}$')
    plt.ylabel('precision = ' + r'$\frac{T_p}{T_p + F_p}$')
    plt.tight_layout(True)

    save_path = join(config.iodir[0], '_'.join(alg_group) + '_precision_recall.pdf')
    fig.savefig(save_path, bbox_inches = 'tight')

def plot_precision_yield(prediction_df, db_search_ref):

    fig, ax = plt.subplots()
    plt.title('precision vs sequence yield')
    x_min = 1
    x_max = 0
    plt.ylim([0, 1])
    plt.xlabel('sequence yield')
    plt.ylabel('precision = ' + r'$\frac{T_p}{T_p + F_p}$')

    db_search_ref_x = len(db_search_ref)
    db_search_ref_y = 0.95

    for multiindex_key in config.is_alg_col_multiindex_keys:

        fig, ax = plt.subplots()
        plt.title('precision vs sequence yield')
        x_min = 1
        x_max = 0
        plt.ylim([0, 1])
        plt.xlabel('sequence yield')
        plt.ylabel('precision = ' + r'$\frac{T_p}{T_p + F_p}$')

        alg_group = tuple([alg for i, alg in enumerate(config.alg_list) if multiindex_key[i]])
        alg_group_data = prediction_df.xs(multiindex_key)
        max_probabilities = alg_group_data.groupby(level = 'scan')['probability'].transform(max)
        best_alg_group_data = alg_group_data[alg_group_data['probability'] == max_probabilities]
        best_alg_group_data = best_alg_group_data.groupby(level = 'scan').first()

        sample_size_list = list(range(1, len(best_alg_group_data) + 1))
        precision = make_precision_col(best_alg_group_data, 'probability')
        best_alg_group_data['random forest precision'] = precision

        x = sample_size_list[::100]
        y = precision[::100]
        z = (best_alg_group_data['probability'].argsort() / best_alg_group_data['probability'].size).tolist()[::100]
        model_line_collection = colorline(x, y, z)
        plt.colorbar(model_line_collection, label = 'moving threshold:\nrandom forest probability or\nde novo algorithm score percentile')
        annotation_x = x[len(x) // 2]
        annotation_y = y[len(y) // 2]
        plt.annotate('_'.join(alg_group) + '\n' + 'random forest',
                     xy = (annotation_x, annotation_y),
                     xycoords = 'data',
                     xytext = (25, 25),
                     textcoords = 'offset pixels',
                     arrowprops = dict(facecolor = 'black', shrink = 0.01, width = 1, headwidth = 6),
                     horizontalalignment = 'left', verticalalignment = 'bottom',
                     )

        if x[-1] > x_max:
            x_max = x[-1]
            plt.xlim([x_min, x_max])

        arrow_position = 2.5
        for i, alg in enumerate(alg_group):
            if alg == 'novor':
                score_str = 'avg novor aa score'
            elif alg == 'pn':
                score_str = 'rank score'
            precision = make_precision_col(best_alg_group_data, score_str)
            best_alg_group_data[alg + ' precision'] = precision

            x = sample_size_list[::100]
            y = precision[::100]
            z = (best_alg_group_data[score_str].argsort() / best_alg_group_data[score_str].size).tolist()[::100]
            colorline(x, y, z)
            annotation_x = x[int(len(x) / arrow_position)]
            annotation_y = y[int(len(y) / arrow_position)]
            arrow_position -= 1
            plt.annotate('_'.join(alg_group) + '\n' + score_str,
                         xy = (annotation_x, annotation_y),
                         xycoords = 'data',
                         xytext = (-25, -25),
                         textcoords = 'offset pixels',
                         arrowprops = dict(facecolor = 'black', shrink = 0.01, width = 1, headwidth = 6),
                         horizontalalignment = 'right', verticalalignment = 'top'
                         )

            if x[-1] > x_max:
                x_max = x[-1]
                plt.xlim([x_min, x_max])

        ax.plot(db_search_ref_x, db_search_ref_y, color = 'r', marker = '*')
        if db_search_ref_x > x_max:
            x_max = db_search_ref_x + 1000
            plt.xlim([x_min, x_max])

        plt.tight_layout(True)
        save_path = join(config.iodir[0], '_'.join(alg_group) + '_precision_yield.pdf')
        fig.savefig(save_path, bbox_inches = 'tight')
        plt.close()

def colorline(x, y, z, cmap = 'jet', norm = plt.Normalize(0.0, 1.0), linewidth = 3, alpha = 1.0):

    z = np.asarray(z)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    line_collection = mcoll.LineCollection(segments, array = z, cmap = cmap, norm = norm, linewidth = linewidth, alpha = alpha)

    ax = plt.gca()
    ax.add_collection(line_collection)
    return line_collection

def make_precision_col(df, sort_col):
    df.sort_values(sort_col, ascending = False, inplace = True)
    ref_matches = df['ref match']
    cumulative_ref_matches = [ref_matches.iat[0]]
    precision = [cumulative_ref_matches[0] / 1]
    for i, ref_match in enumerate(ref_matches[1:]):
        cumulative_ref_match_sum = cumulative_ref_matches[-1] + ref_match
        cumulative_ref_matches.append(cumulative_ref_match_sum)
        precision.append(cumulative_ref_match_sum / (i + 2))

    return precision